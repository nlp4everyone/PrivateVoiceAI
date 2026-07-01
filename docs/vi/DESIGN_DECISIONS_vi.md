# Quyết định thiết kế

Tài liệu này giải thích lý do đằng sau mỗi lựa chọn kiến trúc quan trọng, kèm phân tích ưu/nhược điểm và các phương án thay thế đã cân nhắc.

---

## 1. Pipeline kết hợp VAD + ASR hai giai đoạn

### Mô tả

Thay vì gửi audio thô trực tiếp tới model ASR, một `VADDeployment` riêng (Pyannote Audio, `pyannote/segmentation-3.0`) phát hiện vùng giọng nói trước. Deployment `Pipeline` cắt audio thành tensor theo từng đoạn, và chỉ những đoạn đó mới được gửi tới `ASRDeployment`.

### Ưu điểm

- **Tiết kiệm tài nguyên** — khoảng lặng, tiếng ồn nền, và audio không phải giọng nói không bao giờ chạm tới model ASR (vốn tốn kém hơn tương đối).
- **Scale tốt hơn cho bản ghi dài/thưa** — file dài nhưng giọng nói thưa thớt chỉ tốn chi phí tỉ lệ với thời lượng giọng nói, không phải toàn bộ thời lượng file.
- **Timestamp sạch hơn** — ASR chỉ cần align từ/đoạn trong một vùng giọng nói đã biết, thay vì phải xử lý cả khoảng lặng xen giữa.

### Nhược điểm

- **Thêm một bước network và một model** — mỗi request phải trả chi phí cho một lần inference VAD trước khi ASR có thể bắt đầu; với clip ngắn, giọng nói dày đặc, điều này tăng latency so với gửi thẳng audio vào ASR.
- **Nguy cơ cắt nhầm ở biên đoạn** — từ nằm sát biên đoạn do VAD phát hiện có thể bị cắt nếu VAD ước lượng biên đoạn thấp hơn thực tế; hiện chưa có padding quanh các đoạn.
- **Thời lượng báo cáo là cả file, không chỉ phần giọng nói** — `estimate_audio_duration()` đo trên audio gốc, có thể gây nhầm lẫn nếu caller kỳ vọng thời lượng "audio đã xử lý".

### Phương án đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Pipeline chỉ có ASR (không VAD) | Đơn giản hơn, nhưng lãng phí GPU compute với audio nhiều khoảng lặng và không hỗ trợ use-case theo đoạn mà branch này hướng tới |
| VAD phía client | Đẩy độ phức tạp sang phía caller; hành vi không nhất quán giữa các client |

---

## 2. Các Ray Serve deployment độc lập theo từng giai đoạn

### Mô tả

`VADDeployment`, `ASRDeployment`, `Pipeline`, và `IngressDeployment` mỗi thành phần là một class `@serve.deployment` riêng biệt, được liên kết trong `bind_app()`, thay vì một class service monolithic duy nhất.

### Ưu điểm

- **Scale độc lập** — `VAD_NUM_REPLICAS` (mặc định `4`) có thể tinh chỉnh riêng với `ASR_NUM_REPLICAS` (mặc định `1`) để phù hợp chi phí tương đối từng giai đoạn; VAD thường rẻ hơn mỗi lần gọi, nên tăng replica giúp hấp thụ request đồng thời mà không cần cấp GPU dư thừa cho model ASR nặng hơn.
- **GPU phân số độc lập** — `VAD_NUM_GPUS=0.1` và `ASR_NUM_GPUS=0.2` cho phép cả hai model chia sẻ một GPU vật lý duy nhất qua cơ chế cấp GPU phân số của Ray, thay vì mỗi model cần một thiết bị riêng.
- **Cô lập lỗi** — một replica của deployment nào đó crash hoặc restart không nhất thiết kéo theo các deployment khác.
- **Ingress chỉ dùng CPU** — `INGRESS_NUM_GPUS=0` nghĩa là tầng xử lý HTTP không bao giờ tranh chấp tài nguyên GPU.

### Nhược điểm

- **Nhiều thành phần hơn để quản lý** — bốn cấu hình deployment cần theo dõi thay vì một; `config.toml` có bốn section (`system`, `asr`, `vad`, `ingress`) cần giữ đồng bộ.
- **Thêm lượt gọi giữa các actor** — `Ingress → Pipeline → VAD` và `Pipeline → ASR` đều là các Ray remote call riêng biệt, mỗi call có overhead serialization/scheduling nhỏ riêng.

---

## 3. Fan-out theo cấp đoạn với gộp batch ASR xuyên request

### Mô tả

`Pipeline` không gửi toàn bộ câu nói tới ASR trong một call. Nó gửi một `ASRDeployment.batched_transcribe_tensors.remote()` cho mỗi đoạn giọng nói phát hiện được và chờ tất cả cùng nhau bằng `asyncio.gather`. Vì `batched_transcribe_tensors` được decorate bằng `@serve.batch`, Ray Serve gộp các call này — bao gồm cả đoạn đến từ những request HTTP đồng thời *khác* — vào cùng một GPU batch.

### Ưu điểm

- **Gộp batch đúng độ hạt** — mức sử dụng GPU được điều khiển bởi tổng số đoạn trên toàn dịch vụ, không phụ thuộc số đoạn tình cờ có trong audio của một caller.
- **Trong suốt với caller** — một request HTTP vẫn nhận về một response gộp duy nhất; việc fan-out/gộp batch hoàn toàn ẩn với bên ngoài.
- **Đánh đổi throughput/latency có thể cấu hình** — `ASR_MAX_BATCH_SIZE` và `ASR_BATCH_WAIT_TIMEOUT_S` điều chỉnh mức độ gộp đoạn trước khi dispatch.

### Nhược điểm

- **Nhiều Ray remote call hơn mỗi request** — file audio có nhiều đoạn giọng nói sẽ sinh ra nhiều call ASR nhỏ thay vì một call lớn, tăng overhead scheduling theo từng đoạn.
- **Không sort-by-length hay tách mixed-batch** — khác với các chiến lược gộp batch nhóm input có kích thước tương đồng để giảm padding, `process_batch_transcription()` ở đây chỉ phân nhóm theo yêu cầu timestamp (có/không), nên một batch vẫn có thể trộn đoạn ngắn và dài, gây lãng phí padding.
- **Chờ đoạn chậm nhất** — `asyncio.gather` trên các call ASR cấp đoạn nghĩa là toàn bộ request chỉ hoàn tất khi mọi đoạn (một số có thể rơi vào các batch khác nhau, lệch thời điểm) đã trả về.

---

## 4. Điều chỉnh offset timestamp

### Mô tả

Vì mỗi đoạn giọng nói được phiên âm độc lập, timestamp từ/đoạn do NeMo trả về tương đối với mốc bắt đầu của waveform (đã cắt) của chính đoạn đó. `Pipeline` cộng lại `VADSegment.start` tương ứng vào mỗi timestamp trước khi trả về response cuối, và đánh lại `id` đoạn tuần tự trên kết quả đã gộp.

### Ưu điểm

- **Timestamp tuyệt đối chính xác** — caller nhận timestamp tương đối theo file audio gốc, đúng như kỳ vọng từ một lần phiên âm duy nhất.
- **Sửa đơn giản, cục bộ** — chỉ một phép cộng cho mỗi timestamp; không cần thay đổi cách NeMo tính alignment.

### Nhược điểm

- **Không kiểm tra tính liên tục giữa các đoạn** — nếu các đoạn VAD chồng lấp hoặc có khoảng hở nhỏ, việc điều chỉnh không cố gắng đối chiếu thời gian giữa các đoạn liền kề; mỗi đoạn được sửa độc lập.
- **Cộng dồn sai số làm tròn** — timestamp VAD được làm tròn 3 chữ số, timestamp ASR cũng được làm tròn riêng 3 chữ số; tổng có thể mang sai số làm tròn nhỏ cộng dồn (dưới mili-giây trong thực tế).

---

## 5. Ray Serve để điều phối deployment

### Mô tả

Toàn bộ đồ thị (`VADDeployment`, `ASRDeployment`, `Pipeline`, `IngressDeployment`) được phục vụ qua **Ray Serve** thay vì một tiến trình FastAPI/Uvicorn đơn thuần gọi model trong cùng tiến trình.

### Ưu điểm

- **Hỗ trợ native cho đồ thị đa deployment** — Ray Serve được thiết kế đúng cho kiểu kết hợp deployment nối tiếp này (`.bind()` deployment này vào constructor của deployment khác).
- **Auto-batching có sẵn cho từng deployment** — `@serve.batch` trên `ASRDeployment` không cần tự quản lý queue.
- **Cấp GPU phân số và không đồng nhất** — các deployment khác nhau có thể yêu cầu tỉ lệ GPU khác nhau trên cùng một GPU vật lý.
- **Dashboard** — Ray Serve dashboard (`:8265`) hiển thị tình trạng replica, throughput request, và lỗi trên cả bốn deployment.

### Nhược điểm

- **Overhead khởi động cao hơn** — khởi tạo Ray cluster cộng với tải hai model (VAD + ASR) chậm sẵn sàng hơn một dịch vụ một model duy nhất.
- **Phức tạp vận hành** — log, replica, vòng đời actor của bốn deployment cần theo dõi thay vì một.

### Phương án đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Một service FastAPI duy nhất gọi model VAD và ASR trong cùng tiến trình | Không kiểm soát được scale/GPU phân số độc lập theo từng giai đoạn; khó gộp batch ASR xuyên request đồng thời |
| Các microservice riêng qua message queue | Overhead vận hành cao hơn cho mục tiêu triển khai single-container; Ray Serve đã cung cấp sẵn routing và batching cần thiết |

---

## 6. Model gated yêu cầu `HF_TOKEN` tường minh

### Mô tả

`pyannote/segmentation-3.0` là model Hugging Face gated, yêu cầu tài khoản đã xác thực và chấp nhận license. `VAD_DEVICE`/`VAD_MODEL_NAME` được cấu hình trong `config/config.toml`, nhưng bản thân `HF_TOKEN` được đọc từ biến môi trường (`app/core/config/vad.py: os.getenv("HF_TOKEN", "")`) thay vì từ file TOML.

### Ưu điểm

- **Secret không nằm trong version control** — `config.toml` được commit vào git; `.env` thì không, nên token không bao giờ lọt vào lịch sử git.
- **Nhất quán với các giá trị khác theo môi trường** — port và đường dẫn cache HF cũng được cấu hình qua biến môi trường.

### Nhược điểm

- **Chế độ lỗi im lặng** — nếu `HF_TOKEN` rỗng, `Model.from_pretrained()` sẽ thất bại khi khởi động deployment VAD với lỗi xác thực Hugging Face phía upstream, thay vì một thông báo rõ ràng, riêng của dịch vụ.
- **Cần bước thủ công** — người vận hành phải tự chấp nhận license của model trên Hugging Face và tạo token trước khi dịch vụ có thể khởi động; bước này chưa được tự động hóa hay kiểm tra trước.

---

## 7. Kiểm tra định dạng audio bằng MIME Type

### Mô tả

`is_audio_file()` dùng `python-magic` để kiểm tra MIME type từ 2048 byte đầu của file upload, xác nhận có bắt đầu bằng `audio/` hoặc khớp `application/ogg` hay không, trước khi bytes được đưa vào VAD hoặc ASR.

### Ưu điểm

- **Không phụ thuộc định dạng cụ thể** — hoạt động với mọi container mà `torchaudio` có thể giải mã (MP3, WAV, FLAC, OGG, v.v.) mà không cần allowlist theo phần mở rộng.
- **Từ chối sớm, nhanh** — chỉ đọc 2048 byte; upload không hợp lệ bị từ chối trước khi gọi VAD, tiết kiệm một vòng round-trip toàn pipeline.

### Nhược điểm

- **False positive/negative với trường hợp biên** — một số container hợp lệ có thể báo MIME type không chuẩn (ví dụ: file video chỉ có track audio).
- **Dependency `python-magic`** — yêu cầu thư viện hệ thống `libmagic` có mặt trong Docker image.

---

## 8. Bề mặt API tương thích OpenAI

### Mô tả

Endpoint duy nhất được expose có signature giống OpenAI `POST /v1/audio/transcriptions`: `file`, `model`, `timestamp_granularities[]` (`"word"` hoặc `"segment"`), và `response_format` (hiện chỉ `verbose_json` được hỗ trợ có ý nghĩa).

### Ưu điểm

- **Tương thích drop-in** — client đang dùng OpenAI Python SDK có thể trỏ `base_url` sang dịch vụ này mà không cần đổi code.
- **Interface quen thuộc** — bên tích hợp không cần học một API riêng.

### Nhược điểm

- **`model` được chấp nhận nhưng chưa bắt buộc** — trong branch này, `IngressDeployment.transcribe_audio()` không kiểm tra trường `model` được yêu cầu với model ASR thực tế đang chạy trước khi dispatch vào pipeline (khác với một dịch vụ ASR đơn giai đoạn thuần túy, nơi việc kiểm tra này hợp lý đặt ở tầng ingress).
- **Chỉ là tập con của OpenAI spec** — các trường như `language` và `prompt` không được hỗ trợ, và các giá trị `response_format` khác `verbose_json` được chấp nhận nhưng không phân biệt có nghĩa.

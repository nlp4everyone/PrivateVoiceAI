# Quyết định thiết kế

Tài liệu này giải thích lý do đằng sau mỗi lựa chọn kiến trúc quan trọng, kèm phân tích ưu/nhược điểm và các phương án thay thế đã cân nhắc.

---

## 1. Ray Serve để triển khai dịch vụ

### Mô tả

Model ASR được phục vụ qua **Ray Serve** thay vì một tiến trình FastAPI/Uvicorn đơn thuần. Ray Serve quản lý vòng đời của các replica `ASRService`, cấp phát tài nguyên GPU, routing request, và auto-batching.

### Ưu điểm

- **Scale đa replica** — đặt `NUM_REPLICAS > 1` để chạy nhiều bản sao model độc lập trên nhiều GPU mà không cần thay đổi code.
- **Auto-batching có sẵn** — `@serve.batch` gom các request đồng thời trước GPU call, không cần tự quản lý queue.
- **Cô lập tài nguyên GPU** — mỗi replica khai báo `num_gpus` qua `ray_actor_options`; Ray đảm bảo cấp phát.
- **Dashboard** — Ray Serve dashboard cung cấp trạng thái replica, throughput request, và tỉ lệ lỗi theo thời gian thực.

### Nhược điểm

- **Overhead khởi động cao hơn** — khởi tạo Ray cluster và load model mất nhiều thời gian hơn Uvicorn đơn giản.
- **Phức tạp vận hành** — Ray thêm logging, actor model, và quản lý tiến trình riêng bên cạnh FastAPI.

### Phương án đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Chỉ Uvicorn + FastAPI | Không có auto-batching hay quản lý replica đa GPU |
| Triton Inference Server | Cấu hình nặng hơn; model NeMo cần bước export thêm |
| TorchServe | Kém linh hoạt cho pre/post processing tùy biến; hệ sinh thái NeMo nhỏ hơn |

---

## 2. Tự động gộp batch (`@serve.batch`)

### Mô tả

`batched_transcribe` được decorate bằng `@serve.batch(max_batch_size=MAX_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)`. Ray Serve gom các call `.remote()` đồng thời thành một list trước khi gọi handler.

### Ưu điểm

- **Tận dụng GPU** — N item trong một lần `model.transcribe()` nhanh hơn đáng kể so với N lần gọi tuần tự.
- **Đánh đổi throughput/latency có thể điều chỉnh** — `BATCH_WAIT_TIMEOUT_S` thấp ưu tiên latency; cao ưu tiên throughput.
- **Trong suốt với caller** — mỗi request nhận một `TranscriptionResult`; batching hoàn toàn ẩn.

### Nhược điểm

- **Tăng latency** — request đến ngay sau khi một batch vừa dispatch phải chờ đến `BATCH_WAIT_TIMEOUT_S` trước khi batch tiếp theo hình thành.
- **Batch size không đều** — khi tải thấp, hầu hết batch có size 1 nên lợi ích throughput không đáng kể.

---

## 3. Luồng GPU riêng (Single-Thread Executor)

### Mô tả

`process_batch_transcription()` được dispatch qua `asyncio.get_event_loop().run_in_executor(self._gpu_executor, ...)`, trong đó `self._gpu_executor` là `ThreadPoolExecutor(max_workers=1)` được tạo một lần mỗi replica.

### Ưu điểm

- **Không có CUDA context migration** — CUDA nhanh nhất trên một thread duy nhất. Chuyển GPU work qua nhiều thread sẽ buộc context switching, làm tăng latency.
- **Tránh tranh chấp pool** — executor mặc định của `asyncio` được chia sẻ cho mọi task async; đưa GPU work vào đó sẽ cạnh tranh với các tác vụ CPU-bound.
- **GPU access tuần tự** — một thread đảm bảo các batch chạy nối tiếp, không chồng chéo.

### Nhược điểm

- **Không có parallelism trong replica** — một batch lớn mất nhiều thời gian xử lý không thể được overlap với việc decode batch tiếp theo.

---

## 4. Tách Mixed-Batch

### Mô tả

Khi batch chứa cả request có và không có timestamp, `process_batch_transcription()` thực hiện 2 GPU sub-call (`split_mixed_batch=True`):
1. `model.transcribe(ts_items, timestamps=True)`
2. `model.transcribe(no_ts_items, timestamps=False)`

### Ưu điểm

- **Tránh GPU→CPU logit transfer không cần thiết** — CTC alignment của NeMo (cần cho timestamp) chuyển logit từ GPU xuống CPU. Item không cần timestamp sẽ phải chịu chi phí này một cách không cần thiết khi gọi unified với `timestamps=True`.
- **Latency thấp hơn cho item no-timestamp** — hoàn thành mà không phải chờ CTC alignment của item timestamp.

### Nhược điểm

- **Hai lần khởi động GPU kernel** — nếu toàn bộ batch cùng loại, overhead này không phát sinh; nhưng với mixed batch cần 2 call.
- **Có thể kiểm soát** — `SPLIT_MIXED_BATCH=False` fallback về 1 call với stripping nếu overhead 2 call vượt hơn chi phí logit transfer.

---

## 5. Sắp xếp Batch theo Độ dài

### Mô tả

Trước khi gọi `model.transcribe()`, tensor audio được sắp xếp theo độ dài tăng dần. Hoán vị được ghi lại và kết quả được sắp xếp ngược lại trước khi trả về.

### Ưu điểm

- **Giảm padding** — NeMo tự động chia batch lớn thành sub-batch. Nhóm các item có độ dài tương đương với nhau giảm lượng zero-padding thêm vào item ngắn hơn, từ đó giảm GPU compute lãng phí.

### Nhược điểm

- **Overhead sort** — `O(N log N)` mỗi batch; không đáng kể với batch size thông thường (≤ 32).
- **Cần khôi phục thứ tự** — phải theo dõi hoán vị index và đảo ngược lại.

---

## 6. AMP Autocast

### Mô tả

GPU inference chạy dưới `torch.cuda.amp.autocast()`, tự động cast các phép toán phù hợp sang FP16 (hoặc BF16 trên Ampere+).

### Ưu điểm

- **Inference nhanh hơn** — tensor core FP16 trên GPU NVIDIA hiện đại nhanh hơn đáng kể so với FP32.
- **Dùng ít VRAM hơn** — tensor trung gian nhỏ hơn cho phép batch lớn hơn vừa trong memory.
- **Không mất độ chính xác** — model ASR của NeMo robust với inference FP16.

### Nhược điểm

- **Yêu cầu thiết bị** — không có tác dụng trên CPU; chỉ có lợi trên phần cứng CUDA.

---

## 7. Cached Resampler (`lru_cache`)

### Mô tả

Các instance `torchaudio.transforms.Resample(sr_src, sr_tgt)` được cache theo cặp `(sr_src, sr_tgt)` với `lru_cache(maxsize=8)`.

### Ưu điểm

- **Loại bỏ việc tạo lại nhiều lần** — tạo `Resample` transform liên quan đến cấp phát filter kernel. Cache giúp các lần gọi tiếp theo với cùng cặp tốc độ mẫu gần như miễn phí.
- **Overhead thấp** — hầu hết deployment chỉ có 1–2 cặp tốc độ mẫu duy nhất; `maxsize=8` là quá đủ.

### Nhược điểm

- **Giữ memory vô thời hạn** — transform được cache không bao giờ bị thu hồi trong vòng đời tiến trình (tối đa `maxsize` entry). Với 8 entry điều này không đáng kể.

---

## 8. Kiểm tra Định dạng Audio bằng MIME Type

### Mô tả

`is_audio_file()` dùng `python-magic` để kiểm tra MIME type từ 2048 byte đầu của file upload, xác nhận có bắt đầu bằng `audio/` hoặc khớp `application/ogg` không.

### Ưu điểm

- **Không phụ thuộc định dạng cụ thể** — hoạt động với mọi container audio (MP3, WAV, FLAC, OGG, M4A) mà không cần allowlist tường minh.
- **Nhanh** — chỉ đọc 2048 byte; không cần giải mã toàn bộ.
- **Từ chối sớm** — file không hợp lệ bị loại trước khi giải mã audio và trước khi vào GPU queue.

### Nhược điểm

- **Có thể có false positive** — một số container audio hợp lệ có thể có MIME type không chuẩn (ví dụ: file video chỉ có track audio).
- **Dependency `python-magic`** — yêu cầu thư viện hệ thống `libmagic` trong Docker image.

---

## 9. API Tương thích OpenAI

### Mô tả

Endpoint có signature giống OpenAI `POST /v1/audio/transcriptions`:
- `file` — file audio (multipart)
- `model` — định danh model
- `timestamp_granularities[]` — `"word"` hoặc `"segment"`
- `response_format` — hiện tại là `verbose_json`

### Ưu điểm

- **Tương thích drop-in** — bất kỳ client nào dùng OpenAI Python SDK đều có thể chuyển sang dịch vụ này chỉ bằng cách đổi `base_url`.
- **Interface quen thuộc** — developer không cần học API mới.

### Nhược điểm

- **Chỉ là tập con của OpenAI spec** — các tùy chọn `response_format` khác ngoài `verbose_json` được chấp nhận nhưng không phân biệt có nghĩa; một số trường OpenAI (như `language`, `prompt`) không được hỗ trợ.

# Tổng quan kỹ thuật

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CLIENT (OpenAI SDK / curl)                       │
│            POST /v1/audio/transcriptions  (multipart/form-data)      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Ray Serve  (serve.start)                            │
│  host: RAY_HOST   port: RAY_PORT                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  IngressDeployment  (@serve.deployment + @serve.ingress)      │    │
│  │  num_replicas=INGRESS_NUM_REPLICAS                            │    │
│  │  num_gpus=INGRESS_NUM_GPUS (0 — chỉ dùng CPU)                 │    │
│  │  max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS             │    │
│  │                                                                │    │
│  │  POST /v1/audio/transcriptions                                 │    │
│  │      │ đọc bytes, kiểm tra MIME qua is_audio_file()             │    │
│  │      │ pipeline.remote(audio_bytes, timestamp_granularity)     │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Pipeline  (@serve.deployment, actor thường — không @serve.batch)│  │
│  │      │ load_audio_from_bytes(audio_bytes) → tensor waveform     │    │
│  │      ▼                                                        │    │
│  │  VADDeployment.remote(audio_bytes)                             │    │
│  │      │ PyannoteVADDetector.detect() → List[VADSegment]         │    │
│  │      ▼                                                        │    │
│  │  cắt waveform theo từng đoạn → List[Tensor]                    │    │
│  │      ▼                                                        │    │
│  │  ASRDeployment.batched_transcribe_tensors.remote(seg, gran) ×N │    │
│  │      (fan-out: một call cho mỗi đoạn giọng nói, chờ cùng nhau)  │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  ASRDeployment  (@serve.deployment)                            │    │
│  │  num_replicas=ASR_NUM_REPLICAS  num_gpus=ASR_NUM_GPUS           │    │
│  │                                                                │    │
│  │  @serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE,                │    │
│  │               batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)    │    │
│  │  batched_transcribe_tensors(batch, granularities)               │    │
│  │      │ process_batch_transcription()                           │    │
│  │      ▼                                                        │    │
│  │  ParakeetRecognizer.transcribe() → NeMo ASRModel (GPU)          │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  VADDeployment  (@serve.deployment)                            │    │
│  │  num_replicas=VAD_NUM_REPLICAS  num_gpus=VAD_NUM_GPUS           │    │
│  │                                                                │    │
│  │  __call__(audio_bytes) → PyannoteVADDetector (GPU) → segments   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

Bốn Ray Serve deployment độc lập tạo thành đồ thị: `IngressDeployment` (tầng HTTP), `Pipeline` (điều phối), `VADDeployment`, và `ASRDeployment`. Mỗi deployment được liên kết trong `bind_app()` và có thể scale, cấp GPU, cấu hình hoàn toàn độc lập với nhau.

---

## Pipeline xử lý

### Giai đoạn 1 — Kiểm tra request (Ingress)

`IngressDeployment.transcribe_audio()` nhận file upload và:
1. Đọc toàn bộ audio bytes vào memory.
2. Kiểm tra MIME type qua `python-magic` (`is_audio_file()`) — raise `UnsupportedAudioFormatException` nếu không phải audio.
3. Chuyển tiếp `(audio_bytes, timestamp_granularity)` sang deployment `Pipeline` qua `.remote()` và chờ kết quả.

Trường form `model` được chấp nhận nhưng hiện chưa được đối chiếu với model ASR đang chạy tại tầng ingress (khác với biến thể ASR đơn thuần của dịch vụ này).

### Giai đoạn 2 — Giải mã toàn bộ audio + VAD (Pipeline + VADDeployment)

`Pipeline.__call__()`:
1. Giải mã toàn bộ audio một lần bằng `load_audio_from_bytes()` thành tensor waveform mono 16kHz — bản này dùng để cắt đoạn ở bước sau.
2. Gọi `VADDeployment.remote(audio_bytes)`, nội bộ chạy `PyannoteVADDetector.detect()`. Detector tự giải mã lại bytes (qua `load_audio_from_bytes`) và chạy pipeline `VoiceActivityDetection` của `pyannote.audio` (dựa trên `pyannote/segmentation-3.0`), trả về danh sách `VADSegment(start, end)` tính bằng giây, làm tròn 3 chữ số thập phân.
3. Nếu VAD không tìm thấy giọng nói, pipeline dừng sớm và trả về `TranscriptionResponse` / `WordResponse` / `SegmentResponse` rỗng tùy theo granularity được yêu cầu.

### Giai đoạn 3 — Trích xuất đoạn audio

`Pipeline._extract_audio_segments()` chuyển start/end (giây) của mỗi `VADSegment` thành chỉ số sample dựa trên waveform 16kHz, giới hạn trong biên của tensor, rồi cắt ra một tensor 1 chiều cho mỗi vùng giọng nói. Các đoạn có khoảng không hợp lệ (không dương) sẽ bị bỏ qua kèm cảnh báo.

### Giai đoạn 4 — Gộp batch ASR xuyên request (ASRDeployment)

Mỗi tensor đoạn được trích xuất được gửi độc lập tới `ASRDeployment.batched_transcribe_tensors.remote(segment, timestamp_granularity)`, và tất cả các call của một request HTTP được chờ cùng nhau bằng `asyncio.gather`. Vì method này được decorate bằng `@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)`, Ray Serve tự động gộp các đoạn đến từ *nhiều* request HTTP đồng thời khác nhau vào cùng một GPU batch — việc gộp batch xảy ra ở cấp độ đoạn (segment), không phải cấp độ request.

`process_batch_transcription()` chia batch thành nhóm có timestamp và nhóm không có timestamp (mỗi nhóm gọi `ParakeetRecognizer.transcribe()` một lần) rồi ghép kết quả lại đúng thứ tự ban đầu. `ASRDeployment` cũng expose `batched_transcribe` (nhận bytes) cho các call ASR trực tiếp không qua VAD, nhưng pipeline VAD luôn dùng đường tensor `batched_transcribe_tensors` để tránh giải mã dư thừa.

### Giai đoạn 5 — Điều chỉnh offset timestamp và định dạng response (Pipeline)

Mỗi `TranscriptionResult` theo từng đoạn mang timestamp tính tương đối theo mốc bắt đầu của chính đoạn (đã cắt) đó. Pipeline khôi phục timestamp tuyệt đối bằng cách cộng `VADSegment.start` tương ứng vào mỗi timestamp từ/đoạn trước khi tạo response cuối:

- `None` → `TranscriptionResponse` — text đã nối + token usage ước lượng (dựa trên `tiktoken`)
- `"word"` → `WordResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo từ (đã điều chỉnh offset)
- `"segment"` → `SegmentResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo đoạn (đã điều chỉnh offset, đánh lại `id` tuần tự)

Phát hiện ngôn ngữ dùng `pycld2`, chỉ chạy cho response có timestamp. Thời lượng được ước lượng từ audio bytes gốc qua `soundfile.info()` (`estimate_audio_duration`), không phải tổng thời lượng các đoạn giọng nói đã trích xuất.

---

## Cấu hình

Toàn bộ tham số nằm trong `config/config.toml` (topology deployment ổn định, lựa chọn model) cộng với `.env` (port, secret). Thay đổi yêu cầu restart container/service.

### `[system]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `AUDIO_TEMP_DIR` | `/dev/shm` | Thư mục tạm (tmpfs chia sẻ bộ nhớ) |
| `DEPLOYMENT_NAME` | `ASRService` | Tên logic của deployment ASR |
| `VAD_DEPLOYMENT_NAME` | `VADService` | Tên logic của deployment VAD |
| `RAY_HOST` | `0.0.0.0` | HTTP host của Ray Serve |
| `RAY_PORT` | `8000` | HTTP port của Ray Serve (nội bộ container) |

### `[asr]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | Định danh model trên HuggingFace |
| `ASR_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `ASR_NUM_GPUS` | `0.2` | GPU phân số cấp cho mỗi replica ASR |
| `ASR_NUM_REPLICAS` | `1` | Số replica `ASRDeployment` |
| `ASR_MAX_ONGOING_REQUESTS` | `8` | Số request đang xử lý tối đa mỗi replica ASR |
| `ASR_MAX_BATCH_SIZE` | `8` | Số đoạn tối đa mỗi GPU batch |
| `ASR_BATCH_WAIT_TIMEOUT_S` | `0.1` | Thời gian chờ tối đa để điền đầy batch ASR (giây) |

### `[vad]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `VAD_MODEL_NAME` | `pyannote/segmentation-3.0` | Model segmentation Pyannote trên HuggingFace |
| `VAD_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VAD_NUM_GPUS` | `0.1` | GPU phân số cấp cho mỗi replica VAD |
| `VAD_NUM_REPLICAS` | `4` | Số replica `VADDeployment` |
| `VAD_MAX_ONGOING_REQUESTS` | `8` | Số request đang xử lý tối đa mỗi replica VAD |

### `[ingress]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `INGRESS_NUM_GPUS` | `0` | GPU cấp cho mỗi replica ingress (tầng chỉ dùng CPU) |
| `INGRESS_NUM_REPLICAS` | `1` | Số replica `IngressDeployment` |
| `INGRESS_MAX_ONGOING_REQUESTS` | `16` | Số request HTTP đang xử lý tối đa mỗi replica ingress |

### Biến môi trường (`.env` / `docker-compose.yml`)

| Biến | Mô tả |
|---|---|
| `RAY_FASTAPI_PORT` | Host port map tới Ray Serve HTTP (mặc định `8000`, `.env` mẫu dùng `8005`) |
| `RAY_DASHBOARD_PORT` | Host port map tới Ray dashboard (mặc định `8265`) |
| `HF_TOKEN` | Token truy cập Hugging Face — bắt buộc để tải model `pyannote/segmentation-3.0` (model gated) |
| `HF_HOME` | Thư mục cache HuggingFace (map vào volume cache đã mount) |

Xem [CONFIGURATION.md](../CONFIGURATION.md) để có bảng tham chiếu đầy đủ, dạng phẳng, của toàn bộ tham số.

---

## Cấu trúc thư mục

```
VoicePlatform/
├── app/
│   ├── app.py                                # Entry point: bind đồ thị deployment, ray.init/serve.start
│   ├── core/config/
│   │   ├── asr.py                            # ASR_MODEL_NAME, ASR_DEVICE
│   │   ├── vad.py                            # VAD_MODEL_NAME, VAD_DEVICE, HF_TOKEN
│   │   ├── serving.py                        # Cấu hình replica & batch cho ASR_/VAD_/INGRESS_
│   │   └── system.py                         # RAY_HOST, RAY_PORT, DEPLOYMENT_NAME, VAD_DEPLOYMENT_NAME
│   ├── services/
│   │   ├── asr/
│   │   │   ├── factory.py                    # RecognizerFactory — tạo ParakeetRecognizer
│   │   │   └── nemo/recognizer.py            # ParakeetRecognizer — bọc NeMo ASRModel
│   │   ├── vad/
│   │   │   ├── factory.py                    # VADFactory — tạo PyannoteVADDetector
│   │   │   └── pyannote/detector.py          # PyannoteVADDetector — bọc pipeline pyannote.audio
│   │   └── deployments/
│   │       ├── asr_deployment.py             # ASRDeployment — phiên âm ASR theo batch
│   │       ├── vad_deployment.py             # VADDeployment — phát hiện đoạn giọng nói
│   │       └── ingress_deployment.py         # Pipeline + IngressDeployment — điều phối + FastAPI
│   ├── utils/
│   │   ├── audio/io.py                       # load_audio_from_bytes, is_audio_file, estimate_audio_duration
│   │   ├── transcription/helper.py           # process_batch_transcription, get_timestamp_indices
│   │   ├── language_detect/detector.py       # LanguageDetector (pycld2)
│   │   └── token_counter/token_counter.py    # approximate_count_tokens (tiktoken)
│   ├── schema/
│   │   ├── segment/base.py                   # TimeSegment, VADSegment
│   │   ├── transcription/                    # TranscriptionResult, các kiểu response, TranscriptionType
│   │   └── vad/                               # BaseVADDetector, schema response VAD
│   └── exceptions/                            # TranscriptedModelNotFoundException, UnsupportedAudioFormatException
├── config/config.toml                         # Các section [system] [asr] [vad] [ingress]
├── docker/
│   ├── Dockerfile                             # Image CUDA 12.8 + NeMo 2.7.2 + Ray Serve
│   └── docker-compose.yml                     # Triển khai GPU single-container
└── examples/                                  # audio_transcription_example.py, concurrent_requests_example.py
```

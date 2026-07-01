# Tổng quan kỹ thuật

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                  CLIENT (OpenAI SDK / curl)                      │
│         POST /v1/audio/transcriptions  (multipart/form-data)     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Ray Serve  (serve.start)                        │
│  host: RAY_HOST   port: RAY_PORT                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ASRService  (@serve.deployment)                           │  │
│  │  num_replicas=NUM_REPLICAS                                 │  │
│  │  num_gpus=NUM_GPUS  mỗi replica                            │  │
│  │  max_ongoing_requests=MAX_ONGOING_REQUESTS                 │  │
│  │                                                            │  │
│  │  FastAPI ingress (@serve.ingress)                          │  │
│  │   POST /v1/audio/transcriptions                            │  │
│  │       │ kiểm tra model, đọc bytes, MIME check              │  │
│  │       │ .batched_transcribe.remote(bytes, granularity)     │  │
│  │       ▼                                                    │  │
│  │  @serve.batch                                              │  │
│  │  batched_transcribe(batch, granularities)                  │  │
│  │       │ asyncio.to_thread(load_audio_from_bytes) × N       │  │
│  │       │ run_in_executor(GPU thread, process_batch)         │  │
│  │       ▼                                                    │  │
│  │  process_batch_transcription()                             │  │
│  │       │ tách ts / no-ts sub-batch                          │  │
│  │       ▼                                                    │  │
│  │  ParakeetRecognizer.transcribe()                           │  │
│  │       │ sắp xếp theo độ dài → autocast → model.transcribe()│  │
│  │       ▼                                                    │  │
│  │  NeMo ASRModel  (GPU)                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline xử lý

### Giai đoạn 1 — Kiểm tra request

`transcribe_audio()` nhận file upload và:
1. Kiểm tra `model` có khớp với `asr_model.model_name` đang chạy không — nếu không raise `TranscriptedModelNotFoundException`.
2. Đọc toàn bộ audio bytes vào memory.
3. Kiểm tra MIME type qua `python-magic` — raise `UnsupportedAudioFormatException` nếu không phải audio.

### Giai đoạn 2 — Xếp hàng batch

Audio bytes và `timestamp_granularity` được gửi vào `batched_transcribe` qua `.remote()`. Ray Serve gom các call đồng thời thành một batch (tối đa `MAX_BATCH_SIZE`) trong thời gian tối đa `BATCH_WAIT_TIMEOUT_S` giây.

### Giai đoạn 3 — Giải mã audio (song song, CPU)

Mỗi item trong batch được giải mã song song qua `asyncio.to_thread` bằng `load_audio_from_bytes()`:
- `torchaudio.load` → waveform + sample rate
- Chuyển về mono (mean theo channel)
- Resample về 16kHz nếu cần (dùng `Resample` transform đã cache qua `lru_cache`)
- Trả về `(waveform: torch.Tensor, duration: float)`

Raw bytes được giải phóng ngay sau khi giải mã.

### Giai đoạn 4 — GPU transcription (luồng riêng)

`process_batch_transcription()` chạy trên `ThreadPoolExecutor` 1 luồng (mỗi replica một executor) để giữ GPU work trên một thread duy nhất, tránh CUDA context migration:

| Thành phần batch | Số GPU call |
|---|---|
| Toàn bộ không có timestamp | 1 call, `timestamps=False` |
| Toàn bộ có timestamp | 1 call, `timestamps=True` |
| Hỗn hợp | 2 sub-call — item có ts + item không có ts |

Trong mỗi GPU call, `ParakeetRecognizer.transcribe()`:
1. Sắp xếp tensor theo độ dài tăng dần — giảm padding thừa trong các sub-batch nội bộ của NeMo.
2. Chạy `model.transcribe()` dưới `torch.cuda.amp.autocast()`.
3. Khôi phục thứ tự gốc trước khi trả về.

### Giai đoạn 5 — Định dạng response

Dựa trên `timestamp_granularity`:
- `None` → `TranscriptionResponse` — text + token usage
- `"word"` → `WordResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo từ
- `"segment"` → `SegmentResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo đoạn

Phát hiện ngôn ngữ dùng `pycld2`, chỉ chạy cho response có timestamp.

---

## Cấu hình

Tất cả tham số nằm trong `config/config.toml`. Thay đổi yêu cầu restart container.

### `[serving]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `NUM_GPUS` | `1` | Số GPU cấp phát mỗi replica |
| `NUM_REPLICAS` | `1` | Số replica ASRService |
| `MAX_ONGOING_REQUESTS` | `16` | Số request đang xử lý tối đa mỗi replica |
| `MAX_BATCH_SIZE` | `8` | Số item tối đa mỗi GPU batch |
| `BATCH_WAIT_TIMEOUT_S` | `0.1` | Thời gian chờ tối đa để điền đầy batch (giây) |

### `[asr]`

| Key | Mặc định | Mô tả |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | Định danh model trên HuggingFace |
| `ASR_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

### `[system]`

| Key | Mô tả |
|---|---|
| `RAY_HOST` | HTTP host của Ray Serve |
| `RAY_PORT` | HTTP port của Ray Serve |
| `DEPLOYMENT_NAME` | Tên đăng ký với Ray Serve |

### Biến môi trường (`.env` / `docker-compose.yml`)

| Biến | Mô tả |
|---|---|
| `RAY_FASTAPI_PORT` | Host port map tới Ray Serve HTTP (mặc định `8000`) |
| `RAY_DASHBOARD_PORT` | Host port map tới Ray dashboard (mặc định `8265`) |
| `RAY_LOG_LEVEL` | Log level nội bộ của Ray (mặc định `WARNING`) |
| `RAY_SERVE_LOG_TO_STDERR` | Chuyển Serve logs ra stderr (mặc định `1`) |
| `HF_HOME` | Thư mục cache HuggingFace |

---

## Cấu trúc thư mục

```
VoicePlatform/
├── app/
│   ├── app.py                          # Entry point: ray.init, serve.start, ASRService.bind
│   ├── core/config/
│   │   ├── asr.py                      # ASR_MODEL_NAME, ASR_DEVICE
│   │   ├── serving.py                  # NUM_GPUS, NUM_REPLICAS, MAX_BATCH_SIZE, …
│   │   └── system.py                   # RAY_HOST, RAY_PORT, DEPLOYMENT_NAME
│   ├── services/
│   │   ├── asr/
│   │   │   ├── factory.py              # RecognizerFactory — tạo ParakeetRecognizer
│   │   │   └── nemo/recognizer.py      # ParakeetRecognizer — bọc NeMo ASRModel
│   │   └── deployments/
│   │       └── asr_deployment.py       # ASRService — Ray Serve deployment + FastAPI
│   ├── utils/
│   │   ├── audio/io.py                 # load_audio_from_bytes, is_audio_file
│   │   └── transcription/helper.py     # process_batch_transcription
│   └── schema/transcription/           # TranscriptionResult, các kiểu response
├── config/config.toml                  # Toàn bộ cấu hình runtime
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── examples/                           # Ví dụ sử dụng
```

# Chi tiết các component

## ASRService (`app/services/deployments/asr_deployment.py`)

Deployment của Ray Serve, sở hữu FastAPI ingress và logic xử lý batch.

**Cấu hình deployment** (qua `@serve.deployment`):
- `num_replicas=NUM_REPLICAS` — số replica độc lập; mỗi replica giữ một bản sao model
- `num_gpus=NUM_GPUS` — tài nguyên GPU dành riêng mỗi replica
- `max_ongoing_requests=MAX_ONGOING_REQUESTS` — số request đang xử lý tối đa mỗi replica trước khi Ray Serve áp dụng backpressure

**`__init__()`**
- Áp dụng lại log level INFO cho `ray.serve` (Ray Serve reset trong quá trình khởi tạo actor)
- Gọi `RecognizerFactory.create()` để tải model ASR
- Cache deployment handle (`serve.get_deployment_handle`) để tự routing vào batch queue
- Tạo `ThreadPoolExecutor(max_workers=1)` — một luồng riêng giữ tất cả GPU work tuần tự, tránh CUDA context migration

**`transcribe_audio()`** — handler FastAPI endpoint
- Kiểm tra model request khớp với model đang chạy
- Kiểm tra định dạng audio qua MIME check
- Chuyển tiếp sang `batched_transcribe` qua `.remote()` và chờ kết quả
- Định dạng response theo `timestamp_granularity`

**`batched_transcribe()`** — handler `@serve.batch`
- Nhận batch các cặp `(audio_bytes, granularity)` được Ray Serve gom lại
- Giải mã audio song song bằng `asyncio.to_thread`
- Giải phóng raw bytes ngay sau khi giải mã để giảm peak memory
- Dispatch sang `process_batch_transcription()` trên GPU executor

---

## RecognizerFactory (`app/services/asr/factory.py`)

Factory đơn giản với storage singleton ở cấp class. `create(model_name, device)`:
1. Log cảnh báo nếu `model_name` không bắt đầu bằng `"nvidia/parakeet"` (nhưng vẫn tiếp tục)
2. Khởi tạo `ParakeetRecognizer` và trả về
3. Lưu instance vào `_recognizer` — các lần gọi sau sẽ thay thế instance cũ

`get_recognizer_model()` — trả về `_recognizer.model_name` hoặc `None`.

---

## ParakeetRecognizer (`app/services/asr/nemo/recognizer.py`)

Bọc `nemo_asr.models.ASRModel` với sorting, autocast, và structured output.

**Model được hỗ trợ:**
- `nvidia/parakeet-ctc-0.6b-vi`
- `nvidia/parakeet-tdt-0.6b-v3`

Tên model không hợp lệ sẽ fallback về `SUPPORTED_MODELS[0]`.

**Khởi tạo:**
1. Xác định device: `"auto"` → `"cuda"` nếu `torch.cuda.is_available()`, không thì `"cpu"`
2. `nemo_asr.models.ASRModel.from_pretrained(model_name)` — tải từ HF cache hoặc download
3. `model.cuda()` nếu CUDA
4. `model.eval()`

**`transcribe(audio, enable_timestamps, precision=3)`**
- Chuẩn hóa input đơn lẻ thành `[item]`
- Batch nhiều tensor: sắp xếp tăng dần theo độ dài, ghi lại hoán vị
- Chạy inference dưới `torch.cuda.amp.autocast()`
- Khôi phục thứ tự gốc sau inference
- Trả về `List[TranscriptionResult]`; timestamp được làm tròn đến `precision` chữ số thập phân

**Properties:** `model_name`, `supported_models`

---

## Audio Utils (`app/utils/audio/io.py`)

**`load_audio_from_bytes(audio_bytes, target_sr=16000) → (Tensor, float)`**
- `torchaudio.load(BytesIO(audio_bytes))` — hỗ trợ MP3, WAV, FLAC, OGG và các định dạng torchaudio khác
- Mono: `waveform[0]` nếu 1 channel, `waveform.mean(dim=0)` nếu nhiều channel
- Resample về target_sr nếu `sr != target_sr` dùng `torchaudio.transforms.Resample` đã cache (key `(sr_src, sr_tgt)` qua `lru_cache(maxsize=8)`)
- Trả về `(waveform, duration_seconds)`

**`is_audio_file(data, buffer_size=2048) → bool`**
- Dùng `python-magic` để kiểm tra MIME type từ 2048 byte đầu
- Trả về `True` cho `audio/*` và `application/ogg`; `False` trong mọi trường hợp còn lại kể cả exception

**`estimate_audio_duration(audio_bytes) → float`**
- Dùng `soundfile.info()` để ước lượng thời lượng nhẹ mà không cần giải mã toàn bộ

---

## Batch Processing Helper (`app/utils/transcription/helper.py`)

**`process_batch_transcription(asr_model, audio_data, timestamp_granularities, split_mixed_batch=True)`**

Dispatch sang 1 hoặc 2 GPU call tùy thành phần batch:

```
ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

Toàn bộ no-ts:  asr_model.transcribe(all, enable_timestamps=False)          → 1 GPU call
Toàn bộ ts:     asr_model.transcribe(all, enable_timestamps=True)           → 1 GPU call
Hỗn hợp (split_mixed_batch=True):
    asr_model.transcribe(ts_items, enable_timestamps=True)                  → GPU call 1
    asr_model.transcribe(no_ts_items, enable_timestamps=False)              → GPU call 2
    ghép kết quả về vị trí gốc
Hỗn hợp (split_mixed_batch=False):
    asr_model.transcribe(all, enable_timestamps=True)                       → 1 GPU call
    xóa .words / .segments khỏi item no-ts
```

**`get_timestamp_indices(timestamp_granularities) → (ts_indices, no_ts_indices)`**
- Phân chia danh sách granularity string (`None`, `"word"`, `"segment"`) thành 2 danh sách index

**`get_transcription_type(type) → TranscriptionType`**
- `None` → `Text`, `"word"` → `Word`, `"segment"` → `Segment`

---

## Response Schemas (`app/schema/transcription/`)

| Class | Dùng khi | Các trường |
|---|---|---|
| `TranscriptionResponse` | `timestamp_granularity=None` | `text`, `usage` (token count) |
| `WordResponse` | `timestamp_granularity="word"` | `text`, `language`, `duration`, `usage` (giây), `words` |
| `SegmentResponse` | `timestamp_granularity="segment"` | `text`, `language`, `duration`, `usage` (giây), `segments` |

`TranscribedWord`: `start`, `end`, `word`
`AdvancedTranscribedSegment`: `id`, `start`, `end`, `text`

---

## Configuration Loader (`app/utils/config_loader/toml_loader.py`)

Đọc `config/config.toml` và trả về từng section dưới dạng dict. Được dùng bởi `app/core/config/*.py` để điền các hằng số cấp module, import qua `*`.

---

## Exception Handlers (`app/exceptions/`)

| Exception | Điều kiện kích hoạt | HTTP response |
|---|---|---|
| `TranscriptedModelNotFoundException` | Model request ≠ model đang chạy | 404 |
| `UnsupportedAudioFormatException` | MIME check thất bại | 415 |

Cả hai được đăng ký qua `asr_app.add_exception_handler()` và xử lý bởi `common_exception_handler`.

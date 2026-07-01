# Chi tiết các component

## IngressDeployment (`app/services/deployments/ingress_deployment.py`)

Deployment của Ray Serve sở hữu tầng HTTP FastAPI. Nó không giữ model nào — chỉ kiểm tra input và chuyển tiếp sang `Pipeline`.

**Cấu hình deployment** (`@serve.deployment`):
- `ray_actor_options={"num_gpus": INGRESS_NUM_GPUS}` — mặc định `0` (chỉ dùng CPU)
- `num_replicas=INGRESS_NUM_REPLICAS`
- `max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS`

**`transcribe_audio()`** — handler endpoint FastAPI (`POST /v1/audio/transcriptions`)
- Đọc file upload vào memory (`await file.read()`)
- Kiểm tra định dạng audio qua `is_audio_file()` — raise `UnsupportedAudioFormatException(file_extension)` nếu MIME sniff thất bại
- Chuyển tiếp sang `Pipeline.remote(audio_bytes, timestamp_granularity)` và chờ kết quả
- Trả kết quả trực tiếp (đã là `TranscriptionResponse` / `WordResponse` / `SegmentResponse`)

Đăng ký exception handler cho `TranscriptedModelNotFoundException` và `UnsupportedAudioFormatException` qua `common_exception_handler`.

---

## Pipeline (`app/services/deployments/ingress_deployment.py`)

Actor `@serve.deployment` thông thường (không `@serve.batch`, không phải HTTP ingress) sở hữu logic điều phối VAD→ASR. Được khởi tạo với handle tới cả `VADDeployment` và `ASRDeployment`.

**`__call__(audio_bytes, timestamp_granularity)`**
1. `load_audio_from_bytes(audio_bytes)` — giải mã toàn bộ audio một lần thành tensor waveform mono, 16kHz.
2. `self.vad.remote(audio_bytes)` — chờ danh sách `VADSegment` từ `VADDeployment`.
3. Nếu không phát hiện đoạn nào, trả về ngay response rỗng loại phù hợp (`TranscriptionResponse`, `WordResponse`, hoặc `SegmentResponse`).
4. `self._extract_audio_segments(audio_tensor, vad_segments)` — cắt waveform thành từng tensor riêng cho mỗi đoạn giọng nói.
5. Gửi mỗi đoạn một call `self.asr.batched_transcribe_tensors.remote(segment, timestamp_granularity)` và chờ tất cả bằng `asyncio.gather`.
6. Nối tất cả text các đoạn bằng dấu cách thành `combined_text`.
7. Xây dựng response cuối cùng, điều chỉnh timestamp từ/đoạn bằng cách cộng lại offset start của đoạn VAD tương ứng (xem [Điều chỉnh offset timestamp](#điều-chỉnh-offset-timestamp) bên dưới).

**`_extract_audio_segments(audio_tensor, vad_segments, sample_rate=16000)`**
- Chuyển `VADSegment.start` / `.end` (giây) thành chỉ số sample: `int(seconds * sample_rate)`
- Giới hạn `start_idx` `>= 0` và `end_idx` `<= len(audio_tensor)`
- Bỏ qua (và log cảnh báo) đoạn nào có `end_idx <= start_idx`
- Trả về `List[torch.Tensor]`, mỗi tensor 1 chiều cho một vùng giọng nói hợp lệ

### Điều chỉnh offset timestamp

Timestamp ASR được tính độc lập theo từng đoạn nên tương đối với mốc bắt đầu của audio đoạn đó (đã cắt), không phải file gốc. Pipeline khôi phục thời gian tuyệt đối bằng cách cộng `VADSegment.start` tương ứng vào mỗi timestamp:

```python
adjusted_word = word.model_copy(update={
    "start": word.start + vad_seg.start,
    "end": word.end + vad_seg.start,
})
```

Cùng offset đó được áp dụng cho timestamp cấp đoạn, và `id` của đoạn được đánh lại tuần tự (`id=len(all_segments)`) trên toàn bộ response, thay vì dùng lại id riêng của từng đoạn ASR.

---

## VADDeployment (`app/services/deployments/vad_deployment.py`)

Deployment Ray Serve bọc model VAD.

**Cấu hình deployment:**
- `ray_actor_options={"num_gpus": VAD_NUM_GPUS}`
- `num_replicas=VAD_NUM_REPLICAS`
- `max_ongoing_requests=VAD_MAX_ONGOING_REQUESTS`

**`__init__()`** — gọi `VADFactory.create(model_name=VAD_MODEL_NAME, device=VAD_DEVICE, token=HF_TOKEN)`; raise `RuntimeError` nếu factory trả về `None`.

**`__call__(audio_bytes) -> List[VADSegment]`** — chuyển tiếp sang `self._vad_model.detect(audio=audio_bytes, precision=3)` và log số đoạn phát hiện được. Đây là một call deployment thông thường (không `@serve.batch`) — mỗi request HTTP kích hoạt riêng một lần inference VAD, dù Ray Serve vẫn cân bằng tải các call qua `VAD_NUM_REPLICAS` replica.

---

## ASRDeployment (`app/services/deployments/asr_deployment.py`)

Deployment Ray Serve bọc model ASR, với hai entry point dạng batch.

**Cấu hình deployment:**
- `ray_actor_options={"num_gpus": ASR_NUM_GPUS}`
- `num_replicas=ASR_NUM_REPLICAS`
- `max_ongoing_requests=ASR_MAX_ONGOING_REQUESTS`

**`__init__()`** — gọi `RecognizerFactory.create(model_name=ASR_MODEL_NAME, device=ASR_DEVICE)`; raise `RuntimeError` nếu factory trả về `None`.

**`batched_transcribe(batch: List[bytes], timestamp_granularities)`** — `@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)`. Giải mã mỗi item bằng `load_audio_from_bytes` (song song qua `asyncio.gather`), sau đó gọi `process_batch_transcription()`. Dùng cho call ASR trực tiếp nhận bytes (không được pipeline VAD sử dụng, vì pipeline đã có sẵn tensor đã giải mã).

**`batched_transcribe_tensors(batch: List[torch.Tensor], timestamp_granularities)`** — cùng decorator `@serve.batch`; bỏ hoàn toàn bước giải mã vì caller (deployment `Pipeline`) đã cắt và cung cấp sẵn tensor waveform. Đây là method mà pipeline VAD+ASR thực sự gọi, một lần cho mỗi đoạn giọng nói phát hiện được. Vì `@serve.batch` hoạt động ở cấp deployment, các đoạn từ những request HTTP không liên quan có thể được gộp vào cùng batch `ASRDeployment` bên dưới.

**`__call__(audio_bytes, timestamp_granularity=None)`** — đường gọi trực tiếp tiện lợi, chờ `self.batched_transcribe.remote(...)`; không được pipeline VAD sử dụng.

---

## RecognizerFactory (`app/services/asr/factory.py`)

Factory singleton cấp class. `create(model_name, device)`:
1. Log lỗi nếu `model_name` không bắt đầu bằng `"nvidia/parakeet"` (nhưng vẫn thử khởi tạo recognizer).
2. Khởi tạo `ParakeetRecognizer(model_name, device)`.
3. Lưu instance vào `_recognizer` — các lần gọi sau sẽ thay thế instance cũ.

`get_recognizer_model()` trả về `_recognizer.model_name` hoặc `None`.

---

## ParakeetRecognizer (`app/services/asr/nemo/recognizer.py`)

Bọc `nemo_asr.models.ASRModel`.

**Model được hỗ trợ (`SUPPORTED_MODELS`):**
- `nvidia/parakeet-ctc-0.6b-vi`
- `nvidia/parakeet-tdt-0.6b-v3`

Tên model không hợp lệ sẽ fallback về `SUPPORTED_MODELS[0]` kèm cảnh báo được log.

**Khởi tạo:**
1. Xác định device: `"auto"` → `"cuda"` nếu `torch.cuda.is_available()`, không thì `"cpu"`; fallback về `"cpu"` kèm cảnh báo nếu yêu cầu `"cuda"` nhưng không khả dụng.
2. `nemo_asr.models.ASRModel.from_pretrained(model_name)` — tải từ cache HuggingFace hoặc download.
3. `model.cuda()` nếu device xác định là CUDA.
4. `model.eval()`.

**`transcribe(audio, enable_timestamps=False, precision=3)`**
- Chuẩn hóa input `str`/`Tensor` đơn lẻ thành list một phần tử.
- Gọi trực tiếp `self.model.transcribe(audio, timestamps=enable_timestamps)` — không sắp xếp theo độ dài, không autocast, không tách batch.
- Không có timestamp: trả về `[TranscriptionResult(text=...)]` cho mỗi item.
- Có timestamp: trích `result.timestamp["word"]` và `result.timestamp["segment"]` cho mỗi item, làm tròn mỗi `start`/`end` đến `precision` chữ số, trả về `TranscriptionResult(text, segments, words)` cho mỗi item.
- Khi có exception: log lỗi, input audio, và device, sau đó raise lại.

**Properties:** `model_name`, `supported_models`.

---

## VADFactory (`app/services/vad/factory.py`)

Factory singleton cấp class, tương tự `RecognizerFactory`. `create(model_name, device, **kwargs)`:
1. Nếu `model_name` bắt đầu bằng `"pyannote"`, khởi tạo `PyannoteVADDetector(model_name, token=HF_TOKEN, device=device)`.
2. Ngược lại, log lỗi nhưng vẫn thử khởi tạo `PyannoteVADDetector` với tên đó (bản thân detector sẽ tự fallback nội bộ — xem bên dưới).
3. Lưu instance vào `_detector`.

`get_detector_model()` trả về `_detector.model_name` hoặc `None`.

---

## PyannoteVADDetector (`app/services/vad/pyannote/detector.py`)

Bọc `Model` + pipeline `VoiceActivityDetection` của `pyannote.audio`.

**Model được hỗ trợ (`SUPPORTED_VAD_MODELS`):** chỉ `pyannote/segmentation-3.0`. Tên không nhận diện được sẽ fallback về model mặc định này.

**Khởi tạo:**
1. Tắt TF32 trên cả `torch.backends.cuda.matmul` và `torch.backends.cudnn` để đảm bảo tính tái lập giữa các phiên bản CUDA/cuDNN.
2. Kiểm tra `model_name` với `SUPPORTED_VAD_MODELS`, fallback kèm log lỗi/cảnh báo nếu không hỗ trợ.
3. Xác định device theo cùng cách `auto`/`cuda`/`cpu` như `ParakeetRecognizer`.
4. `load_model()`:
   - `Model.from_pretrained(model_name, token=self._token)` — yêu cầu `HF_TOKEN` hợp lệ vì `pyannote/segmentation-3.0` là model gated.
   - Chuyển model tới device đã xác định.
   - Bọc trong `pyannote.audio.pipelines.VoiceActivityDetection(segmentation=model)`.
   - Khởi tạo pipeline với `min_duration_on=0.0` và `min_duration_off=0.0` (mặc định không lọc thời lượng giọng nói/khoảng lặng tối thiểu; đây là tham số constructor nhưng hiện chưa được expose qua `config.toml`).

**`detect(audio, precision=3) -> List[VADSegment]`** (async)
- Nếu `audio` là `bytes`: giải mã qua `load_audio_from_bytes` (mono, 16kHz) và chạy pipeline trên `{"waveform": waveform.unsqueeze(0), "sample_rate": 16000}`.
- Nếu `audio` là chuỗi đường dẫn file: chạy pipeline trực tiếp trên đường dẫn.
- Duyệt `result.itertracks(yield_label=True)` và tạo mỗi `VADSegment(start, end)` cho mỗi vùng phát hiện, với `start`/`end` làm tròn đến `precision` chữ số thập phân.

**Properties:** `model_name`, `supported_models`.

---

## Audio Utils (`app/utils/audio/io.py`)

**`load_audio_from_bytes(audio_bytes, target_sr=16000, normalize=False) -> torch.Tensor`** (async)
- `torchaudio.load` (chạy qua `asyncio.to_thread`) trên buffer `BytesIO` trong bộ nhớ — raise `ValueError` nếu giải mã thất bại.
- Chuyển về mono qua `waveform.mean(dim=0, keepdim=True)` khi có nhiều hơn 1 channel.
- Resample về `target_sr` nếu cần, dùng `get_resampler(sr, target_sr)` — một cache `dict` cấp module thông thường theo key `(sr, target_sr)` (không phải `functools.lru_cache`).
- Tùy chọn chuẩn hóa biên độ về `[-1, 1]` (mặc định tắt; hiện chưa được caller nào trong branch này sử dụng).
- Trả về tensor 1 chiều (`waveform.squeeze(0)`).

**`is_audio_file(data, buffer_size=2048) -> bool`**
- Dùng `python-magic` để kiểm tra MIME type từ 2048 byte đầu.
- Trả về `True` cho `audio/*` và `application/ogg`; `False` khi có exception hoặc input rỗng.

**`estimate_audio_duration(audio_bytes) -> float`**
- Dùng `soundfile.info()` trên buffer trong bộ nhớ (`frames / samplerate`) để ước lượng thời lượng nhẹ mà không cần giải mã toàn bộ. Được `Pipeline` dùng để điền trường `duration` của response cuối — phản ánh toàn bộ clip gốc, không phải tổng các đoạn giọng nói đã trích xuất.

---

## Batch Processing Helper (`app/utils/transcription/helper.py`)

**`process_batch_transcription(asr_model, audio_data, timestamp_granularities) -> List[TranscriptionResult]`**

Chia batch thành 2 nhóm theo yêu cầu timestamp và gọi recognizer một lần cho mỗi nhóm:

```
ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

no_ts_indices không rỗng → asr_model.transcribe(audio=[...], enable_timestamps=False)
ts_indices không rỗng    → asr_model.transcribe(audio=[...], enable_timestamps=True)

kết quả được ghi vào list output có độ dài len(timestamp_granularities)
tại đúng vị trí index gốc.
```

Nếu input là đường dẫn file thay vì tensor, chúng sẽ được chuyển thành string trước khi truyền vào `transcribe()`.

**`get_timestamp_indices(timestamp_granularities) -> (ts_indices, no_ts_indices)`** — phân chia danh sách giá trị granularity (`None`, `"word"`, `"segment"`) thành 2 danh sách index.

**`get_transcription_type(type) -> TranscriptionType`** — `None` → `Text`, `"word"` → `Word`, `"segment"` → `Segment`, giá trị khác → `Text`.

---

## Response Schemas (`app/schema/transcription/`, `app/schema/segment/`)

| Class | Dùng khi | Các trường |
|---|---|---|
| `TranscriptionResponse` | `timestamp_granularity=None` | `text`, `usage` (`Usage` — số lượng token) |
| `WordResponse` (kế thừa `BaseResponse`) | `timestamp_granularity="word"` | `task`, `language`, `duration`, `text`, `usage` (`DurationUsage` — giây), `words` |
| `SegmentResponse` (kế thừa `BaseResponse`) | `timestamp_granularity="segment"` | `task`, `language`, `duration`, `text`, `usage` (`DurationUsage`), `segments` |
| `TranscriptionResult` | nội bộ, output ASR theo từng đoạn | `text`, `words` tùy chọn, `segments` tùy chọn |

`TranscribedWord`: `start`, `end`, `word`
`AdvancedTranscribedSegment`: `id`, `start`, `end`, `text`
`VADSegment` (kế thừa `TimeSegment`): `start`, `end`, `confidence` tùy chọn

---

## Configuration Loader (`app/utils/config_loader/toml_loader.py`)

`TomlConfigLoader` đọc `config/config.toml` (đường dẫn mặc định xác định tương đối theo module) và expose từng section qua `get_section(name)` hoặc dot-notation `get(key)`. Các module `app/core/config/*.py` gọi `get_toml_config().get_section(...)` một lần khi import để điền các hằng số cấp module, sau đó được các module deployment import bằng `from ... import *`.

---

## Exception Handlers (`app/exceptions/`)

| Exception | Điều kiện kích hoạt | HTTP status |
|---|---|---|
| `TranscriptedModelNotFoundException` | Định nghĩa cho việc kiểm tra model được yêu cầu (lỗi kiểu OpenAI `model_not_found`) | 400 |
| `UnsupportedAudioFormatException` | Kiểm tra MIME `is_audio_file()` thất bại trong `IngressDeployment.transcribe_audio()` | 400 |

Cả hai được đăng ký qua `ingress_app.add_exception_handler(...)` và render bởi `common_exception_handler`, serialize `BaseResponse` của exception (kiểu OpenAI `message` / `type` / `param` / `code`) thành JSON.

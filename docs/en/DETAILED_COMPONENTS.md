# Core Components

## IngressDeployment (`app/services/deployments/ingress_deployment.py`)

Ray Serve deployment that owns the FastAPI HTTP layer. It holds no models — it only validates input and delegates to `Pipeline`.

**Deployment config** (`@serve.deployment`):
- `ray_actor_options={"num_gpus": INGRESS_NUM_GPUS}` — `0` by default (CPU-only)
- `num_replicas=INGRESS_NUM_REPLICAS`
- `max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS`

**`transcribe_audio()`** — FastAPI endpoint handler (`POST /v1/audio/transcriptions`)
- Reads the uploaded file into memory (`await file.read()`)
- Validates audio format via `is_audio_file()` — raises `UnsupportedAudioFormatException(file_extension)` if the MIME sniff fails
- Forwards to `Pipeline.remote(audio_bytes, timestamp_granularity)` and awaits the result
- Returns the result object directly (already a `TranscriptionResponse` / `WordResponse` / `SegmentResponse`)

Registers exception handlers for `TranscriptedModelNotFoundException` and `UnsupportedAudioFormatException` via `common_exception_handler`.

---

## Pipeline (`app/services/deployments/ingress_deployment.py`)

Plain `@serve.deployment` actor (no `@serve.batch`, no HTTP ingress) that owns the VAD→ASR orchestration logic. Constructed with handles to both `VADDeployment` and `ASRDeployment`.

**`__call__(audio_bytes, timestamp_granularity)`**
1. `load_audio_from_bytes(audio_bytes)` — decodes the full audio once into a mono, 16kHz waveform tensor.
2. `self.vad.remote(audio_bytes)` — awaits the list of `VADSegment` objects from `VADDeployment`.
3. If no segments were detected, returns an empty response of the appropriate type (`TranscriptionResponse`, `WordResponse`, or `SegmentResponse`) immediately.
4. `self._extract_audio_segments(audio_tensor, vad_segments)` — slices the waveform into one tensor per speech segment.
5. Fires one `self.asr.batched_transcribe_tensors.remote(segment, timestamp_granularity)` call per segment and awaits them all with `asyncio.gather`.
6. Joins all segment texts with spaces into `combined_text`.
7. Builds the final response, adjusting word/segment timestamps by adding back each segment's VAD start offset (see [Timestamp Offset Adjustment](#timestamp-offset-adjustment) below).

**`_extract_audio_segments(audio_tensor, vad_segments, sample_rate=16000)`**
- Converts each `VADSegment.start` / `.end` (seconds) to sample indices: `int(seconds * sample_rate)`
- Clamps `start_idx` to `>= 0` and `end_idx` to `<= len(audio_tensor)`
- Skips (and logs a warning for) any segment where `end_idx <= start_idx`
- Returns a `List[torch.Tensor]`, one 1-D tensor per valid speech region

### Timestamp Offset Adjustment

ASR timestamps are computed independently per segment and are therefore relative to the start of that segment's audio, not the original file. The pipeline restores absolute time by adding the corresponding `VADSegment.start` to every timestamp:

```python
adjusted_word = word.model_copy(update={
    "start": word.start + vad_seg.start,
    "end": word.end + vad_seg.start,
})
```

The same offset is applied to segment-level timestamps, and segment `id`s are reassigned sequentially (`id=len(all_segments)`) across the whole response rather than reusing each ASR segment's own id.

---

## VADDeployment (`app/services/deployments/vad_deployment.py`)

Ray Serve deployment wrapping the VAD model.

**Deployment config:**
- `ray_actor_options={"num_gpus": VAD_NUM_GPUS}`
- `num_replicas=VAD_NUM_REPLICAS`
- `max_ongoing_requests=VAD_MAX_ONGOING_REQUESTS`

**`__init__()`** — calls `VADFactory.create(model_name=VAD_MODEL_NAME, device=VAD_DEVICE, token=HF_TOKEN)`; raises `RuntimeError` if the factory returns `None`.

**`__call__(audio_bytes) -> List[VADSegment]`** — delegates to `self._vad_model.detect(audio=audio_bytes, precision=3)` and logs the number of segments detected. This is a plain deployment call (not `@serve.batch`) — each HTTP request triggers its own VAD inference call, though Ray Serve still load-balances calls across `VAD_NUM_REPLICAS` replicas.

---

## ASRDeployment (`app/services/deployments/asr_deployment.py`)

Ray Serve deployment wrapping the ASR model, with two batched entry points.

**Deployment config:**
- `ray_actor_options={"num_gpus": ASR_NUM_GPUS}`
- `num_replicas=ASR_NUM_REPLICAS`
- `max_ongoing_requests=ASR_MAX_ONGOING_REQUESTS`

**`__init__()`** — calls `RecognizerFactory.create(model_name=ASR_MODEL_NAME, device=ASR_DEVICE)`; raises `RuntimeError` if the factory returns `None`.

**`batched_transcribe(batch: List[bytes], timestamp_granularities)`** — `@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)`. Decodes each item with `load_audio_from_bytes` (parallel via `asyncio.gather`), then calls `process_batch_transcription()`. Used for direct bytes-in ASR calls (not exercised by the VAD pipeline, which already holds decoded tensors).

**`batched_transcribe_tensors(batch: List[torch.Tensor], timestamp_granularities)`** — same `@serve.batch` decorator; skips the decode step entirely since the caller (the `Pipeline` deployment) has already sliced and provided waveform tensors. This is the method the VAD+ASR pipeline actually calls, once per detected speech segment. Because `@serve.batch` operates on the deployment level, segments from unrelated concurrent HTTP requests can be folded into the same underlying `ASRDeployment` batch.

**`__call__(audio_bytes, timestamp_granularity=None)`** — convenience direct-call path that awaits `self.batched_transcribe.remote(...)`; not used by the VAD pipeline.

---

## RecognizerFactory (`app/services/asr/factory.py`)

Class-level singleton factory. `create(model_name, device)`:
1. Logs an error if `model_name` does not start with `"nvidia/parakeet"` (but still attempts to construct the recognizer).
2. Instantiates `ParakeetRecognizer(model_name, device)`.
3. Stores the instance in `_recognizer` — subsequent calls replace it.

`get_recognizer_model()` returns `_recognizer.model_name` or `None`.

---

## ParakeetRecognizer (`app/services/asr/nemo/recognizer.py`)

Wraps `nemo_asr.models.ASRModel`.

**Supported models (`SUPPORTED_MODELS`):**
- `nvidia/parakeet-ctc-0.6b-vi`
- `nvidia/parakeet-tdt-0.6b-v3`

Unknown model names fall back to `SUPPORTED_MODELS[0]` with a logged warning.

**Initialization:**
1. Resolves device: `"auto"` → `"cuda"` if `torch.cuda.is_available()` else `"cpu"`; falls back to `"cpu"` with a warning if `"cuda"` was requested but unavailable.
2. `nemo_asr.models.ASRModel.from_pretrained(model_name)` — downloads/loads from the HuggingFace cache.
3. `model.cuda()` if the resolved device is CUDA.
4. `model.eval()`.

**`transcribe(audio, enable_timestamps=False, precision=3)`**
- Normalizes a single `str`/`Tensor` input into a one-item list.
- Calls `self.model.transcribe(audio, timestamps=enable_timestamps)` directly — no length sorting, no autocast, no batch splitting.
- Without timestamps: returns `[TranscriptionResult(text=...)]` per item.
- With timestamps: extracts `result.timestamp["word"]` and `result.timestamp["segment"]` per item, rounds each `start`/`end` to `precision` decimals, and returns `TranscriptionResult(text, segments, words)` per item.
- On exception, logs the error, the audio input, and the device, then re-raises.

**Properties:** `model_name`, `supported_models`.

---

## VADFactory (`app/services/vad/factory.py`)

Class-level singleton factory, mirroring `RecognizerFactory`. `create(model_name, device, **kwargs)`:
1. If `model_name` starts with `"pyannote"`, constructs `PyannoteVADDetector(model_name, token=HF_TOKEN, device=device)`.
2. Otherwise logs an error and still attempts to construct a `PyannoteVADDetector` with the given name (which will itself fall back internally — see below).
3. Stores the instance in `_detector`.

`get_detector_model()` returns `_detector.model_name` or `None`.

---

## PyannoteVADDetector (`app/services/vad/pyannote/detector.py`)

Wraps `pyannote.audio`'s `Model` + `VoiceActivityDetection` pipeline.

**Supported models (`SUPPORTED_VAD_MODELS`):** `pyannote/segmentation-3.0` only. Unrecognized names fall back to this default.

**Initialization:**
1. Disables TF32 on both `torch.backends.cuda.matmul` and `torch.backends.cudnn` for reproducibility across CUDA/cuDNN versions.
2. Validates `model_name` against `SUPPORTED_VAD_MODELS`, falling back with a logged error/warning if unsupported.
3. Resolves device the same `auto`/`cuda`/`cpu` way as `ParakeetRecognizer`.
4. `load_model()`:
   - `Model.from_pretrained(model_name, token=self._token)` — requires a valid `HF_TOKEN` since `pyannote/segmentation-3.0` is a gated model.
   - Moves the model to the resolved device.
   - Wraps it in `pyannote.audio.pipelines.VoiceActivityDetection(segmentation=model)`.
   - Instantiates the pipeline with `min_duration_on=0.0` and `min_duration_off=0.0` (no minimum speech/gap filtering by default; these are constructor parameters but not currently exposed via `config.toml`).

**`detect(audio, precision=3) -> List[VADSegment]`** (async)
- If `audio` is `bytes`: decodes via `load_audio_from_bytes` (mono, 16kHz) and runs the pipeline on `{"waveform": waveform.unsqueeze(0), "sample_rate": 16000}`.
- If `audio` is a file path string: runs the pipeline directly on the path.
- Iterates `result.itertracks(yield_label=True)` and builds one `VADSegment(start, end)` per detected region, with `start`/`end` rounded to `precision` decimal places.

**Properties:** `model_name`, `supported_models`.

---

## Audio Utils (`app/utils/audio/io.py`)

**`load_audio_from_bytes(audio_bytes, target_sr=16000, normalize=False) -> torch.Tensor`** (async)
- `torchaudio.load` (run via `asyncio.to_thread`) on an in-memory `BytesIO` buffer — raises `ValueError` on decode failure.
- Downmixes to mono via `waveform.mean(dim=0, keepdim=True)` when more than one channel is present.
- Resamples to `target_sr` if needed, using `get_resampler(sr, target_sr)` — a plain module-level `dict` cache keyed by `(sr, target_sr)` (not `functools.lru_cache`).
- Optional peak normalization to `[-1, 1]` (off by default; not currently used by any caller in this branch).
- Returns a 1-D tensor (`waveform.squeeze(0)`).

**`is_audio_file(data, buffer_size=2048) -> bool`**
- Uses `python-magic` to sniff the MIME type of the first 2048 bytes.
- Returns `True` for `audio/*` and `application/ogg`; `False` on any exception or empty input.

**`estimate_audio_duration(audio_bytes) -> float`**
- Uses `soundfile.info()` on an in-memory buffer (`frames / samplerate`) for a lightweight duration estimate without a full decode. Used by `Pipeline` to populate the `duration` field of the final response — this reflects the full original clip, not the sum of extracted speech segments.

---

## Batch Processing Helper (`app/utils/transcription/helper.py`)

**`process_batch_transcription(asr_model, audio_data, timestamp_granularities) -> List[TranscriptionResult]`**

Splits a batch into two groups by timestamp requirement and calls the recognizer once per group:

```
ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

no_ts_indices non-empty → asr_model.transcribe(audio=[...], enable_timestamps=False)
ts_indices non-empty    → asr_model.transcribe(audio=[...], enable_timestamps=True)

results are written back into an output list of length len(timestamp_granularities)
at their original global indices.
```

If the input items are file paths rather than tensors, they are stringified before being passed to `transcribe()`.

**`get_timestamp_indices(timestamp_granularities) -> (ts_indices, no_ts_indices)`** — partitions a list of granularity values (`None`, `"word"`, `"segment"`) into two index lists.

**`get_transcription_type(type) -> TranscriptionType`** — `None` → `Text`, `"word"` → `Word`, `"segment"` → `Segment`, anything else → `Text`.

---

## Response Schemas (`app/schema/transcription/`, `app/schema/segment/`)

| Class | Used when | Fields |
|---|---|---|
| `TranscriptionResponse` | `timestamp_granularity=None` | `text`, `usage` (`Usage` — token counts) |
| `WordResponse` (extends `BaseResponse`) | `timestamp_granularity="word"` | `task`, `language`, `duration`, `text`, `usage` (`DurationUsage` — seconds), `words` |
| `SegmentResponse` (extends `BaseResponse`) | `timestamp_granularity="segment"` | `task`, `language`, `duration`, `text`, `usage` (`DurationUsage`), `segments` |
| `TranscriptionResult` | internal, per-segment ASR output | `text`, optional `words`, optional `segments` |

`TranscribedWord`: `start`, `end`, `word`
`AdvancedTranscribedSegment`: `id`, `start`, `end`, `text`
`VADSegment` (extends `TimeSegment`): `start`, `end`, optional `confidence`

---

## Configuration Loader (`app/utils/config_loader/toml_loader.py`)

`TomlConfigLoader` reads `config/config.toml` (default path resolved relative to the module) and exposes sections via `get_section(name)` or dot-notation `get(key)`. `app/core/config/*.py` modules call `get_toml_config().get_section(...)` once at import time to populate module-level constants, which are then imported with `from ... import *` by the deployment modules.

---

## Exception Handlers (`app/exceptions/`)

| Exception | Trigger | HTTP status |
|---|---|---|
| `TranscriptedModelNotFoundException` | Defined for requested-model validation (OpenAI-style `model_not_found` error) | 400 |
| `UnsupportedAudioFormatException` | `is_audio_file()` MIME check fails in `IngressDeployment.transcribe_audio()` | 400 |

Both are registered via `ingress_app.add_exception_handler(...)` and rendered by `common_exception_handler`, which serializes the exception's `BaseResponse` (OpenAI-style `message` / `type` / `param` / `code`) as JSON.

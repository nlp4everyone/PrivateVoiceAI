# Core Components

## ASRService (`app/services/deployments/asr_deployment.py`)

Ray Serve deployment that owns the FastAPI ingress and the batch transcription logic.

**Deployment config** (set via `@serve.deployment`):
- `num_replicas=NUM_REPLICAS` — number of independent replicas; each replica holds one model copy
- `num_gpus=NUM_GPUS` — GPU resources reserved per replica
- `max_ongoing_requests=MAX_ONGOING_REQUESTS` — maximum in-flight requests per replica before Ray Serve applies backpressure

**`__init__()`**
- Re-applies `ray.serve` log level to INFO (Ray Serve resets it during actor initialization)
- Calls `RecognizerFactory.create()` to load the ASR model
- Caches the deployment handle (`serve.get_deployment_handle`) for self-routing into the batch queue
- Creates a `ThreadPoolExecutor(max_workers=1)` — a single dedicated thread keeps all GPU work serialized and prevents CUDA context migration across threads

**`transcribe_audio()`** — FastAPI endpoint handler
- Validates the requested model matches the loaded model
- Validates audio format via MIME check
- Forwards to `batched_transcribe` via `.remote()` and awaits the result
- Formats the response based on `timestamp_granularity`

**`batched_transcribe()`** — `@serve.batch` handler
- Receives a batch of `(audio_bytes, granularity)` pairs aggregated by Ray Serve
- Decodes audio in parallel with `asyncio.to_thread`
- Frees raw bytes immediately after decoding to reduce peak memory
- Dispatches to `process_batch_transcription()` on the GPU executor

---

## RecognizerFactory (`app/services/asr/factory.py`)

Thin factory with class-level singleton storage. `create(model_name, device)`:
1. Logs a warning if `model_name` does not start with `"nvidia/parakeet"` (but proceeds anyway)
2. Instantiates `ParakeetRecognizer` and returns it
3. Stores the instance in `_recognizer` — subsequent calls replace it

`get_recognizer_model()` — returns `_recognizer.model_name` or `None`.

---

## ParakeetRecognizer (`app/services/asr/nemo/recognizer.py`)

Wraps `nemo_asr.models.ASRModel` with sorting, autocast, and structured output.

**Supported models:**
- `nvidia/parakeet-ctc-0.6b-vi`
- `nvidia/parakeet-tdt-0.6b-v3`

Unknown model names fall back to `SUPPORTED_MODELS[0]`.

**Initialization:**
1. Resolves device: `"auto"` → `"cuda"` if `torch.cuda.is_available()` else `"cpu"`
2. `nemo_asr.models.ASRModel.from_pretrained(model_name)` — downloads/loads from HF cache
3. `model.cuda()` if CUDA
4. `model.eval()`

**`transcribe(audio, enable_timestamps, precision=3)`**
- Normalizes single inputs to `[item]`
- If batch is multi-item tensors: sorts ascending by length, records the permutation
- Runs inference under `torch.cuda.amp.autocast()`
- Restores original order after inference
- Returns `List[TranscriptionResult]`; timestamps are rounded to `precision` decimal places

**Properties:** `model_name`, `supported_models`

---

## Audio Utils (`app/utils/audio/io.py`)

**`load_audio_from_bytes(audio_bytes, target_sr=16000) → (Tensor, float)`**
- `torchaudio.load(BytesIO(audio_bytes))` — supports MP3, WAV, FLAC, OGG, and other torchaudio-supported formats
- Mono: `waveform[0]` for single-channel, `waveform.mean(dim=0)` for multi-channel
- Resamples if `sr != target_sr` using a cached `torchaudio.transforms.Resample` (keyed by `(sr_src, sr_tgt)` via `lru_cache(maxsize=8)`)
- Returns `(waveform, duration_seconds)`

**`is_audio_file(data, buffer_size=2048) → bool`**
- Uses `python-magic` to sniff MIME type from the first 2048 bytes
- Returns `True` for `audio/*` and `application/ogg`; returns `False` otherwise and on any exception

**`estimate_audio_duration(audio_bytes) → float`**
- Uses `soundfile.info()` for a lightweight duration estimate without full decode

---

## Batch Processing Helper (`app/utils/transcription/helper.py`)

**`process_batch_transcription(asr_model, audio_data, timestamp_granularities, split_mixed_batch=True)`**

Dispatches to 1 or 2 GPU calls based on the batch composition:

```
ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

Pure no-ts:  asr_model.transcribe(all, enable_timestamps=False)         → 1 GPU call
Pure ts:     asr_model.transcribe(all, enable_timestamps=True)          → 1 GPU call
Mixed (split_mixed_batch=True):
    asr_model.transcribe(ts_items, enable_timestamps=True)              → GPU call 1
    asr_model.transcribe(no_ts_items, enable_timestamps=False)          → GPU call 2
    merge results back to original indices
Mixed (split_mixed_batch=False):
    asr_model.transcribe(all, enable_timestamps=True)                   → 1 GPU call
    strip .words / .segments from no-ts results
```

**`get_timestamp_indices(timestamp_granularities) → (ts_indices, no_ts_indices)`**
- Partitions a list of granularity strings (`None`, `"word"`, `"segment"`) into two index lists

**`get_transcription_type(type) → TranscriptionType`**
- `None` → `Text`, `"word"` → `Word`, `"segment"` → `Segment`

---

## Response Schemas (`app/schema/transcription/`)

| Class | Used when | Fields |
|---|---|---|
| `TranscriptionResponse` | `timestamp_granularity=None` | `text`, `usage` (token counts) |
| `WordResponse` | `timestamp_granularity="word"` | `text`, `language`, `duration`, `usage` (seconds), `words` |
| `SegmentResponse` | `timestamp_granularity="segment"` | `text`, `language`, `duration`, `usage` (seconds), `segments` |

`TranscribedWord`: `start`, `end`, `word`
`AdvancedTranscribedSegment`: `id`, `start`, `end`, `text`

---

## Configuration Loader (`app/utils/config_loader/toml_loader.py`)

Reads `config/config.toml` and exposes sections as dicts. Used by `app/core/config/*.py` to populate module-level constants imported via `*`.

---

## Exception Handlers (`app/exceptions/`)

| Exception | Trigger | HTTP response |
|---|---|---|
| `TranscriptedModelNotFoundException` | Requested model ≠ loaded model | 404 |
| `UnsupportedAudioFormatException` | MIME check fails | 415 |

Both are registered via `asr_app.add_exception_handler()` and handled by `common_exception_handler`.

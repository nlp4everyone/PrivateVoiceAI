# Technical Overview

## Architecture

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
│  │  num_gpus=INGRESS_NUM_GPUS (0 — CPU only)                     │    │
│  │  max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS             │    │
│  │                                                                │    │
│  │  POST /v1/audio/transcriptions                                 │    │
│  │      │ read bytes, is_audio_file() MIME check                  │    │
│  │      │ pipeline.remote(audio_bytes, timestamp_granularity)     │    │
│  └──────────────────────────┬───────────────────────────────────┘    │
│                              ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Pipeline  (@serve.deployment, plain actor — not @serve.batch)│    │
│  │      │ load_audio_from_bytes(audio_bytes) → waveform tensor    │    │
│  │      ▼                                                        │    │
│  │  VADDeployment.remote(audio_bytes)                             │    │
│  │      │ PyannoteVADDetector.detect() → List[VADSegment]         │    │
│  │      ▼                                                        │    │
│  │  slice waveform per segment → List[Tensor]                     │    │
│  │      ▼                                                        │    │
│  │  ASRDeployment.batched_transcribe_tensors.remote(seg, gran) ×N │    │
│  │      (fan-out: one call per speech segment, awaited together)  │    │
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

Four independent Ray Serve deployments make up the graph: `IngressDeployment` (HTTP layer), `Pipeline` (orchestration), `VADDeployment`, and `ASRDeployment`. Each is bound together in `bind_app()` and can be scaled, GPU-allocated, and configured independently.

---

## Processing Pipeline

### Stage 1 — Request validation (Ingress)

`IngressDeployment.transcribe_audio()` receives the uploaded file and:
1. Reads all audio bytes into memory.
2. Validates MIME type via `python-magic` (`is_audio_file()`) — raises `UnsupportedAudioFormatException` on non-audio input.
3. Forwards `(audio_bytes, timestamp_granularity)` to the `Pipeline` deployment via `.remote()` and awaits the result.

The requested `model` form field is accepted but not currently cross-checked against the loaded ASR model in the ingress layer (unlike the plain-ASR variant of this service).

### Stage 2 — Full-audio decode + VAD (Pipeline + VADDeployment)

`Pipeline.__call__()`:
1. Decodes the full audio once with `load_audio_from_bytes()` into a mono 16kHz waveform tensor — this copy is used later for segment slicing.
2. Calls `VADDeployment.remote(audio_bytes)`, which runs `PyannoteVADDetector.detect()`. Internally the detector re-decodes the bytes itself (via `load_audio_from_bytes`) and runs `pyannote.audio`'s `VoiceActivityDetection` pipeline (built on `pyannote/segmentation-3.0`), returning a list of `VADSegment(start, end)` in seconds, rounded to 3 decimal places.
3. If VAD finds no speech, the pipeline short-circuits and returns an empty `TranscriptionResponse` / `WordResponse` / `SegmentResponse` depending on the requested granularity.

### Stage 3 — Segment extraction

`Pipeline._extract_audio_segments()` converts each `VADSegment`'s start/end (seconds) into sample indices against the 16kHz waveform, clamps them to the tensor bounds, and slices out one 1-D tensor per speech region. Segments with an invalid (non-positive) range are dropped with a warning.

### Stage 4 — Cross-request ASR batching (ASRDeployment)

Each extracted segment tensor is sent independently to `ASRDeployment.batched_transcribe_tensors.remote(segment, timestamp_granularity)`, and all calls for one HTTP request are awaited together with `asyncio.gather`. Because this method is decorated with `@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)`, Ray Serve transparently folds segments coming from *different* concurrent HTTP requests into the same GPU batch — batching happens at the segment level, not the request level.

`process_batch_transcription()` splits the batch into a with-timestamps group and a without-timestamps group (each calls `ParakeetRecognizer.transcribe()` once) and reassembles results in original order. `ASRDeployment` also exposes `batched_transcribe` (bytes-in) for direct, non-VAD ASR calls, but the VAD pipeline always uses the tensor-based `batched_transcribe_tensors` path to avoid a redundant decode.

### Stage 5 — Timestamp offset adjustment and response formatting (Pipeline)

Each per-segment `TranscriptionResult` carries timestamps that are relative to the start of its own (sliced) segment. The pipeline restores absolute timestamps by adding each `VADSegment.start` back onto every word/segment timestamp before building the final response:

- `None` → `TranscriptionResponse` — concatenated text + approximate token usage (`tiktoken`-based)
- `"word"` → `WordResponse` — text, detected language, duration, word-level timestamps (offset-adjusted)
- `"segment"` → `SegmentResponse` — text, detected language, duration, segment-level timestamps (offset-adjusted, re-indexed by `id`)

Language detection uses `pycld2` and runs only for timestamp responses. Duration is estimated from the original audio bytes via `soundfile.info()` (`estimate_audio_duration`), not from the sum of speech-segment durations.

---

## Configuration

All parameters live in `config/config.toml` (stable, per-deployment topology and model choice) plus `.env` (ports, secrets). Changes require a container/service restart.

### `[system]`

| Key | Default | Description |
|---|---|---|
| `AUDIO_TEMP_DIR` | `/dev/shm` | Scratch directory (shared-memory tmpfs) |
| `DEPLOYMENT_NAME` | `ASRService` | Logical name for the ASR deployment |
| `VAD_DEPLOYMENT_NAME` | `VADService` | Logical name for the VAD deployment |
| `RAY_HOST` | `0.0.0.0` | HTTP host for Ray Serve |
| `RAY_PORT` | `8000` | HTTP port for Ray Serve (container-internal) |

### `[asr]`

| Key | Default | Description |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | HuggingFace model identifier |
| `ASR_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `ASR_NUM_GPUS` | `0.2` | Fractional GPU allocated per ASR replica |
| `ASR_NUM_REPLICAS` | `1` | Number of `ASRDeployment` replicas |
| `ASR_MAX_ONGOING_REQUESTS` | `8` | Max in-flight requests per ASR replica |
| `ASR_MAX_BATCH_SIZE` | `8` | Max segments per GPU batch |
| `ASR_BATCH_WAIT_TIMEOUT_S` | `0.1` | Max wait time to fill an ASR batch (seconds) |

### `[vad]`

| Key | Default | Description |
|---|---|---|
| `VAD_MODEL_NAME` | `pyannote/segmentation-3.0` | HuggingFace Pyannote segmentation model |
| `VAD_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VAD_NUM_GPUS` | `0.1` | Fractional GPU allocated per VAD replica |
| `VAD_NUM_REPLICAS` | `4` | Number of `VADDeployment` replicas |
| `VAD_MAX_ONGOING_REQUESTS` | `8` | Max in-flight requests per VAD replica |

### `[ingress]`

| Key | Default | Description |
|---|---|---|
| `INGRESS_NUM_GPUS` | `0` | GPUs allocated per ingress replica (CPU-only layer) |
| `INGRESS_NUM_REPLICAS` | `1` | Number of `IngressDeployment` replicas |
| `INGRESS_MAX_ONGOING_REQUESTS` | `16` | Max in-flight HTTP requests per ingress replica |

### Environment (`.env` / `docker-compose.yml`)

| Variable | Description |
|---|---|
| `RAY_FASTAPI_PORT` | Host port mapped to Ray Serve HTTP (default `8000`, sample `.env` uses `8005`) |
| `RAY_DASHBOARD_PORT` | Host port mapped to Ray dashboard (default `8265`) |
| `HF_TOKEN` | Hugging Face access token — required to download the gated `pyannote/segmentation-3.0` model |
| `HF_HOME` | HuggingFace cache directory (set inside the container to the mounted cache volume) |

See [CONFIGURATION.md](../CONFIGURATION.md) for the full flattened parameter reference.

---

## Repository Structure

```
VoicePlatform/
├── app/
│   ├── app.py                                # Entry point: binds deployment graph, ray.init/serve.start
│   ├── core/config/
│   │   ├── asr.py                            # ASR_MODEL_NAME, ASR_DEVICE
│   │   ├── vad.py                            # VAD_MODEL_NAME, VAD_DEVICE, HF_TOKEN
│   │   ├── serving.py                        # ASR_/VAD_/INGRESS_ replica & batch settings
│   │   └── system.py                         # RAY_HOST, RAY_PORT, DEPLOYMENT_NAME, VAD_DEPLOYMENT_NAME
│   ├── services/
│   │   ├── asr/
│   │   │   ├── factory.py                    # RecognizerFactory — creates ParakeetRecognizer
│   │   │   └── nemo/recognizer.py            # ParakeetRecognizer — wraps NeMo ASRModel
│   │   ├── vad/
│   │   │   ├── factory.py                    # VADFactory — creates PyannoteVADDetector
│   │   │   └── pyannote/detector.py          # PyannoteVADDetector — wraps pyannote.audio pipeline
│   │   └── deployments/
│   │       ├── asr_deployment.py             # ASRDeployment — batched ASR transcription
│   │       ├── vad_deployment.py             # VADDeployment — speech segment detection
│   │       └── ingress_deployment.py         # Pipeline + IngressDeployment — orchestration + FastAPI
│   ├── utils/
│   │   ├── audio/io.py                       # load_audio_from_bytes, is_audio_file, estimate_audio_duration
│   │   ├── transcription/helper.py           # process_batch_transcription, get_timestamp_indices
│   │   ├── language_detect/detector.py       # LanguageDetector (pycld2)
│   │   └── token_counter/token_counter.py    # approximate_count_tokens (tiktoken)
│   ├── schema/
│   │   ├── segment/base.py                   # TimeSegment, VADSegment
│   │   ├── transcription/                    # TranscriptionResult, response types, TranscriptionType
│   │   └── vad/                               # BaseVADDetector, VAD response schemas
│   └── exceptions/                            # TranscriptedModelNotFoundException, UnsupportedAudioFormatException
├── config/config.toml                         # [system] [asr] [vad] [ingress] sections
├── docker/
│   ├── Dockerfile                             # CUDA 12.8 + NeMo 2.7.2 + Ray Serve image
│   └── docker-compose.yml                     # Single-container GPU deployment
└── examples/                                  # audio_transcription_example.py, concurrent_requests_example.py
```

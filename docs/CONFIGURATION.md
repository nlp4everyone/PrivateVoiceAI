# Configuration Reference

Config is loaded in priority order (highest → lowest):

1. Environment variables (Docker `-e` flags, CI)
2. `.env` file (local dev / Docker Compose, not version-controlled) — secrets and environment-specific values: ports, `HF_TOKEN`
3. `config/config.toml` — stable deployment topology, model selection, batching parameters (version-controlled)
4. Field defaults in `app/core/config/*.py`

`TomlConfigLoader` (`app/utils/config_loader/toml_loader.py`) reads `config/config.toml` from a fixed path resolved relative to the module — there is currently no environment override for the TOML file location.

## `[system]`

| Parameter | Default | Description |
|---|---|---|
| `AUDIO_TEMP_DIR` | `/dev/shm` | Scratch directory for temporary audio data (shared-memory tmpfs) |
| `DEPLOYMENT_NAME` | `ASRService` | Logical name associated with the ASR deployment |
| `VAD_DEPLOYMENT_NAME` | `VADService` | Logical name associated with the VAD deployment |
| `RAY_HOST` | `0.0.0.0` | Bind host for the Ray Serve HTTP proxy |
| `RAY_PORT` | `8000` | Bind port for the Ray Serve HTTP proxy (container-internal) |

## `[asr]`

| Parameter | Default | Description |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | HuggingFace model identifier loaded by `RecognizerFactory` / `ParakeetRecognizer`. Also supports `nvidia/parakeet-tdt-0.6b-v3` |
| `ASR_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` — `auto` resolves to `cuda` if available |
| `ASR_NUM_GPUS` | `0.2` | Fractional GPU allocated per `ASRDeployment` replica (Ray `ray_actor_options`) |
| `ASR_NUM_REPLICAS` | `1` | Number of `ASRDeployment` replicas |
| `ASR_MAX_ONGOING_REQUESTS` | `8` | Max in-flight requests per ASR replica before Ray Serve backpressure |
| `ASR_MAX_BATCH_SIZE` | `8` | Max items per `@serve.batch` GPU batch (applies to both `batched_transcribe` and `batched_transcribe_tensors`) |
| `ASR_BATCH_WAIT_TIMEOUT_S` | `0.1` | Max wait time to fill an ASR batch, in seconds |

## `[vad]`

| Parameter | Default | Description |
|---|---|---|
| `VAD_MODEL_NAME` | `pyannote/segmentation-3.0` | HuggingFace Pyannote segmentation model loaded by `VADFactory` / `PyannoteVADDetector` (currently the only supported model) |
| `VAD_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `VAD_NUM_GPUS` | `0.1` | Fractional GPU allocated per `VADDeployment` replica |
| `VAD_NUM_REPLICAS` | `4` | Number of `VADDeployment` replicas — set higher than ASR replicas by default since VAD inference is comparatively cheap |
| `VAD_MAX_ONGOING_REQUESTS` | `8` | Max in-flight requests per VAD replica |

`min_duration_on` and `min_duration_off` (minimum speech/gap durations used by the Pyannote `VoiceActivityDetection` pipeline) are hardcoded to `0.0` in `PyannoteVADDetector.load_model()` and are not currently exposed as config parameters.

## `[ingress]`

| Parameter | Default | Description |
|---|---|---|
| `INGRESS_NUM_GPUS` | `0` | GPUs allocated per `IngressDeployment` replica — `0` since the ingress layer is CPU-only (HTTP handling + orchestration) |
| `INGRESS_NUM_REPLICAS` | `1` | Number of `IngressDeployment` replicas |
| `INGRESS_MAX_ONGOING_REQUESTS` | `16` | Max in-flight HTTP requests per ingress replica |

## Environment variables (`.env` / `docker-compose.yml`)

| Variable | Default | Description |
|---|---|---|
| `RAY_FASTAPI_PORT` | `8000` (sample `.env` uses `8005`) | Host port mapped to the container's Ray Serve HTTP port |
| `RAY_DASHBOARD_PORT` | `8265` | Host port mapped to the Ray Serve dashboard |
| `HF_TOKEN` | *(empty)* | Hugging Face access token — **required** to download the gated `pyannote/segmentation-3.0` model; read directly via `os.getenv("HF_TOKEN", "")` in `app/core/config/vad.py`, not from `config.toml` |
| `HF_HOME` | `/home/ray/.cache/huggingface` (set in `docker-compose.yml`) | HuggingFace cache directory inside the container, backed by a host volume mount |

All `config.toml` values are also overridable by setting the corresponding environment variable if your deployment tooling injects them before the TOML file is read; however, in this branch the TOML loader does not automatically look up environment overrides per key — environment variables only take effect for the values explicitly read via `os.getenv()` (currently just `HF_TOKEN`).

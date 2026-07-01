# Design Decisions

---

## 1. Two-Stage VAD + ASR Composition Pipeline

### Description

Instead of sending raw audio directly to the ASR model, a dedicated `VADDeployment` (Pyannote Audio, `pyannote/segmentation-3.0`) first detects speech regions. The `Pipeline` deployment slices the audio into per-segment tensors and only those are sent to `ASRDeployment`.

### Pros

- **Resource efficiency** — silence, background noise, and non-speech audio never reach the (comparatively expensive) ASR model.
- **Better scaling for long/sparse recordings** — a long file with sparse speech costs roughly proportional to speech duration, not total file duration.
- **Cleaner timestamps** — ASR only has to align words/segments within a known speech region rather than across silence gaps.

### Cons

- **Extra network hop and model** — every request pays for a VAD inference call before ASR can start; for short, dense-speech clips this adds latency compared to sending audio straight to ASR.
- **Segment-boundary artifacts** — words at the exact edge of a VAD-detected segment can be clipped if VAD under-estimates segment boundaries; no padding is currently added around segments.
- **Duration reported is whole-file, not speech-only** — `estimate_audio_duration()` measures the original audio, which can be a source of confusion if callers expect "processed audio" duration.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Single ASR-only pipeline (no VAD) | Simpler, but wastes GPU compute on silence-heavy audio and doesn't support the segment-level use case this branch targets |
| Client-side VAD | Pushes complexity to callers; inconsistent behavior across clients |

---

## 2. Independent Ray Serve Deployments per Stage

### Description

`VADDeployment`, `ASRDeployment`, `Pipeline`, and `IngressDeployment` are each separate `@serve.deployment` classes, bound together in `bind_app()`, rather than one monolithic service class.

### Pros

- **Independent scaling** — `VAD_NUM_REPLICAS` (default `4`) can be tuned separately from `ASR_NUM_REPLICAS` (default `1`) to match each stage's relative cost; VAD is typically cheaper per call, so more replicas can absorb concurrent requests without over-provisioning GPU for the heavier ASR model.
- **Independent GPU fractions** — `VAD_NUM_GPUS=0.1` and `ASR_NUM_GPUS=0.2` let both models share a single physical GPU via Ray's fractional GPU allocation, rather than each requiring a dedicated device.
- **Fault isolation** — a crash or restart in one deployment's replica doesn't necessarily take down the others.
- **Ingress stays CPU-only** — `INGRESS_NUM_GPUS=0` means the HTTP-handling layer never competes for GPU resources.

### Cons

- **More moving parts** — four deployment configs to reason about instead of one; `config.toml` has four sections (`system`, `asr`, `vad`, `ingress`) to keep in sync.
- **Extra inter-actor calls** — `Ingress → Pipeline → VAD` and `Pipeline → ASR` are all separate Ray remote calls, each with its own (small) serialization/scheduling overhead.

---

## 3. Segment-Level Fan-Out with Cross-Request ASR Batching

### Description

`Pipeline` does not send the whole utterance to ASR in one call. It fires one `ASRDeployment.batched_transcribe_tensors.remote()` per detected speech segment and awaits them together with `asyncio.gather`. Because `batched_transcribe_tensors` is decorated with `@serve.batch`, Ray Serve pools these calls — including segments originating from *different* concurrent HTTP requests — into a single GPU batch.

### Pros

- **Batching happens at the right granularity** — GPU utilization is driven by segment count across the whole service, not by how many segments happen to exist within one caller's audio file.
- **Transparent to the caller** — a single HTTP request still gets one combined response; the fan-out/batching is invisible externally.
- **Configurable throughput/latency trade-off** — `ASR_MAX_BATCH_SIZE` and `ASR_BATCH_WAIT_TIMEOUT_S` tune how aggressively segments are grouped before dispatch.

### Cons

- **More Ray remote calls per request** — an audio file with many speech segments generates many small ASR calls instead of one large one, increasing per-segment scheduling overhead.
- **No sort-by-length or mixed-batch splitting** — unlike batching strategies that group similarly-sized inputs to reduce padding, `process_batch_transcription()` here only partitions by timestamp requirement (with/without), so a batch can still mix short and long segments with attendant padding waste.
- **Waiting for the slowest segment** — `asyncio.gather` on segment-level ASR calls means the whole request completes only once every segment (some of which may land in different, staggered batches) has returned.

---

## 4. Timestamp Offset Adjustment

### Description

Because each speech segment is transcribed independently, the word/segment timestamps returned by NeMo are relative to the start of that segment's own (sliced) waveform. `Pipeline` adds each segment's `VADSegment.start` back onto every timestamp before returning the final response, and re-indexes segment `id`s sequentially across the merged result.

### Pros

- **Correct absolute timestamps** — callers receive timestamps relative to the original audio file, as they would expect from a single-pass transcription.
- **Simple, local fix** — a single addition per timestamp; no need to change how NeMo computes alignments.

### Cons

- **No cross-segment continuity check** — if VAD segments overlap or have small gaps, the adjustment doesn't attempt to reconcile timing between adjacent segments; each is corrected independently.
- **Precision compounding** — VAD timestamps are rounded to 3 decimals and ASR timestamps are separately rounded to 3 decimals; the sum can carry a small compounding rounding error (sub-millisecond in practice).

---

## 5. Ray Serve for Deployment Orchestration

### Description

The entire graph (`VADDeployment`, `ASRDeployment`, `Pipeline`, `IngressDeployment`) is served through **Ray Serve** rather than a bare FastAPI/Uvicorn process with in-process model calls.

### Pros

- **Native multi-deployment graphs** — Ray Serve is designed for exactly this kind of chained-deployment composition (`.bind()` one deployment into another's constructor).
- **Built-in autobatching per deployment** — `@serve.batch` on `ASRDeployment` requires no manual queue management.
- **Fractional and heterogeneous GPU allocation** — different deployments can request different GPU fractions on the same physical GPU.
- **Dashboard** — the Ray Serve dashboard (`:8265`) surfaces replica health, request throughput, and errors across all four deployments.

### Cons

- **Higher startup overhead** — Ray cluster init plus two model loads (VAD + ASR) is slower to become ready than a single-model service.
- **Operational complexity** — four deployments' worth of logs, replicas, and actor lifecycles to monitor instead of one.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Single FastAPI service calling VAD and ASR models in-process | No independent scaling/GPU-fraction control per stage; harder to batch ASR across concurrent requests |
| Separate microservices behind a message queue | Higher operational overhead for a single-container deployment target; Ray Serve already provides the needed request routing and batching |

---

## 6. Gated Model Requires Explicit `HF_TOKEN`

### Description

`pyannote/segmentation-3.0` is a gated Hugging Face model that requires an authenticated, license-accepted account. `VAD_DEVICE`/`VAD_MODEL_NAME` are configured in `config/config.toml`, but the `HF_TOKEN` itself is read from the environment (`app/core/config/vad.py: os.getenv("HF_TOKEN", "")`) rather than the TOML file.

### Pros

- **Secrets stay out of version control** — `config.toml` is checked in; `.env` is not, so the token never ends up in git history.
- **Consistent with other environment-specific values** — ports and the HF cache path are also environment-driven.

### Cons

- **Silent failure mode** — if `HF_TOKEN` is empty, `Model.from_pretrained()` fails at VAD deployment startup with an upstream Hugging Face authentication error rather than a clear, service-specific message.
- **Manual step required** — operators must accept the model's license on Hugging Face and generate a token before the service can start; this isn't automated or validated ahead of time.

---

## 7. Audio Format Validation via MIME Type

### Description

`is_audio_file()` uses `python-magic` to sniff the first 2048 bytes of the uploaded file and checks whether the MIME type starts with `audio/` or matches `application/ogg`, before the bytes are ever passed to VAD or ASR.

### Pros

- **Format-agnostic** — works for any container `torchaudio` can decode (MP3, WAV, FLAC, OGG, etc.) without an explicit extension allowlist.
- **Fast, early rejection** — only reads 2048 bytes; invalid uploads are rejected before the VAD call, saving a full pipeline round-trip.

### Cons

- **False positives/negatives for edge cases** — some valid containers may report non-standard MIME types (e.g., video files with an audio-only stream).
- **`python-magic` dependency** — requires the `libmagic` system library to be present in the Docker image.

---

## 8. OpenAI-Compatible API Surface

### Description

The single exposed endpoint mirrors OpenAI's `POST /v1/audio/transcriptions` signature: `file`, `model`, `timestamp_granularities[]` (`"word"` or `"segment"`), and `response_format` (currently only `verbose_json` is meaningfully supported).

### Pros

- **Drop-in compatibility** — any client already using the OpenAI Python SDK can point `base_url` at this service without code changes.
- **Familiar interface** — no bespoke API for integrators to learn.

### Cons

- **`model` is accepted but not enforced** — in this branch, `IngressDeployment.transcribe_audio()` does not validate the requested `model` field against the actually-loaded ASR model before dispatching to the pipeline (unlike a plain single-stage ASR service, where such a check is a natural fit at the ingress).
- **Subset of the OpenAI spec** — fields like `language` and `prompt` are not supported, and `response_format` values other than `verbose_json` are accepted but not meaningfully differentiated.

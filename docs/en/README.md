# VoicePlatform

A production-ready, two-stage VAD+ASR (Voice Activity Detection + Automatic Speech Recognition) pipeline with an OpenAI-compatible HTTP API, built on FastAPI and Ray Serve with Pyannote Audio VAD and NVIDIA NeMo Parakeet ASR models.

---

## Key Features

- **VAD + ASR composition pipeline** — a Pyannote Audio VAD model detects speech segments first; each segment is then transcribed independently by a NeMo Parakeet ASR model
- **OpenAI-compatible API** — drop-in replacement for `client.audio.transcriptions.create()`
- **Cross-request batching for both stages** — ASR segments from multiple concurrent HTTP requests are batched together via `@serve.batch`, independent of which original request they came from
- **Timestamp offset adjustment** — word- and segment-level timestamps returned by ASR are local to each speech segment; the pipeline shifts them by the VAD segment's start offset before returning
- **Independent, horizontally scalable deployments** — VAD, ASR, and the ingress/pipeline each run as separate Ray Serve deployments with their own replica count and GPU fraction
- **Resource efficiency** — only detected speech regions are sent to the ASR model, so silence and non-speech audio never reach the GPU-heavy ASR stage
- **Audio format validation** — MIME-type check via `python-magic` before any processing
- **Multi-language model support** — ships with `nvidia/parakeet-ctc-0.6b-vi` (Vietnamese); `nvidia/parakeet-tdt-0.6b-v3` is also a supported model identifier

---

## Architecture Overview

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
IngressDeployment  (FastAPI, Ray Serve ingress)
        │  read bytes, validate audio MIME
        │
        │  pipeline.remote(audio_bytes, timestamp_granularity)
        ▼
Pipeline  (Ray Serve deployment)
        │  load_audio_from_bytes() → full waveform tensor
        │
        ├──▶ VADDeployment.remote(audio_bytes)
        │        PyannoteVADDetector → List[VADSegment(start, end)]
        │
        │  slice waveform into one tensor per speech segment
        │
        ├──▶ ASRDeployment.batched_transcribe_tensors.remote(segment, granularity)  × N
        │        (@serve.batch — segments from *other* concurrent requests
        │         are folded into the same GPU batch)
        │        ParakeetRecognizer.transcribe()
        ▼
Merge results
        │  concatenate segment texts
        │  shift each word/segment timestamp by its VAD segment's start time
        ▼
TranscriptionResponse / WordResponse / SegmentResponse
```

---

## Documentation

- [Technical Overview](TECHNICAL_OVERVIEW.md) — full architecture, processing pipeline, configuration reference
- [Flow](FLOW.md) — step-by-step startup and per-request flow
- [Detailed Components](DETAILED_COMPONENTS.md) — component internals and API reference
- [Design Decisions](DESIGN_DECISIONS.md) — rationale and trade-offs for each major choice

---

## Quick Start

```bash
git clone https://github.com/nlp4everyone/VoicePlatform.git
cd VoicePlatform/
git fetch && git checkout ray/nvidia_asr_with_vad
cp .env.sample .env
```

Edit `.env` to set your Hugging Face token (required to download the Pyannote VAD model) and, optionally, ports:

```
HF_TOKEN=hf_xxx
RAY_FASTAPI_PORT=8005
RAY_DASHBOARD_PORT=8265
```

Edit `config/config.toml` to match your hardware:

```toml
[asr]
ASR_NUM_GPUS = 0.2
ASR_NUM_REPLICAS = 1
ASR_MAX_ONGOING_REQUESTS = 8
ASR_MAX_BATCH_SIZE = 8
ASR_BATCH_WAIT_TIMEOUT_S = 0.1

[vad]
VAD_MODEL_NAME = "pyannote/segmentation-3.0"
VAD_DEVICE = "auto"
VAD_NUM_GPUS = 0.1
VAD_NUM_REPLICAS = 4
VAD_MAX_ONGOING_REQUESTS = 8
```

```bash
bash run_service.sh
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8005/v1", api_key="token")

with open("resources/sample_vi.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="nvidia/parakeet-ctc-0.6b-vi",
        file=f,
        timestamp_granularities=["word"]
    )
print(result)
```

---

## HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio through the VAD+ASR pipeline — OpenAI-compatible |
| `GET` | `/docs` | FastAPI interactive documentation |
| `GET` | `:8265` | Ray Serve dashboard |

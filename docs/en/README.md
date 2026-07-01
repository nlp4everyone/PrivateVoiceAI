# VoicePlatform

A production-ready batch ASR (Automatic Speech Recognition) service with an OpenAI-compatible HTTP API, built on FastAPI and Ray Serve with NVIDIA NeMo Parakeet models.

---

## Key Features

- **OpenAI-compatible API** — drop-in replacement for `client.audio.transcriptions.create()`
- **Automatic request batching** — Ray Serve groups concurrent requests into GPU batches for throughput
- **Timestamp granularities** — word-level and segment-level timestamps on demand
- **Mixed-batch splitting** — timestamp and non-timestamp requests in the same batch are dispatched to separate GPU sub-calls, avoiding unnecessary logit transfers
- **Sort-by-length batching** — audio tensors are sorted by duration before GPU inference to minimize padding waste
- **AMP autocast** — mixed-precision inference (FP16/BF16) via `torch.cuda.amp.autocast`
- **Cached resampler** — `torchaudio.transforms.Resample` instances are cached by `(src_sr, tgt_sr)` pair via `lru_cache`
- **Audio format validation** — MIME-type check via `python-magic` before any processing
- **Multi-replica support** — scale horizontally by setting `NUM_REPLICAS` in config

---

## Architecture Overview

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
FastAPI  (ASRService — Ray Serve ingress)
        │  validate model name, read bytes, validate audio MIME
        │
        │  .batched_transcribe.remote(audio_bytes, granularity)
        ▼
Ray Serve batch queue  (MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S)
        │
        │  load_audio_from_bytes() × N  [parallel asyncio.to_thread]
        │  process_batch_transcription() [dedicated GPU ThreadPoolExecutor]
        ▼
ParakeetRecognizer
        │  sort by length → autocast → model.transcribe()
        ▼
NeMo ASRModel  (nvidia/parakeet-ctc-0.6b-vi  or  nvidia/parakeet-tdt-0.6b-v3)
        │
        ▼
TranscriptionResult  →  TranscriptionResponse / WordResponse / SegmentResponse
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
git fetch && git checkout ray/nvidia_asr
cp .env.sample .env
```

Edit `config/config.toml` to match your hardware:

```toml
[serving]
NUM_GPUS = 1
NUM_REPLICAS = 1
MAX_ONGOING_REQUESTS = 16
MAX_BATCH_SIZE = 8
BATCH_WAIT_TIMEOUT_S = 0.1

[asr]
ASR_MODEL_NAME = "nvidia/parakeet-ctc-0.6b-vi"
ASR_DEVICE = "auto"
```

```bash
bash run_service.sh
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

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
| `POST` | `/v1/audio/transcriptions` | Transcribe audio — OpenAI-compatible |
| `GET` | `/docs` | FastAPI interactive documentation |
| `GET` | `:8265` | Ray Serve dashboard |

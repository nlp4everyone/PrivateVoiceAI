# PrivateVoiceAI

A production-ready batch ASR service with an OpenAI-compatible HTTP API, built on FastAPI and Ray Serve with NVIDIA NeMo Parakeet models.

**Documentation:** [English](docs/en/README.md) | [Tiếng Việt](docs/vi/README_vi.md)

---

## Quick Start

```bash
git clone https://github.com/nlp4everyone/PrivateVoiceAI.git
cd PrivateVoiceAI/
git fetch && git checkout ray/nvidia_asr
cp .env.sample .env
```

Edit `config/config.toml`:

```toml
[serving]
NUM_GPUS = 1
NUM_REPLICAS = 1
MAX_ONGOING_REQUESTS = 16
MAX_BATCH_SIZE = 8

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

## Model Citation

This project uses the **NVIDIA Parakeet CTC 0.6B Vietnamese** model:
➡️ https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi

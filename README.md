# 🛸 Introduction:

A lightweight, production-ready ASR (Automatic Speech Recognition) framework built with FastAPI and Ray Serve for high-performance audio transcription.

<br />

# 🧠 Key Features
### 🎙️ Audio Transcription (FastAPI + Ray Serve)

High-performance audio transcription using Ray Serve for distributed inference.

Supports multi-format audio input (MP3, WAV, etc.) with automatic batching for improved throughput.

### 🧩 Local ASR Runtime

Fully offline, GPU-accelerated ASR inference using NVIDIA NeMo toolkit.

Supports Vietnamese language recognition with the NVIDIA Parakeet CTC model.

### 🔧 Configuration Management

TOML-based configuration system for system settings, serving parameters, and ASR model configuration.

<br />

### 🔗 Installing:

Clone this project:
```
git clone https://github.com/nlp4everyone/PrivateVoiceAI.git
```
Go inside project:
```
cd PrivateVoiceAI/
```
Fetch all branches:
```
git fetch
```
Checkout the branch:
```
git checkout ray/nvidia_asr
```

Create .env file from sample:
```
cp .env.sample .env
```

Configure ports in .env file (optional):
```
RAY_FASTAPI_PORT=8001
RAY_DASHBOARD_PORT=8265
```

Adjust GPU and serving configuration in `config/config.toml`:
```toml
[serving]
NUM_GPUS=1
NUM_REPLICAS=1
MAX_ONGOING_REQUESTS=16
MAX_BATCH_SIZE=8

[asr]
ASR_MODEL_NAME="nvidia/parakeet-ctc-0.6b-vi"
ASR_DEVICE="auto"
```

Run services with Docker Compose:
```
bash run_service.sh
```

Usage Example:
```python
from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8005/v1",
                api_key="token")

audio_file = open("resources/sample_vi.wav", "rb")
begin = time.perf_counter()
transcript = client.audio.transcriptions.create(
  model="nvidia/parakeet-ctc-0.6b-vi",
  file=audio_file,
  timestamp_granularities=["word"]
)
print(transcript)
print(f"Processing time: {time.perf_counter() - begin:.2f}s")
```

<br />

# 💴 Integrations:

- ⚙️ API Layer: FastAPI with OpenAI-compatible endpoints

- 💻 Serving Framework: RayServe with automatic batching

- 🧰 Runtime: Docker Compose with CUDA support

- 🤖 ASR Model: NVIDIA Parakeet CTC ([parakeet-ctc-0.6b-vi](https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi))

- 🔧 ASR Toolkit: NVIDIA NeMo Toolkit v2.7.2

- 🌐 Language Detection: pycld2 for automatic language detection

<br />

# 🧪 Testing:

### 🔹 Example Usage

Run the provided example to test transcription:
```bash
python examples/audio_transcription_example.py
```

### 🔹 API Endpoints

The service provides OpenAI-compatible endpoints:

- **POST /v1/audio/transcriptions** - Main transcription endpoint
- **GET /docs** - Interactive API documentation (FastAPI)
- **Dashboard** - Ray Serve dashboard at http://localhost:8265

### 🔹 Supported Features

- **Timestamp Granularities**: Word-level and segment-level timestamps
- **Language Detection**: Automatic language detection for transcribed text
- **Batch Processing**: Automatic request batching for improved throughput
- **Multiple Audio Formats**: MP3, WAV, and other supported audio formats

### 🔹 Configuration Options

Key configuration parameters in `config/config.toml`:

- `NUM_GPUS`: Number of GPUs per replica
- `NUM_REPLICAS`: Number of service replicas
- `MAX_BATCH_SIZE`: Maximum batch size for processing
- `BATCH_WAIT_TIMEOUT_S`: Batch wait timeout in seconds
- `ASR_MODEL_NAME`: HuggingFace model identifier
- `ASR_DEVICE`: Device selection (auto/cpu/cuda)

# 📋 To-Do / Roadmap

### 🤖 Support Model
- [x] Add multi-language model support (nvidia/parakeet-ctc-0.6b-vi,nvidia/parakeet-tdt-0.6b-v3)

### 🔧 Refactor Code
- [ ] Improve error handling and exception management
- [ ] Add comprehensive logging throughout the codebase

### ⚡ Optimization
- [ ] Optimize audio preprocessing pipeline
- [ ] Add GPU memory optimization and cleanup

# 📚 Model Citation

This project uses the **NVIDIA Parakeet CTC 0.6B Vietnamese model**:

➡️ **HuggingFace Model:** https://huggingface.co/nvidia/parakeet-ctc-0.6b-vi

If you use this model, please consider citing the original authors.



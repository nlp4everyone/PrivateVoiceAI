# 🛸 Introduction:

A lightweight, production-ready audio processing framework built with FastAPI and Ray Serve for high-performance voice activity detection.

<br />

# 🧠 Key Features
### 🎤 Voice Activity Detection (FastAPI + Ray Serve)

High-performance voice activity detection using Ray Serve for distributed inference.

Detects speech segments in audio files with precise timestamps using Pyannote Audio models.

### 🔧 Configuration Management

TOML-based configuration system for system settings, serving parameters, and VAD model configuration.

<br />

### 🔗 Installing:

Clone this project:
```
git clone https://github.com/nlp4everyone/VoicePlatform.git
```
Go inside project:
```
cd VoicePlatform/
```
Fetch all branches:
```
git fetch
```
Checkout the branch:
```
git checkout ray/vad
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
NUM_GPUS=0.25
NUM_REPLICAS=4
MAX_ONGOING_REQUESTS=10

[vad]
VAD_MODEL_NAME="pyannote/segmentation-3.0"
VAD_DEVICE="auto"
```

Run services with Docker Compose:
```
bash run_service.sh
```

### VAD Usage Example:
```python
import requests

url = "http://localhost:8005/v1/audio/activity_detections"

files = {
    "file": ("resources/sample_vi.wav", open("resources/sample_vi.wav", "rb"), "audio/wav")
}

data = {
    "model": "pyannote/segmentation-3.0"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

<br />

# 💴 Integrations:

- 💻 Serving Framework: RayServe with automatic batching

- 🧰 Runtime: Docker Compose with CUDA support

- 🎤 VAD Model: Pyannote Audio ([segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0))

<br />

# 🧪 Testing:

### 🔹 Example Usage

Run the provided example to test the service:
```bash
python examples/vad_example.py
```

### 🔹 API Endpoints

The service provides OpenAI-compatible endpoints:

- **POST /v1/audio/activity_detections** - VAD detection endpoint
- **GET /docs** - Interactive API documentation (FastAPI)
- **Dashboard** - Ray Serve dashboard at http://localhost:8265

### 🔹 Supported Features

- **Speech Segment Detection**: Precise timestamps for detected speech segments
- **Batch Processing**: Automatic request batching for improved throughput
- **Multiple Audio Formats**: MP3, WAV, and other supported audio formats

### 🔹 Configuration Options

Key configuration parameters in `config/config.toml`:

- `NUM_GPUS`: Number of GPUs per replica
- `NUM_REPLICAS`: Number of service replicas
- `MAX_ONGOING_REQUESTS`: Maximum concurrent requests
- `VAD_MODEL_NAME`: VAD model identifier (HuggingFace)
- `VAD_DEVICE`: Device selection for VAD (auto/cpu/cuda)

<br />

# 🧪 Experimental Results

Performance testing with `sample_vi.wav` (8s duration) using different batch sizes:

| Batch Size | Max Time | Avg Time |
|------------|----------|----------|
| 1          |          | 32.49ms  |
| 4          | 104.89ms | 69.65ms  |
| 8          | 220.43ms | 126.01ms |

<br />

# 📋 To-Do List

### 🔧 Core Features
- [x] Add support for multiple VAD models via Factory design

### ⚡ Performance & Scalability
- [ ] Optimize latencies/throughput for concurrent requests

### 🔐 Security & Privacy
- [x] Add input validation

# 📚 Model Citation

This project uses the **Pyannote Audio Segmentation 3.0 model**:

➡️ **HuggingFace Model:** https://huggingface.co/pyannote/segmentation-3.0

If you use this model, please consider citing the original authors.

<br />




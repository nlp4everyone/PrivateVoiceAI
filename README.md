# 🎤 VoicePlatform - Private Audio Transcription Service

A high-performance, privacy-focused audio transcription service powered by Qwen's Automatic Speech Recognition (ASR) model. This project provides a secure, local deployment solution for transcribing audio files without relying on cloud services, ensuring complete data privacy and control.

Built with vLLM for optimal performance and GPU acceleration, this service offers OpenAI-compatible API endpoints for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced ASR Capabilities
- High-accuracy audio transcription using state-of-the-art ASR models
- Support for multiple languages and audio formats
- OpenAI-compatible API for easy integration with existing tools
- Flexible input formats (file upload and base64 encoding)

### 🚀 Performance Optimizations
- GPU-accelerated inference with vLLM for real-time transcription
- Configurable GPU memory utilization and model parameters
- Efficient batch processing for multiple audio files
- Optimized for low-latency transcription workflows

### 🔒 Privacy & Security
- Complete local deployment - no data leaves your infrastructure
- No API keys or external service dependencies required
- Full control over model and data processing pipeline
- Ideal for sensitive audio content and compliance requirements

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints
- Comprehensive examples for different integration scenarios
- Docker-based deployment for consistent environments
- Detailed logging and monitoring capabilities

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (recommended for optimal performance)
   - 💾 **Storage**: SSD recommended for better model loading performance

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)
   - ⚡ **CUDA Toolkit 11.8+** (for GPU acceleration)

<br />

# 🚀 Quick Start

1. **Clone the repository**
   ```bash
   # Clone the repository
   git clone https://github.com/nlp4everyone/VoicePlatform.git
   # Navigate to project directory
   cd VoicePlatform
   ```

2. **Switch to the correct branch**
   ```bash
   # Fetch the latest changes and checkout the correct branch
   git fetch
   git checkout vllm/qwen3_asr
   ```

3. **Set up environment configuration**
   ```bash
   # Copy the sample environment file
   cp .env.sample .env
   # Edit the .env file to customize settings
   # nano .env  # or use your preferred text editor
   ```

4. **Build and start the service**
   ```bash
   # Build and start the ASR service (requires sudo for Docker)
   bash run_service.sh
   ```

5. **Verify the service is running**
   ```bash
   # Check if the service is responding
   curl http://localhost:8001/v1/models
   # View service logs
   sudo docker compose logs -f qwen-asr
   ```

6. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8001/v1 - OpenAI-compatible API
   - 📚 **API Documentation**: http://localhost:8001/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8001/health - Service status

<br />

# 📝 Examples

## 🎵 Audio File Transcription

Check out the `examples/audio_transcription_example.py` file to see how to:
- 🎯 **Transcribe** audio files using the OpenAI-compatible API
- ⏱️ **Measure** transcription performance and processing time
- 🌍 **Handle** different audio formats and languages

```bash
# Run the audio transcription example
python examples/audio_transcription_example.py
```

## 🔐 Base64 Audio Transcription

The `examples/base64_audio_example.py` demonstrates:
- 🔐 **Encoding** audio files in base64 format
- 📤 **Sending** audio data via chat completions API
- ⚙️ **Processing** transcription results programmatically

```bash
# Run the base64 audio example
python examples/base64_audio_example.py
```

## 🌊 Streaming Transcription

### 🔄 Asynchronous Streaming

The `examples/async_streaming_example.py` demonstrates:
- 🔄 **Real-time** transcription using async OpenAI client
- 🌊 **Streaming** responses for immediate feedback
- ⚙️ **Configurable** transcription parameters (temperature, seed, top_p)

```bash
# Run the async streaming example
python examples/async_streaming_example.py
```

### 🔄 Synchronous Streaming

The `examples/sync_streaming_example.py` shows:
- 🔄 **Streaming** transcription using raw HTTP requests
- 📡 **Server-Sent Events** (SSE) format handling
- 🔗 **Direct API** integration without OpenAI SDK

```bash
# Run the sync streaming example
python examples/sync_streaming_example.py
```

## 🎵 Sample Audio File

A sample audio file is included in `resources/sample_vi.mp3` for testing the transcription capabilities immediately after deployment.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service behavior using these environment variables in your `.env` file:

```bash
# 🎯 Model configuration
MODEL_NAME=Qwen/Qwen3-ASR-1.7B          # ASR model to use (configurable)
GPU_MEMORY_UTILIZATION=0.8              # GPU memory allocation (0.0-1.0)
MAX_MODEL_LEN=8192                      # Maximum model context length
MAX_NUM_SEQS=4                          # Maximum concurrent sequences

# 🌐 Network configuration
VLLM_HOST=0.0.0.0                       # Service host address
VLLM_PORT=8001                          # Service port

# 💾 Storage configuration
DOWNLOAD_DIR=/root/.cache/huggingface/hub  # Model cache directory
```

<br />

# 📋 To-Do List
- [x] 🚀 Basic ASR service deployment
- [x] 📝 Example implementations
- [x] 🌊 Streaming transcription capabilities (async and sync)

<br />

# 💻 Technology Stack:
- 🎯 **ASR Model**: Configurable (default: Qwen/Qwen3-ASR-1.7B)
- ⚡ **Inference Engine**: vLLM
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging
- 🛠️ **Client Libraries**: OpenAI Python SDK

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm) for the high-performance inference engine
- 🤖 [Qwen Team](https://github.com/QwenLM/Qwen) for the advanced ASR model


# VoicePlatform

Pipeline VAD+ASR (Voice Activity Detection + Nhận dạng giọng nói tự động) hai giai đoạn, sẵn sàng production, cung cấp HTTP API tương thích OpenAI, xây dựng trên FastAPI và Ray Serve với mô hình VAD Pyannote Audio và mô hình ASR NVIDIA NeMo Parakeet.

---

## Tính năng chính

- **Pipeline kết hợp VAD + ASR** — mô hình VAD Pyannote Audio phát hiện các đoạn có giọng nói trước; mỗi đoạn sau đó được phiên âm độc lập bởi mô hình ASR NeMo Parakeet
- **API tương thích OpenAI** — thay thế trực tiếp cho `client.audio.transcriptions.create()`
- **Gộp batch xuyên request cho cả hai giai đoạn** — các đoạn ASR đến từ nhiều request HTTP đồng thời được gộp chung batch qua `@serve.batch`, không phụ thuộc vào việc chúng thuộc request gốc nào
- **Điều chỉnh offset timestamp** — timestamp theo từ và theo đoạn do ASR trả về vốn tính theo thời gian cục bộ của từng đoạn giọng nói; pipeline dịch chuyển các timestamp này theo mốc bắt đầu (offset) của đoạn VAD trước khi trả về
- **Các deployment độc lập, scale ngang riêng biệt** — VAD, ASR và ingress/pipeline mỗi thành phần chạy như một Ray Serve deployment riêng, có số replica và tỉ lệ GPU cấu hình độc lập
- **Tiết kiệm tài nguyên** — chỉ vùng có giọng nói được phát hiện mới được gửi tới mô hình ASR, nên khoảng lặng và audio không phải giọng nói không bao giờ chạm tới giai đoạn ASR tốn GPU
- **Kiểm tra định dạng audio** — MIME-type check qua `python-magic` trước khi xử lý
- **Hỗ trợ đa mô hình ngôn ngữ** — đi kèm `nvidia/parakeet-ctc-0.6b-vi` (tiếng Việt); `nvidia/parakeet-tdt-0.6b-v3` cũng là một định danh model được hỗ trợ

---

## Tổng quan kiến trúc

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
IngressDeployment  (FastAPI, Ray Serve ingress)
        │  đọc bytes, kiểm tra MIME audio
        │
        │  pipeline.remote(audio_bytes, timestamp_granularity)
        ▼
Pipeline  (Ray Serve deployment)
        │  load_audio_from_bytes() → tensor waveform đầy đủ
        │
        ├──▶ VADDeployment.remote(audio_bytes)
        │        PyannoteVADDetector → List[VADSegment(start, end)]
        │
        │  cắt waveform thành từng tensor riêng cho mỗi đoạn giọng nói
        │
        ├──▶ ASRDeployment.batched_transcribe_tensors.remote(segment, granularity)  × N
        │        (@serve.batch — các đoạn từ những request đồng thời *khác*
        │         cũng được gộp vào cùng GPU batch)
        │        ParakeetRecognizer.transcribe()
        ▼
Ghép kết quả
        │  nối text của từng đoạn
        │  dịch mỗi timestamp từ/đoạn theo mốc start của đoạn VAD tương ứng
        ▼
TranscriptionResponse / WordResponse / SegmentResponse
```

---

## Tài liệu

- [Tổng quan kỹ thuật](TECHNICAL_OVERVIEW_vi.md) — kiến trúc đầy đủ, pipeline xử lý, tham chiếu cấu hình
- [Luồng xử lý](FLOW_vi.md) — từng bước khởi động và xử lý request
- [Chi tiết các component](DETAILED_COMPONENTS_vi.md) — nội bộ từng component và API reference
- [Quyết định thiết kế](DESIGN_DECISIONS_vi.md) — lý do và đánh đổi cho từng lựa chọn kiến trúc

---

## Bắt đầu nhanh

```bash
git clone https://github.com/nlp4everyone/VoicePlatform.git
cd VoicePlatform/
git fetch && git checkout ray/nvidia_asr_with_vad
cp .env.sample .env
```

Chỉnh `.env` để thiết lập Hugging Face token (bắt buộc để tải mô hình VAD Pyannote) và tùy chọn đổi port:

```
HF_TOKEN=hf_xxx
RAY_FASTAPI_PORT=8005
RAY_DASHBOARD_PORT=8265
```

Chỉnh `config/config.toml` phù hợp với phần cứng:

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

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Phiên âm audio qua pipeline VAD+ASR — tương thích OpenAI |
| `GET` | `/docs` | Tài liệu API tương tác (FastAPI) |
| `GET` | `:8265` | Ray Serve dashboard |

# VoicePlatform

Dịch vụ ASR (Nhận dạng giọng nói tự động) batch sẵn sàng production, cung cấp HTTP API tương thích OpenAI, xây dựng trên FastAPI và Ray Serve với các mô hình NVIDIA NeMo Parakeet.

---

## Tính năng chính

- **API tương thích OpenAI** — thay thế trực tiếp cho `client.audio.transcriptions.create()`
- **Tự động gộp batch** — Ray Serve tự động gom các request đồng thời thành GPU batch để tăng throughput
- **Timestamp linh hoạt** — hỗ trợ timestamp theo từng từ (word) và theo đoạn (segment)
- **Tách mixed-batch** — các request có/không có timestamp trong cùng batch được tách thành 2 GPU sub-call riêng, tránh transfer logit không cần thiết
- **Sắp xếp batch theo độ dài** — tensor audio được sắp xếp theo thời lượng trước khi đưa vào GPU để giảm padding lãng phí
- **AMP autocast** — inference mixed-precision (FP16/BF16) qua `torch.cuda.amp.autocast`
- **Cached resampler** — `torchaudio.transforms.Resample` được cache theo cặp `(src_sr, tgt_sr)` qua `lru_cache`
- **Kiểm tra định dạng audio** — MIME-type check qua `python-magic` trước khi xử lý
- **Hỗ trợ đa replica** — scale ngang bằng cách tăng `NUM_REPLICAS` trong config

---

## Tổng quan kiến trúc

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
FastAPI  (ASRService — Ray Serve ingress)
        │  kiểm tra model, đọc bytes, kiểm tra MIME audio
        │
        │  .batched_transcribe.remote(audio_bytes, granularity)
        ▼
Ray Serve batch queue  (MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S)
        │
        │  load_audio_from_bytes() × N  [song song asyncio.to_thread]
        │  process_batch_transcription() [GPU ThreadPoolExecutor riêng]
        ▼
ParakeetRecognizer
        │  sắp xếp theo độ dài → autocast → model.transcribe()
        ▼
NeMo ASRModel  (nvidia/parakeet-ctc-0.6b-vi  hoặc  nvidia/parakeet-tdt-0.6b-v3)
        │
        ▼
TranscriptionResult  →  TranscriptionResponse / WordResponse / SegmentResponse
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
git fetch && git checkout ray/nvidia_asr
cp .env.sample .env
```

Chỉnh `config/config.toml` phù hợp với phần cứng:

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

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Phiên âm audio — tương thích OpenAI |
| `GET` | `/docs` | Tài liệu API tương tác (FastAPI) |
| `GET` | `:8265` | Ray Serve dashboard |

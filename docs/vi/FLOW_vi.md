# Luồng xử lý chi tiết

## Trình tự khởi động

```text
app.py (serve run app.app:deployment)
    │
    ├── bind_app()
    │       │
    │       ├── VADDeployment.bind()
    │       │       đăng ký cấu hình deployment:
    │       │           ray_actor_options={"num_gpus": VAD_NUM_GPUS}
    │       │           num_replicas=VAD_NUM_REPLICAS
    │       │           max_ongoing_requests=VAD_MAX_ONGOING_REQUESTS
    │       │
    │       ├── ASRDeployment.bind()
    │       │       đăng ký cấu hình deployment:
    │       │           ray_actor_options={"num_gpus": ASR_NUM_GPUS}
    │       │           num_replicas=ASR_NUM_REPLICAS
    │       │           max_ongoing_requests=ASR_MAX_ONGOING_REQUESTS
    │       │
    │       ├── Pipeline.bind(vad, asr)
    │       │       actor thường giữ cả hai handle
    │       │
    │       └── IngressDeployment.bind(pipeline)
    │               đăng ký cấu hình deployment:
    │                   ray_actor_options={"num_gpus": INGRESS_NUM_GPUS}
    │                   num_replicas=INGRESS_NUM_REPLICAS
    │                   max_ongoing_requests=INGRESS_MAX_ONGOING_REQUESTS
    │
    ├── logging.basicConfig(level=INFO)
    │
    ├── logger.info("Ingress deployment bound with ... replica(s), ... GPU(s)")
    │
    └── serve.start(detached=False, http_options={"host": RAY_HOST, "port": RAY_PORT})
            │
            khởi động Ray Serve HTTP proxy; __init__ của mỗi deployment chạy
            một lần mỗi replica khi actor được tạo:
            │
            ├── VADDeployment.__init__()
            │       VADFactory.create(VAD_MODEL_NAME, VAD_DEVICE, token=HF_TOKEN)
            │           │
            │           ├── PyannoteVADDetector(model_name, token, device)
            │           │       ├── tắt TF32 (matmul + cudnn)
            │           │       ├── kiểm tra model_name → fallback về SUPPORTED_VAD_MODELS[0]
            │           │       ├── xác định device: "auto" → cuda nếu có, không thì cpu
            │           │       └── load_model()
            │           │               Model.from_pretrained(model_name, token=HF_TOKEN)
            │           │               model.to(device)
            │           │               VoiceActivityDetection(segmentation=model)
            │           │               pipeline.instantiate({min_duration_on: 0.0, min_duration_off: 0.0})
            │           │
            │           └── raise RuntimeError nếu factory trả về None
            │
            └── ASRDeployment.__init__()
                    RecognizerFactory.create(ASR_MODEL_NAME, ASR_DEVICE)
                        │
                        ├── ParakeetRecognizer(model_name, device)
                        │       ├── kiểm tra model_name → fallback về SUPPORTED_MODELS[0]
                        │       ├── xác định device: "auto" → cuda nếu có, không thì cpu
                        │       ├── nemo_asr.models.ASRModel.from_pretrained(model_name)
                        │       ├── model.cuda()   [nếu device == "cuda"]
                        │       └── model.eval()
                        │
                        └── raise RuntimeError nếu factory trả về None
```

---

## Luồng xử lý mỗi request

```text
① Client gửi POST /v1/audio/transcriptions
    multipart/form-data:
        file=<audio bytes>
        model=<tên model>
        timestamp_granularities[]=word|segment  (tùy chọn)
        response_format=verbose_json

② IngressDeployment.transcribe_audio()
    │
    ├── audio_bytes = await file.read()
    │
    ├── is_audio_file(audio_bytes)?
    │       KHÔNG → raise UnsupportedAudioFormatException(file_extension)
    │       CÓ  ↓
    │
    └── result = await self.pipeline.remote(audio_bytes, timestamp_granularity)
            → trả result trực tiếp về client

③ Pipeline.__call__(audio_bytes, timestamp_granularity)
    │
    ├── audio_tensor = await load_audio_from_bytes(audio_bytes)
    │       torchaudio.load(BytesIO) → tensor waveform mono, 16kHz
    │
    ├── vad_segments = await self.vad.remote(audio_bytes)
    │       → VADDeployment.__call__() → PyannoteVADDetector.detect(audio_bytes, precision=3)
    │           tự giải mã lại audio_bytes nội bộ
    │           chạy pipeline pyannote VoiceActivityDetection
    │           trả về List[VADSegment(start, end)]
    │
    ├── vad_segments rỗng?
    │       CÓ → trả về TranscriptionResponse / WordResponse / SegmentResponse rỗng
    │             (tùy theo timestamp_granularity) và dừng tại đây
    │
    ├── speech_segments = self._extract_audio_segments(audio_tensor, vad_segments)
    │       cho mỗi vad_seg:
    │           start_idx = max(0, int(vad_seg.start * 16000))
    │           end_idx   = min(len(audio_tensor), int(vad_seg.end * 16000))
    │           end_idx > start_idx?  → cắt audio_tensor[start_idx:end_idx]
    │                                    ngược lại bỏ qua kèm cảnh báo
    │
    ├── tasks = [
    │       self.asr.batched_transcribe_tensors.remote(seg, timestamp_granularity)
    │       for seg in speech_segments
    │   ]
    │   transcriptions = await asyncio.gather(*tasks)
    │       → mỗi call độc lập vào queue @serve.batch của ASRDeployment
    │         và có thể được gộp cùng các đoạn từ request đồng thời khác
    │
    ├── combined_text = " ".join(t.text for t in transcriptions)
    │
    ├── output_type = get_transcription_type(timestamp_granularity)
    ├── duration = round(estimate_audio_duration(audio_bytes), 3)
    │
    ├── output_type == Text?
    │       output_tokens = approximate_count_tokens(combined_text)
    │       → TranscriptionResponse(text=combined_text, usage=Usage(...))
    │
    ├── output_type == Word?
    │       cho mỗi cặp (transcription, vad_seg):
    │           cho mỗi word trong transcription.words:
    │               adjusted_word.start = word.start + vad_seg.start
    │               adjusted_word.end   = word.end + vad_seg.start
    │       lang = LanguageDetector.detect(combined_text, best_effort=True)
    │       → WordResponse(text, language, duration, usage=DurationUsage(seconds=ceil(duration)), words)
    │
    └── output_type == Segment?
            cho mỗi cặp (transcription, vad_seg):
                cho mỗi segment trong transcription.segments:
                    adjusted_segment.start = segment.start + vad_seg.start
                    adjusted_segment.end   = segment.end + vad_seg.start
                    adjusted_segment.id    = len(all_segments)   [đánh lại index]
            lang = LanguageDetector.detect(combined_text, best_effort=True)
            → SegmentResponse(text, language, duration, usage=DurationUsage(seconds=ceil(duration)), segments)

④ ASRDeployment.batched_transcribe_tensors(batch: List[Tensor], timestamp_granularities)
    [@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)]
    │
    Ray Serve gom các call tensor-đoạn từ bất kỳ số lượng lời gọi Pipeline
    đồng thời nào cho đến khi len(batch) == ASR_MAX_BATCH_SIZE hoặc
    thời gian chờ >= ASR_BATCH_WAIT_TIMEOUT_S, rồi gọi handler này một lần.
    │
    └── transcriptions = process_batch_transcription(
            asr_model=self._asr_model,
            audio_data=batch,               # tensor, không cần giải mã
            timestamp_granularities=timestamp_granularities
        )

⑤ process_batch_transcription(asr_model, audio_data, timestamp_granularities)
    │
    ├── ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)
    │
    ├── no_ts_indices không rỗng?
    │       → asr_model.transcribe(audio=[audio_data[i] for i in no_ts_indices],
    │                               enable_timestamps=False)
    │         ghi kết quả về đúng index gốc
    │
    └── ts_indices không rỗng?
            → asr_model.transcribe(audio=[audio_data[i] for i in ts_indices],
                                    enable_timestamps=True)
              ghi kết quả về đúng index gốc

⑥ ParakeetRecognizer.transcribe(audio, enable_timestamps, precision=3)
    │
    ├── chuẩn hóa input: item đơn → [item]
    │
    ├── results = self.model.transcribe(audio, timestamps=enable_timestamps)
    │       (không sắp xếp theo độ dài, không autocast — gọi trực tiếp)
    │
    ├── enable_timestamps=False?
    │       return [TranscriptionResult(text=result.text) for result in results]
    │
    └── enable_timestamps=True?
            cho mỗi result:
                word timestamps    → [TranscribedWord(start, end, word)]      (làm tròn theo precision)
                segment timestamps → [TranscribedSegment(start, end, text)]   (làm tròn theo precision)
            return [TranscriptionResult(text, segments, words), ...]
```

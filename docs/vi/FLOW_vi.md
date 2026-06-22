# Luồng xử lý chi tiết

## Trình tự khởi động

```text
app.py (serve run app.app:deployment)
    │
    ├── logging.basicConfig(level=WARNING)
    │   suppress: nemo / nemo_logger / lightning / pytorch_lightning /
    │             filelock / datasets / huggingface_hub  → ERROR
    │
    ├── ray.init(logging_level=WARNING)
    │       khởi động Ray cluster cục bộ (hoặc kết nối cluster đã có)
    │
    ├── ASRService.bind()
    │       đăng ký cấu hình deployment:
    │           num_replicas=NUM_REPLICAS
    │           num_gpus=NUM_GPUS  mỗi replica
    │           max_ongoing_requests=MAX_ONGOING_REQUESTS
    │
    ├── serve.start(host=RAY_HOST, port=RAY_PORT)
    │       khởi động Ray Serve HTTP proxy
    │
    └── ASRService.__init__()  [gọi một lần mỗi replica]
            │
            ├── logging.getLogger("ray.serve").setLevel(INFO)
            │       áp dụng lại INFO sau khi Ray Serve reset level
            │
            ├── RecognizerFactory.create(ASR_MODEL_NAME, ASR_DEVICE)
            │       │
            │       ├── logger.info("Loading ASR model ...")
            │       ├── ParakeetRecognizer(model_name, device)
            │       │       │
            │       │       ├── kiểm tra model_name → fallback về SUPPORTED_MODELS[0] nếu không hợp lệ
            │       │       ├── xác định device: "auto" → cuda nếu có, không thì cpu
            │       │       ├── nemo_asr.models.ASRModel.from_pretrained(model_name)
            │       │       │       tải từ cache HF_HOME hoặc download
            │       │       ├── model.cuda()   [nếu device == "cuda"]
            │       │       └── model.eval()
            │       │
            │       └── logger.info("ASR model loaded on CUDA/CPU")
            │
            ├── logger.info("ASRService ready | replicas=N max_ongoing_requests=N max_batch_size=N")
            │
            ├── serve.get_deployment_handle(DEPLOYMENT_NAME)   ← handle được cache
            └── ThreadPoolExecutor(max_workers=1)               ← luồng GPU riêng
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

② ASRService.transcribe_audio()
    │
    ├── model != asr_model.model_name?
    │       CÓ → raise TranscriptedModelNotFoundException(model)
    │
    ├── audio_bytes = await file.read()
    │
    ├── is_audio_file(audio_bytes)?
    │       KHÔNG → raise UnsupportedAudioFormatException(file_extension)
    │       CÓ  ↓
    │
    └── self._handle.batched_transcribe.remote(audio_bytes, timestamp_granularity)
            → chờ TranscriptionResult (block cho đến khi batch hoàn thành)

③ Ray Serve gom batch
    Gom các call .remote() đồng thời cho đến khi:
        len(batch) == MAX_BATCH_SIZE  HOẶC  thời gian chờ >= BATCH_WAIT_TIMEOUT_S
    sau đó gọi batched_transcribe(batch, timestamp_granularities)

④ ASRService.batched_transcribe(batch: List[bytes], timestamp_granularities: List[str|None])
    │
    ├── asyncio.gather(
    │       asyncio.to_thread(load_audio_from_bytes, audio_bytes)
    │       cho mỗi item trong batch
    │   )
    │   [giải mã CPU song song]
    │   load_audio_from_bytes(audio_bytes, target_sr=16000):
    │       torchaudio.load(BytesIO) → waveform, sr
    │       mono: waveform[0] nếu 1 channel, waveform.mean(dim=0) nếu nhiều channel
    │       sr != 16000? → _get_resampler(sr, 16000)(waveform)  [lru_cache]
    │       trả về (waveform: Tensor[T], duration: float)
    │
    ├── batch = None   ← giải phóng raw bytes ngay
    │
    ├── audio_tensors = [t for t, _ in decoded]
    ├── durations     = [d for _, d in decoded]
    ├── decoded = None
    │
    └── asyncio.get_event_loop().run_in_executor(
            self._gpu_executor,           ← executor 1 luồng (mỗi replica một cái)
            process_batch_transcription(
                asr_model=self._asr_model,
                audio_data=audio_tensors,
                timestamp_granularities=timestamp_granularities,
                split_mixed_batch=SPLIT_MIXED_BATCH
            )
        )

⑤ process_batch_transcription(asr_model, audio_data, timestamp_granularities, split_mixed_batch)
    │
    ├── ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)
    │
    ├── Toàn bộ no-ts (ts_indices rỗng)?
    │       → asr_model.transcribe(audio_data, enable_timestamps=False)   [1 GPU call]
    │
    ├── Toàn bộ ts (no_ts_indices rỗng)?
    │       → asr_model.transcribe(audio_data, enable_timestamps=True)    [1 GPU call]
    │
    └── Hỗn hợp:
            split_mixed_batch=True?
                ts_audio    = [audio_data[i] for i in ts_indices]
                no_ts_audio = [audio_data[i] for i in no_ts_indices]
                asr_model.transcribe(ts_audio, enable_timestamps=True)     [GPU call 1]
                asr_model.transcribe(no_ts_audio, enable_timestamps=False) [GPU call 2]
                ghép kết quả về đúng vị trí ban đầu
            split_mixed_batch=False?
                asr_model.transcribe(audio_data, enable_timestamps=True)   [1 GPU call]
                xóa .words / .segments khỏi item không cần timestamp

⑥ ParakeetRecognizer.transcribe(audio, enable_timestamps, precision=3)
    │
    ├── chuẩn hóa input: item đơn → [item]
    │
    ├── len(audio) > 1 VÀ isinstance(audio[0], Tensor)?
    │       sắp xếp theo độ dài tensor tăng dần
    │       order = argsort(len)
    │
    ├── with torch.cuda.amp.autocast():
    │       sorted_results = model.transcribe(audio, timestamps=enable_timestamps)
    │
    ├── khôi phục thứ tự gốc:
    │       results[orig_idx] = sorted_results[rank]
    │
    ├── enable_timestamps=False?
    │       return [TranscriptionResult(text=result.text) for result in results]
    │
    └── enable_timestamps=True?
            cho mỗi result:
                word timestamps  → [TranscribedWord(start, end, word)]
                segment timestamps → [TranscribedSegment(start, end, text)]
            return [TranscriptionResult(text, segments, words)]

⑦ ASRService.batched_transcribe  (tiếp)
    │
    ├── gắn duration: result.duration = duration  cho mỗi result
    └── return transcriptions   [danh sách khớp thứ tự với batch]

⑧ ASRService.transcribe_audio  (định dạng response)
    │
    ├── output_type = get_transcription_type(timestamp_granularity)
    │
    ├── TranscriptionType.Text
    │       → TranscriptionResponse(text, usage)
    │           usage.output_tokens = approximate_count_tokens(text)
    │
    ├── TranscriptionType.Word
    │       lang = LanguageDetector.detect(text)
    │       → WordResponse(text, language, duration, usage, words)
    │
    └── TranscriptionType.Segment
            lang = LanguageDetector.detect(text)
            segments với id, start, end, text
            → SegmentResponse(text, language, duration, usage, segments)
```

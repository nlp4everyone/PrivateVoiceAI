# Detailed Flow

## Startup Sequence

```text
app.py (serve run app.app:deployment)
    │
    ├── logging.basicConfig(level=WARNING)
    │   suppress: nemo / nemo_logger / lightning / pytorch_lightning /
    │             filelock / datasets / huggingface_hub  → ERROR
    │
    ├── ray.init(logging_level=WARNING)
    │       starts local Ray cluster (or connects to existing)
    │
    ├── ASRService.bind()
    │       registers deployment config:
    │           num_replicas=NUM_REPLICAS
    │           num_gpus=NUM_GPUS  per replica
    │           max_ongoing_requests=MAX_ONGOING_REQUESTS
    │
    ├── serve.start(host=RAY_HOST, port=RAY_PORT)
    │       starts Ray Serve HTTP proxy
    │
    └── ASRService.__init__()  [called once per replica]
            │
            ├── logging.getLogger("ray.serve").setLevel(INFO)
            │       re-applies INFO after Ray Serve resets it
            │
            ├── RecognizerFactory.create(ASR_MODEL_NAME, ASR_DEVICE)
            │       │
            │       ├── logger.info("Loading ASR model ...")
            │       ├── ParakeetRecognizer(model_name, device)
            │       │       │
            │       │       ├── validate model_name → fallback to SUPPORTED_MODELS[0] if unknown
            │       │       ├── resolve device: "auto" → cuda if available, else cpu
            │       │       ├── nemo_asr.models.ASRModel.from_pretrained(model_name)
            │       │       │       downloads / loads from HF_HOME cache
            │       │       ├── model.cuda()   [if device == "cuda"]
            │       │       └── model.eval()
            │       │
            │       └── logger.info("ASR model loaded on CUDA/CPU")
            │
            ├── logger.info("ASRService ready | replicas=N max_ongoing_requests=N max_batch_size=N")
            │
            ├── serve.get_deployment_handle(DEPLOYMENT_NAME)   ← cached handle
            └── ThreadPoolExecutor(max_workers=1)               ← dedicated GPU thread
```

---

## Per-Request Flow

```text
① Client sends POST /v1/audio/transcriptions
    multipart/form-data:
        file=<audio bytes>
        model=<model name>
        timestamp_granularities[]=word|segment  (optional)
        response_format=verbose_json

② ASRService.transcribe_audio()
    │
    ├── model != asr_model.model_name?
    │       YES → raise TranscriptedModelNotFoundException(model)
    │
    ├── audio_bytes = await file.read()
    │
    ├── is_audio_file(audio_bytes)?
    │       NO  → raise UnsupportedAudioFormatException(file_extension)
    │       YES ↓
    │
    └── self._handle.batched_transcribe.remote(audio_bytes, timestamp_granularity)
            → awaits TranscriptionResult (blocks until batch completes)

③ Ray Serve batch accumulation
    Collects concurrent .remote() calls until:
        len(batch) == MAX_BATCH_SIZE  OR  wait >= BATCH_WAIT_TIMEOUT_S
    then calls batched_transcribe(batch, timestamp_granularities)

④ ASRService.batched_transcribe(batch: List[bytes], timestamp_granularities: List[str|None])
    │
    ├── asyncio.gather(
    │       asyncio.to_thread(load_audio_from_bytes, audio_bytes)
    │       for each item in batch
    │   )
    │   [parallel CPU decode]
    │   load_audio_from_bytes(audio_bytes, target_sr=16000):
    │       torchaudio.load(BytesIO) → waveform, sr
    │       mono: waveform[0] if channels==1 else waveform.mean(dim=0)
    │       sr != 16000? → _get_resampler(sr, 16000)(waveform)  [lru_cache]
    │       returns (waveform: Tensor[T], duration: float)
    │
    ├── batch = None   ← free raw bytes immediately
    │
    ├── audio_tensors = [t for t, _ in decoded]
    ├── durations     = [d for _, d in decoded]
    ├── decoded = None  ← free decoded tuples
    │
    └── asyncio.get_event_loop().run_in_executor(
            self._gpu_executor,           ← single-thread executor (one per replica)
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
    ├── Pure no-ts (ts_indices empty)?
    │       → asr_model.transcribe(audio_data, enable_timestamps=False)   [1 GPU call]
    │
    ├── Pure ts (no_ts_indices empty)?
    │       → asr_model.transcribe(audio_data, enable_timestamps=True)    [1 GPU call]
    │
    └── Mixed batch:
            split_mixed_batch=True?
                ts_audio   = [audio_data[i] for i in ts_indices]
                no_ts_audio = [audio_data[i] for i in no_ts_indices]
                asr_model.transcribe(ts_audio, enable_timestamps=True)    [GPU call 1]
                asr_model.transcribe(no_ts_audio, enable_timestamps=False) [GPU call 2]
                merge results back to original indices
            split_mixed_batch=False?
                asr_model.transcribe(audio_data, enable_timestamps=True)  [1 GPU call]
                strip .words / .segments from no-ts items afterwards

⑥ ParakeetRecognizer.transcribe(audio, enable_timestamps, precision=3)
    │
    ├── normalize input: single item → [item]
    │
    ├── len(audio) > 1 AND isinstance(audio[0], Tensor)?
    │       sort by tensor length ascending
    │       order = argsort(len)
    │
    ├── with torch.cuda.amp.autocast():
    │       sorted_results = model.transcribe(audio, timestamps=enable_timestamps)
    │
    ├── restore original order:
    │       results[orig_idx] = sorted_results[rank]
    │
    ├── enable_timestamps=False?
    │       return [TranscriptionResult(text=result.text) for result in results]
    │
    └── enable_timestamps=True?
            for each result:
                word timestamps  → [TranscribedWord(start, end, word)]
                segment timestamps → [TranscribedSegment(start, end, text)]
            return [TranscriptionResult(text, segments, words)]

⑦ ASRService.batched_transcribe  (continued)
    │
    ├── attach duration: result.duration = duration  for each result
    └── return transcriptions   [list aligned with batch order]

⑧ ASRService.transcribe_audio  (response formatting)
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
            segments with id, start, end, text
            → SegmentResponse(text, language, duration, usage, segments)
```

# Detailed Flow

## Startup Sequence

```text
app.py (serve run app.app:deployment)
    │
    ├── bind_app()
    │       │
    │       ├── VADDeployment.bind()
    │       │       registers deployment config:
    │       │           ray_actor_options={"num_gpus": VAD_NUM_GPUS}
    │       │           num_replicas=VAD_NUM_REPLICAS
    │       │           max_ongoing_requests=VAD_MAX_ONGOING_REQUESTS
    │       │
    │       ├── ASRDeployment.bind()
    │       │       registers deployment config:
    │       │           ray_actor_options={"num_gpus": ASR_NUM_GPUS}
    │       │           num_replicas=ASR_NUM_REPLICAS
    │       │           max_ongoing_requests=ASR_MAX_ONGOING_REQUESTS
    │       │
    │       ├── Pipeline.bind(vad, asr)
    │       │       plain deployment holding both handles
    │       │
    │       └── IngressDeployment.bind(pipeline)
    │               registers deployment config:
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
            starts Ray Serve HTTP proxy; each deployment's __init__ runs
            once per replica as the actor is created:
            │
            ├── VADDeployment.__init__()
            │       VADFactory.create(VAD_MODEL_NAME, VAD_DEVICE, token=HF_TOKEN)
            │           │
            │           ├── PyannoteVADDetector(model_name, token, device)
            │           │       ├── disable TF32 (matmul + cudnn)
            │           │       ├── validate model_name → fallback to SUPPORTED_VAD_MODELS[0]
            │           │       ├── resolve device: "auto" → cuda if available, else cpu
            │           │       └── load_model()
            │           │               Model.from_pretrained(model_name, token=HF_TOKEN)
            │           │               model.to(device)
            │           │               VoiceActivityDetection(segmentation=model)
            │           │               pipeline.instantiate({min_duration_on: 0.0, min_duration_off: 0.0})
            │           │
            │           └── raise RuntimeError if factory returns None
            │
            └── ASRDeployment.__init__()
                    RecognizerFactory.create(ASR_MODEL_NAME, ASR_DEVICE)
                        │
                        ├── ParakeetRecognizer(model_name, device)
                        │       ├── validate model_name → fallback to SUPPORTED_MODELS[0]
                        │       ├── resolve device: "auto" → cuda if available, else cpu
                        │       ├── nemo_asr.models.ASRModel.from_pretrained(model_name)
                        │       ├── model.cuda()   [if device == "cuda"]
                        │       └── model.eval()
                        │
                        └── raise RuntimeError if factory returns None
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

② IngressDeployment.transcribe_audio()
    │
    ├── audio_bytes = await file.read()
    │
    ├── is_audio_file(audio_bytes)?
    │       NO  → raise UnsupportedAudioFormatException(file_extension)
    │       YES ↓
    │
    └── result = await self.pipeline.remote(audio_bytes, timestamp_granularity)
            → return result directly to the client

③ Pipeline.__call__(audio_bytes, timestamp_granularity)
    │
    ├── audio_tensor = await load_audio_from_bytes(audio_bytes)
    │       torchaudio.load(BytesIO) → mono, 16kHz waveform tensor
    │
    ├── vad_segments = await self.vad.remote(audio_bytes)
    │       → VADDeployment.__call__() → PyannoteVADDetector.detect(audio_bytes, precision=3)
    │           re-decodes audio_bytes internally
    │           runs pyannote VoiceActivityDetection pipeline
    │           returns List[VADSegment(start, end)]
    │
    ├── vad_segments empty?
    │       YES → return empty TranscriptionResponse / WordResponse / SegmentResponse
    │              (based on timestamp_granularity) and stop here
    │
    ├── speech_segments = self._extract_audio_segments(audio_tensor, vad_segments)
    │       for each vad_seg:
    │           start_idx = max(0, int(vad_seg.start * 16000))
    │           end_idx   = min(len(audio_tensor), int(vad_seg.end * 16000))
    │           end_idx > start_idx?  → slice audio_tensor[start_idx:end_idx]
    │                                    else skip with warning
    │
    ├── tasks = [
    │       self.asr.batched_transcribe_tensors.remote(seg, timestamp_granularity)
    │       for seg in speech_segments
    │   ]
    │   transcriptions = await asyncio.gather(*tasks)
    │       → each call independently enters ASRDeployment's @serve.batch queue
    │         and may be grouped with segments from other concurrent requests
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
    │       for each (transcription, vad_seg) pair:
    │           for each word in transcription.words:
    │               adjusted_word.start = word.start + vad_seg.start
    │               adjusted_word.end   = word.end + vad_seg.start
    │       lang = LanguageDetector.detect(combined_text, best_effort=True)
    │       → WordResponse(text, language, duration, usage=DurationUsage(seconds=ceil(duration)), words)
    │
    └── output_type == Segment?
            for each (transcription, vad_seg) pair:
                for each segment in transcription.segments:
                    adjusted_segment.start = segment.start + vad_seg.start
                    adjusted_segment.end   = segment.end + vad_seg.start
                    adjusted_segment.id    = len(all_segments)   [re-indexed]
            lang = LanguageDetector.detect(combined_text, best_effort=True)
            → SegmentResponse(text, language, duration, usage=DurationUsage(seconds=ceil(duration)), segments)

④ ASRDeployment.batched_transcribe_tensors(batch: List[Tensor], timestamp_granularities)
    [@serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE, batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)]
    │
    Ray Serve accumulates segment-tensor calls from any number of concurrent
    Pipeline invocations until len(batch) == ASR_MAX_BATCH_SIZE or
    wait >= ASR_BATCH_WAIT_TIMEOUT_S, then calls this handler once.
    │
    └── transcriptions = process_batch_transcription(
            asr_model=self._asr_model,
            audio_data=batch,               # tensors, no decode needed
            timestamp_granularities=timestamp_granularities
        )

⑤ process_batch_transcription(asr_model, audio_data, timestamp_granularities)
    │
    ├── ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)
    │
    ├── no_ts_indices non-empty?
    │       → asr_model.transcribe(audio=[audio_data[i] for i in no_ts_indices],
    │                               enable_timestamps=False)
    │         write results back at their global indices
    │
    └── ts_indices non-empty?
            → asr_model.transcribe(audio=[audio_data[i] for i in ts_indices],
                                    enable_timestamps=True)
              write results back at their global indices

⑥ ParakeetRecognizer.transcribe(audio, enable_timestamps, precision=3)
    │
    ├── normalize input: single item → [item]
    │
    ├── results = self.model.transcribe(audio, timestamps=enable_timestamps)
    │       (no length sorting, no autocast — direct call)
    │
    ├── enable_timestamps=False?
    │       return [TranscriptionResult(text=result.text) for result in results]
    │
    └── enable_timestamps=True?
            for each result:
                word timestamps    → [TranscribedWord(start, end, word)]      (rounded to precision)
                segment timestamps → [TranscribedSegment(start, end, text)]   (rounded to precision)
            return [TranscriptionResult(text, segments, words), ...]
```

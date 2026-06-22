# Configuration Reference

Config is loaded in priority order (highest → lowest):

1. Environment variables (Docker `-e` flags, CI)
2. `.env` file (local dev, not version-controlled) — environment-specific: URLs, paths, ports, concurrency limits
3. `config/settings.yaml` — stable algorithm params: inference intervals, VAD thresholds, stabilizer settings (version-controlled)
4. Field defaults in `app/core/config.py`

Override `SETTINGS_YAML` env var to point to a different YAML file.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `AUDIO_PACKET_MS` | 20 | Expected client packet size |
| `RING_BUFFER_SECONDS` | 12 | Max audio retained per session (pre-allocated np.int16 ring buffer) |
| `INFERENCE_INTERVAL_MS` | 600 | Fixed inference interval (ms) — chunker and fallback when `ADAPTIVE_INTERVAL_ENABLED=false` |
| `ADAPTIVE_INTERVAL_ENABLED` | `true` | Dynamically switch pacing between `ONSET_INTERVAL_MS` and `STABLE_INTERVAL_MS` |
| `ONSET_INTERVAL_MS` | 400 | Adaptive interval (onset): pacing right after speech begins — favors fast partials |
| `STABLE_INTERVAL_MS` | 1200 | Adaptive interval (stable): pacing when transcript stops changing — reduces redundant ASR calls |
| `RMS_SILENCE_THRESHOLD` | 300 | int16 RMS energy gate — skips VAD+ASR on silent windows when session is not mid-utterance; frees VAD pool for active sessions |
| `INFERENCE_WINDOW_SECONDS` | 6 | Audio window fed to STT |
| `SILENCE_THRESHOLD_MS` | 800 | Silence before utterance finalize |
| `TRAILING_SILENCE_MS` | 1000 | Trailing silence in the inference window that overrides `is_speech=True` — prevents stale VAD detections; reduces ASR calls by ~50% at utterance end |
| `SPEECH_PADDING_MS` | 200 | Context padding around speech region before ASR |
| `MIN_TRIMMED_AUDIO_MS` | 500 | Trimmed audio shorter than this (ms) skips ASR entirely — avoids sending near-silent segments |
| `INTRA_SILENCE_COMMIT_ENABLED` | `true` | Commit partial as final on mid-utterance pauses |
| `INTRA_SILENCE_MS` | 300 | Pause duration to trigger intra-utterance commit; must be < `SILENCE_THRESHOLD_MS` |
| `FINALIZE_RIGHT_PADDING_ENABLED` | `true` | Run a dedicated final ASR pass with precise speech boundaries on utterance end |
| `FINALIZE_RIGHT_PADDING_MS` | 200 | Right padding after `last_speech_time` in the final ASR window; keep ≤ `SPEECH_PADDING_MS` |
| `VAD_THRESHOLD` | 0.6 | Silero speech probability cutoff (`ema_smoothed` / `consecutive_frames`) |
| `VAD_ONSET_THRESHOLD` | 0.65 | Prob to **enter** speaking state (`state_machine` strategy) |
| `VAD_OFFSET_THRESHOLD` | 0.40 | Prob to **exit** speaking state — hysteresis band = [0.40, 0.65] |
| `VAD_SAMPLE_RATE` | 16000 | VAD expected sample rate |
| `VAD_WINDOW_SIZE_SAMPLES` | 512 | Frame size for VAD scoring (32ms at 16kHz) |
| `VAD_TRIGGER_STRATEGY` | `state_machine` | Active VAD strategy (`consecutive_frames` \| `ema_smoothed` \| `state_machine`) |
| `VAD_POOL_SIZE` | 8 | Number of parallel VAD instances in the async pool |
| `VAD_MODEL_PATH` | `/app/models/silero_vad.onnx` | Path to the Silero VAD ONNX model |
| `VAD_USE_INT8` | `false` | Quantize FP32 → INT8 on first startup (`_int8.onnx` cached on disk) |
| `STABILIZER_STRATEGY` | `frozen_prefix` | Rollback suppression strategy (`frozen_prefix` \| `hard_length` \| `edit_distance` \| `n_consecutive` \| `hard_then_frozen`) |
| `STABILIZER_MODE` | `word_level` | LCP granularity (`word_level` \| `character_level`) |
| `STABILIZER_FREEZE_THRESHOLD` | 3 | Consecutive agreements before freezing a prefix (`frozen_prefix`, `hard_then_frozen`) |
| `STABILIZER_MAX_EDIT_DISTANCE` | 2 | Max word edits allowed vs last output (`edit_distance`) |
| `STABILIZER_N_CONSECUTIVE` | 3 | Frames required to confirm a rollback (`n_consecutive`) |
| `STT_DEVICE` | `cuda` | Inference device — `cuda` \| `cpu` (set in `.env`) |
| `STT_BATCH_SIZE` | 1 | Batch size reserved for future local model use |
| `NEMO_API_URL` | `http://localhost:8005/v1/audio/transcriptions` | NeMo server endpoint (set in `.env`) |
| `NEMO_MODEL` | `nvidia/parakeet-ctc-0.6b-vi` | Model identifier |
| `ASR_SEMAPHORE_LIMIT` | 8 | Max concurrent NeMo HTTP requests across all sessions (set in `.env`) |
| `INFERENCE_QUEUE_MAXSIZE` | 3 | Per-session queue depth; excess windows are dropped |
| `ASR_CONNECT_TIMEOUT` | 2.0 | Seconds to establish TCP connection to NeMo |
| `ASR_REQUEST_TIMEOUT` | 10.0 | Seconds for full NeMo request (connect + transfer + response) |
| `WS_MAX_CONNECTIONS` | 200 | Hard cap on concurrent WebSocket sessions; excess closed with code 1013 |
| `WS_MAX_QUEUE_SIZE` | 100 | Per-connection send queue depth |
| `WS_PING_INTERVAL` | 20 | Keepalive ping interval (s) |
| `WS_PING_TIMEOUT` | 20 | Ping response timeout (s) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | 8000 | Server port |
| `WORKERS` | 1 | Uvicorn worker count |

All values are overridable via environment variables or `.env`.

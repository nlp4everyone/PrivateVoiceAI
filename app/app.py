# FastAPI components
from fastapi import FastAPI, UploadFile, File, Form

# Transcription components
from .schema.transcription.response import (
    TranscriptionResponse,
    WordResponse,
    SegmentResponse,
    TranscriptionResult
)
from .schema.transcription.base import AdvancedTranscribedSegment
from .schema.transcription.type import TranscriptionType
from .schema.transcription.base.usage import (
    Usage,
    InputTokenDetails,
    DurationUsage
)

# Ray Component
from ray import serve

# Typing
from typing import List, Optional, Union, Any

# Util
from .utils.audio import (
    save_temp_audio,
    clean_up_temp_audio,
    estimate_audio_duration
)
from .utils.transcription.helper import (
    get_timestamp_indices,
    get_transcription_type
)
from .utils.token_counter import approximate_count_tokens
from .utils.language_detect import LanguageDetector

# ASR Model
from .services.asr import RecognizerFactory

# Dependencies
from pathlib import Path
import os, logging, math, asyncio
from .exceptions.transcription import TranscriptedModelNotFoundException
from .exceptions.handlers import common_exception_handler

# Config
from .core.config.serving import *
from .core.config.system import *
from .core.config.asr import *

# Configure logger for Ray Serve
logger = logging.getLogger("ray.serve")

# OpenAPI tags for documentation
openapi_tags = [
    {
        "name": "Audio",
        "description": "Endpoints for interacting with ASR models"
    }
]

# Create FastAPI application with custom tags
app = FastAPI(openapi_tags=openapi_tags)

# Register exception handler for model not found errors
app.add_exception_handler(
    TranscriptedModelNotFoundException,
    common_exception_handler
)

@serve.deployment(
    ray_actor_options={"num_gpus": NUM_GPUS},
    num_replicas=NUM_REPLICAS,
    max_ongoing_requests=MAX_ONGOING_REQUESTS
)
@serve.ingress(app)
class ASRService:
    """
    Ray Serve deployment for ASR (Automatic Speech Recognition) service.
    
    This class handles audio transcription requests using configurable ASR models.
    It supports batch processing for improved throughput and provides different
    transcription formats based on client requirements.
    """
    
    def __init__(self):
        """
        Initialize the ASR service.
        
        Sets up the ASR model and temporary directory for audio processing.
        """
        # Initialize ASR model using factory pattern
        self._asr_model = RecognizerFactory.create(
            model_name=ASR_MODEL_NAME,
            device=ASR_DEVICE
        )
        
        # Setup temporary directory for audio files
        self.tmp_dir = Path(AUDIO_TEMP_DIR)
        if not self.tmp_dir.exists():
            os.makedirs(self.tmp_dir, exist_ok=True)

    def batch_transcribe(
        self,
        audio_paths: List[Path],
        timestamp_granularities: List[Union[str, None]]
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.
        
        Optimizes performance by grouping requests with similar timestamp requirements
        and processing them together. Supports both timestamped and non-timestamped
        transcriptions in the same batch.
        
        Args:
            audio_paths: List of paths to audio files to transcribe
            timestamp_granularities: List specifying timestamp requirements for each file
            
        Returns:
            List of transcription results corresponding to input files
        """
        # Convert Path objects to strings for ASR model compatibility
        audio_paths = [str(path) for path in audio_paths]

        # Separate requests by timestamp requirements for optimization
        ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

        # Initialize output array
        outputs: List[Union[Any]] = [None] * len(timestamp_granularities)
        
        # Process files without timestamps (more efficient)
        if no_ts_indices:
            transcriptions = self._asr_model.transcribe_audio(
                audio=audio_paths,
                enable_timestamps=False
            )
            # Map results back to original request order
            for local_idx, global_idx in enumerate(no_ts_indices):
                outputs[global_idx] = transcriptions[local_idx]

        # Process files with timestamps (word/segment level)
        if ts_indices:
            transcriptions = self._asr_model.transcribe_audio(
                audio=audio_paths,
                enable_timestamps=True
            )
            # Map results back to original request order
            for local_idx, global_idx in enumerate(ts_indices):
                outputs[global_idx] = transcriptions[local_idx]
                
        return outputs

    @serve.batch(
        max_batch_size=MAX_BATCH_SIZE,
        batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S
    )
    async def batched_transcribe(
        self,
        batch: List[bytes],
        timestamp_granularities: List[Union[str, None]]
    ) -> List[TranscriptionResult]:
        """
        Batched transcription endpoint with Ray Serve automatic batching.
        
        Automatically batches individual requests for improved throughput.
        Handles temporary file management and cleanup.
        
        Args:
            batch: List of audio data as bytes
            timestamp_granularities: Timestamp requirements for each audio file
            
        Returns:
            List of transcription results
        """
        # Save audio bytes to temporary files
        audio_paths = [save_temp_audio(audio_bytes) for audio_bytes in batch]
        
        try:
            # Process transcriptions
            transcriptions = self.batch_transcribe(
                audio_paths=audio_paths,
                timestamp_granularities=timestamp_granularities
            )
            return transcriptions
        finally:
            # Clean up temporary files
            clean_up_temp_audio(audio_paths)

    @app.post("/v1/audio/transcriptions")
    async def transcribe_endpoint(
        self,
        file: UploadFile = File(...),
        model: str = Form("gpt-4o-transcribe"),
        timestamp_granularity: Optional[str] = Form(
            default=None, alias="timestamp_granularities[]"
        ),
        response_format: str = Form("verbose_json")
    ):
        """
        Main transcription endpoint for audio files.
        
        Accepts audio uploads and returns transcriptions in various formats
        based on the requested granularity. Supports text-only, word-level,
        and segment-level transcriptions.
        
        Args:
            file: Audio file to transcribe
            model: ASR model name (must match configured model)
            timestamp_granularity: Level of timestamp detail (word/segment)
            response_format: Output format (currently verbose_json)
            
        Returns:
            Transcription response in requested format
            
        Raises:
            TranscriptedModelNotFoundException: If requested model is not available
        """
        # Read uploaded audio file
        audio_bytes = await file.read()

        # Validate model name
        if model != ASR_MODEL_NAME:
            raise TranscriptedModelNotFoundException(model)
            
        # Get deployment handle and process transcription
        handle = serve.get_deployment_handle(DEPLOYMENT_NAME)
        transcription_result: TranscriptionResult = await handle.batched_transcribe.remote(
            audio_bytes,
            timestamp_granularity
        )

        # Determine response format based on granularity
        output_type = get_transcription_type(timestamp_granularity)
        
        # Return appropriate response format
        if output_type == TranscriptionType.Text:
            # Simple text transcription with token usage
            output_tokens = await asyncio.to_thread(approximate_count_tokens, transcription_result.text)
            return TranscriptionResponse(
                text=transcription_result.text,
                usage=Usage(
                    input_tokens=0,
                    input_token_details=InputTokenDetails(
                        text_tokens=0,
                        audio_tokens=0
                    ),
                    output_tokens=output_tokens,
                    total_tokens=output_tokens
                )
            )
            
        elif output_type == TranscriptionType.Word:
            # Word-level transcription with timestamps
            lang_property = await asyncio.to_thread(LanguageDetector.detect,transcription_result.text,True)
            duration = await asyncio.to_thread(estimate_audio_duration, audio_bytes)
            return WordResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                words=[word.model_dump() for word in transcription_result.words]
            )
            
        elif output_type == TranscriptionType.Segment:
            # Segment-level transcription with timestamps
            lang_property = await asyncio.to_thread(LanguageDetector.detect,transcription_result.text,True)
            duration = await asyncio.to_thread(estimate_audio_duration, audio_bytes)

            # Build segment list with IDs
            segments = []
            for index, segment in enumerate(transcription_result.segments):
                segments.append(AdvancedTranscribedSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text
                ))
                
            return SegmentResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                segments=segments
            )

# Create and bind the ASR service deployment
deployment = ASRService.bind()

# Start Ray Serve with HTTP server configuration
serve.start(
    detached=False,
    http_options={
        "host": RAY_HOST,
        "port": RAY_PORT
    }
)


# FastAPI components
from fastapi import UploadFile, File, Form, FastAPI
# Ray Serve for deployment
from ray import serve
# Type hints
from typing import Union
# ASR Model
from app.services.asr import RecognizerFactory
# Configuration imports
from app.core.config.system import *
from app.core.config.serving import *
from app.core.config.asr import *
# Utils
from app.utils.audio import load_audio_from_bytes
from app.utils.transcription.helper import (get_transcription_type,
                                            process_batch_transcription)
from app.utils.token_counter import approximate_count_tokens
from app.utils.language_detect import LanguageDetector
from app.utils.audio import is_audio_file
# Schema
from app.schema.transcription.response import *
from app.schema.transcription.base import AdvancedTranscribedSegment
from app.schema.transcription.type import TranscriptionType
from app.schema.transcription.base.usage import *
# Custom exceptions
from app.exceptions.transcription import TranscriptedModelNotFoundException
from app.exceptions.audio import UnsupportedAudioFormatException
from app.exceptions.handlers import common_exception_handler
# Other utils
import logging, math, asyncio
from pathlib import Path

logger = logging.getLogger("ray.serve")

# Define tags metadata for API documentation
tags_metadata = [
    {
        "name": "Audio",
        "description": "Endpoints for interacting with ASR models"
    },
]

# Initialize FastAPI application for ASR service
asr_app = FastAPI(openapi_tags=tags_metadata)

# Register custom exception handler for ASR model not found errors
asr_app.add_exception_handler(TranscriptedModelNotFoundException, common_exception_handler)
asr_app.add_exception_handler(UnsupportedAudioFormatException, common_exception_handler)

@serve.deployment(ray_actor_options={"num_gpus": NUM_GPUS},
                  num_replicas=NUM_REPLICAS,
                  max_ongoing_requests=MAX_ONGOING_REQUESTS)
@serve.ingress(asr_app)
class ASRService:
    """
    Ray Serve deployment for ASR (Automatic Speech Recognition) service.

    This class handles audio transcription requests using configurable ASR models.
    It supports batch processing for improved throughput and provides different
    transcription formats based on client requirements.

    The deployment is configured with:
    - GPU resources based on NUM_GPUS configuration
    - Multiple replicas for scalability (NUM_REPLICAS)
    - Maximum concurrent requests limit (MAX_ONGOING_REQUESTS)
    """

    def __init__(self):
        """
        Initialize the ASR service.

        Sets up the ASR model.
        Raises RuntimeError if model initialization fails.
        """
        # Initialize ASR model using factory pattern with configuration
        self._asr_model = RecognizerFactory.create(model_name=ASR_MODEL_NAME,
                                                   device=ASR_DEVICE)

        # Validate model initialization
        if self._asr_model is None:
            raise RuntimeError(f"Failed to initialize ASR model: {ASR_MODEL_NAME}")

    @serve.batch(max_batch_size=MAX_BATCH_SIZE,
                 batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def batched_transcribe(self,
                                 batch: List[bytes],
                                 timestamp_granularities: List[Union[str, None]]):
        """
        Batched transcription endpoint with Ray Serve automatic batching.

        Automatically batches individual requests for improved throughput.
        Converts audio bytes to tensors directly without temporary files.

        Args:
            batch: List of audio data as bytes
            timestamp_granularities: Timestamp requirements for each audio file

        Returns:
            List of transcription results
        """
        # torchaudio.load + resample is CPU-bound; offloading each item to a thread
        # reduces wall-clock from sum(decode) to max(decode).
        decoded = await asyncio.gather(
            *[asyncio.to_thread(load_audio_from_bytes, audio_bytes) for audio_bytes in batch]
        )
        audio_tensors = [tensor for tensor, _ in decoded]
        # Duration is derived from tensor shape to avoid re-parsing audio bytes later.
        durations = [duration for _, duration in decoded]

        transcriptions = process_batch_transcription(asr_model=self._asr_model,
                                                     audio_data=audio_tensors,
                                                     timestamp_granularities=timestamp_granularities)
        for result, duration in zip(transcriptions, durations):
            result.duration = duration
        return transcriptions

    @asr_app.post("/v1/audio/transcriptions",
                  name="Transcribe audio files with optional timestamps",
                  tags=["Audio"])
    async def transcribe_audio(self,
                               file: UploadFile = File(...),
                               model: str = Form(ASR_MODEL_NAME),
                               timestamp_granularity: Optional[Literal["word", "segment"]] = Form(
                                   default=None,
                                   alias="timestamp_granularities[]",
                                   description="Level of timestamp detail: 'word' for word-level timestamps, 'segment' for segment-level timestamps, or None for no timestamps"),
                               response_format: str = Form("verbose_json")):
        """
        ## Transcribe audio files with optional timestamps.

        ### Args:
        - `file`: Audio file to transcribe
        - `model`: ASR model name (must match configured model)
        - `timestamp_granularity`: Level of timestamp detail (word/segment)
        - `response_format`: Output format (currently verbose_json)

        ### Returns:
        - Transcription response in requested format

        ### Raises:
        - `TranscriptedModelNotFoundException`: If requested model is not available
        - `ValueError`: If audio processing fails
        """
        logger.info(f"ASR request received - model: {model}")

        # Validate that requested model matches the loaded model
        # This ensures the client requests a model that is actually available
        if model != self._asr_model.model_name:
            raise TranscriptedModelNotFoundException(model=model)

        # Read uploaded audio file into memory for validation and processing
        audio_bytes = await file.read()

        # Validate that the uploaded file is a valid audio format
        if not is_audio_file(audio_bytes):
            # Extract file extension from filename without the dot (e.g., "mp3", "wav")
            file_extension = Path(file.filename).suffix.lstrip('.') if file.filename else ""
            raise UnsupportedAudioFormatException(file_format=file_extension)

        # Get deployment handle and process transcription
        handle = serve.get_deployment_handle(DEPLOYMENT_NAME)
        # Get the result
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
            # Return transcription response
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
            lang_property = LanguageDetector.detect(transcription_result.text, True)
            duration = transcription_result.duration

            # Return transcription response
            return WordResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                words=[word.model_dump() for word in transcription_result.words]
            )

        elif output_type == TranscriptionType.Segment:
            # Segment-level transcription with timestamps
            lang_property = LanguageDetector.detect(transcription_result.text, True)
            duration = transcription_result.duration

            # Build segment list with IDs
            segments = []
            for index, segment in enumerate(transcription_result.segments):
                segments.append(AdvancedTranscribedSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text
                ))

            # Return transcription response
            return SegmentResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                segments=segments
            )

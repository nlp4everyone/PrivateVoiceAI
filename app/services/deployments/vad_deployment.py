# FastAPI components
from fastapi import UploadFile, File, Form, FastAPI
# Ray Serve for deployment
from ray import serve
# Type hints
from typing import List
# VAD Model
from app.services.vad.factory import VADFactory
# File system utilities
from pathlib import Path

# Configuration imports
from app.core.config.system import *
from app.core.config.serving import *
from app.core.config.vad import *
# Utils
from app.utils.audio.io import (estimate_audio_duration,
                                is_audio_file)
# Schema
from app.schema.vad import (VoiceActivityResponse,
                            VoiceActivitySegment)
from app.schema.segment import VADSegment
# Custom exceptions
from app.exceptions.vad import VADModelNotFoundException, UnsupportedFileFormatException
from app.exceptions.handlers import common_exception_handler
import logging, os

logger = logging.getLogger("ray.serve")

# Define tags metadata for API documentation
tags_metadata = [
    {
        "name": "VAD",
        "description": "Contains operations related to Voice Activity Detection",
    },
]

# Initialize FastAPI application for VAD service
vad_app = FastAPI(openapi_tags=tags_metadata)

# Register custom exception handler for VAD model not found errors
vad_app.add_exception_handler(VADModelNotFoundException, common_exception_handler)
vad_app.add_exception_handler(UnsupportedFileFormatException, common_exception_handler)

@serve.deployment(ray_actor_options={"num_gpus": NUM_GPUS},
                  num_replicas=NUM_REPLICAS,
                  max_ongoing_requests=MAX_ONGOING_REQUESTS)
@serve.ingress(vad_app)
class VADService:
    """
    Ray Serve deployment for VAD (Voice Activity Detection) service.

    This class handles voice activity detection requests using configurable VAD models.
    It processes audio files and returns segments where speech is detected.

    The deployment is configured with:
    - GPU resources based on NUM_GPUS configuration
    - Multiple replicas for scalability (NUM_REPLICAS)
    - Maximum concurrent requests limit (MAX_ONGOING_REQUESTS)
    """

    def __init__(self):
        """
        Initialize the VAD service.

        Sets up the VAD model using factory pattern.
        Raises RuntimeError if model initialization fails.
        """
        # Initialize VAD model using factory pattern with configuration
        self._vad_model = VADFactory.create(model_name=VAD_MODEL_NAME,
                                            device=VAD_DEVICE,
                                            token=HF_TOKEN)

        # Validate model initialization
        if self._vad_model is None:
            raise RuntimeError(f"Failed to initialize VAD model: {VAD_MODEL_NAME}")


    @vad_app.post("/v1/audio/activity_detections",
                  name="Detects voice activity segments in audio files with precise timestamps.",
                  response_model = VoiceActivityResponse,
                  tags=["VAD"])
    async def detect_voice_activity(self,
                                    file: UploadFile = File(...),
                                    model: str = Form(VAD_MODEL_NAME)):
        """
        ## Detects voice activity segments in audio files with precise timestamps.

        ### Args:
        - `file`: Audio file to process for voice activity detection
        - `model`: VAD model name (must match configured model)

        ### Returns:
        - `VoiceActivityResponse`: Response containing detected speech segments with start/end timestamps

        ### Raises:
        - `VADModelNotFoundException`: If requested model is not available
        - `ValueError`: If audio processing fails
        """
        logger.info(f"VAD request received - model: {model}")

        # Validate that requested model matches the loaded model
        # This ensures the client requests a model that is actually available
        if model != self._vad_model.model_name:
            raise VADModelNotFoundException(model=model)

        # Read uploaded audio file into memory for validation and processing
        audio_bytes = await file.read()
        
        # Validate that the uploaded file is a valid audio format
        if not is_audio_file(audio_bytes):
            # Extract file extension from filename without the dot (e.g., "mp3", "wav")
            file_extension = Path(file.filename).suffix.lstrip('.') if file.filename else ""
            raise UnsupportedFileFormatException(file_format = file_extension)

        # Run VAD detection directly on audio bytes
        # precision=3 rounds timestamps to 3 decimal places (milliseconds)
        vad_segments: List[VADSegment] = self._vad_model.detect(
            audio=audio_bytes,
            precision=3
        )

        # Convert internal VADSegment objects to API response format
        segments_data = [
            VoiceActivitySegment(start=segment.start, end=segment.end)
            for segment in vad_segments
        ]
        # Estimate duration
        duration = estimate_audio_duration(audio_bytes)

        # Return structured response with detected speech segments
        return VoiceActivityResponse(
            duration=round(duration,3),
            segments=segments_data
        )

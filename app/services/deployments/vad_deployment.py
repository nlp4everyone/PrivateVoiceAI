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
from app.utils.audio.io import (save_temp_audio,
                                clean_up_temp_audio)
# Schema
from app.schema.vad import VoiceActivityResponse, VoiceActivitySegment
from app.schema.segment import VADSegment
# Custom exceptions
from app.exceptions.vad import VADModelNotFoundException
from app.exceptions.handlers import common_exception_handler
import logging, os

logger = logging.getLogger("ray.serve")

# Initialize FastAPI application for VAD service
vad_app = FastAPI()

# Register custom exception handler for VAD model not found errors
vad_app.add_exception_handler(VADModelNotFoundException, common_exception_handler)

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

        Sets up the VAD model using factory pattern and temporary directory for audio processing.
        Raises RuntimeError if model initialization fails.
        """
        # Initialize VAD model using factory pattern with configuration
        self._vad_model = VADFactory.create(model_name=VAD_MODEL_NAME,
                                            device=VAD_DEVICE,
                                            token=HF_TOKEN)
        
        # Validate model initialization
        if self._vad_model is None:
            raise RuntimeError(f"Failed to initialize VAD model: {VAD_MODEL_NAME}")

        # Setup temporary directory for audio file processing
        self.tmp_dir = Path(AUDIO_TEMP_DIR)
        if not self.tmp_dir.exists():
            os.makedirs(self.tmp_dir, exist_ok=True)


    @vad_app.post("/v1/audio/activity_detections", response_model = VoiceActivityResponse)
    async def vad_endpoint(self,
                           file: UploadFile = File(...),
                           model: str = Form(VAD_MODEL_NAME)):
        """
        Voice Activity Detection endpoint for audio files.

        Accepts audio uploads and returns segments where speech is detected.
        Processes the audio using the configured VAD model.

        Args:
            file: Audio file to process for voice activity detection
            model: VAD model name (must match configured model)

        Returns:
            VoiceActivityResponse: Response containing detected speech segments with start/end timestamps

        Raises:
            VADModelNotFoundException: If requested model is not available
            ValueError: If audio processing fails
        """
        logger.info(f"VAD request received - model: {model}")

        # Validate that requested model matches the loaded model
        if model != self._vad_model.model_name:
            raise VADModelNotFoundException(model=model)

        # Read uploaded audio file into memory
        audio_bytes = await file.read()

        # Save audio bytes to temporary file for processing
        temp_audio_path = save_temp_audio(audio_bytes)
        
        try:
            # Run VAD detection on the audio file
            # precision=3 rounds timestamps to 3 decimal places (milliseconds)
            vad_segments: List[VADSegment] = self._vad_model.detect(
                audio=temp_audio_path,
                precision=3
            )
            
            # Convert internal VADSegment objects to API response format
            segments_data = [
                VoiceActivitySegment(start=segment.start, end=segment.end)
                for segment in vad_segments
            ]
            
            # Return structured response with detected speech segments
            return VoiceActivityResponse(
                segments=segments_data
            )
            
        except Exception as e:
            # Log and re-raise any processing errors
            logger.error(f"VAD processing failed: {str(e)}")
            raise ValueError(f"VAD processing failed: {str(e)}")
            
        finally:
            # Ensure temporary audio file is cleaned up regardless of success/failure
            clean_up_temp_audio([temp_audio_path])

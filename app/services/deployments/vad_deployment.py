# Ray Serve for deployment
from ray import serve
# Type hints
from typing import List
# VAD Model
from app.services.vad.factory import VADFactory
# Configuration imports
from app.core.config.serving import *
from app.core.config.vad import *
# Schema
from app.schema.segment import VADSegment
import logging

logger = logging.getLogger("ray.serve")

@serve.deployment(ray_actor_options={"num_gpus": VAD_NUM_GPUS},
                  num_replicas=VAD_NUM_REPLICAS,
                  max_ongoing_requests=VAD_MAX_ONGOING_REQUESTS)
class VADDeployment:
    """
    Ray Serve deployment for VAD (Voice Activity Detection).

    This class handles voice activity detection using configurable VAD models.
    It processes audio bytes and returns segments where speech is detected.

    The deployment is configured with:
    - GPU resources based on NUM_GPUS configuration
    - Multiple replicas for scalability (NUM_REPLICAS)
    - Maximum concurrent requests limit (MAX_ONGOING_REQUESTS)
    """

    def __init__(self):
        """
        Initialize the VAD deployment.

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

    async def __call__(self, audio_bytes: bytes) -> List[VADSegment]:
        """
        Process audio bytes and detect voice activity segments.

        Args:
            audio_bytes: Audio data as bytes

        Returns:
            List of VADSegment objects with start/end timestamps
        """
        # Run VAD detection directly on audio bytes
        # precision=3 rounds timestamps to 3 decimal places (milliseconds)
        vad_segments: List[VADSegment] = await self._vad_model.detect(
            audio=audio_bytes,
            precision=3
        )
        
        logger.info(f"VAD detected {len(vad_segments)} segments")
        return vad_segments

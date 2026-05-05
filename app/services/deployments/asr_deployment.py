# Ray Serve for deployment
from ray import serve
# Type hints
from typing import Union, List
# ASR Model
from app.services.asr import RecognizerFactory
# Configuration imports
from app.core.config.serving import *
from app.core.config.asr import *
# Utils
from app.utils.audio import load_audio_from_bytes
from app.utils.transcription.helper import process_batch_transcription
import torch
import asyncio
# Schema
from app.schema.transcription import TranscriptionResult
import logging

logger = logging.getLogger("ray.serve")

@serve.deployment(ray_actor_options={"num_gpus": ASR_NUM_GPUS},
                  num_replicas=ASR_NUM_REPLICAS,
                  max_ongoing_requests=ASR_MAX_ONGOING_REQUESTS)
class ASRDeployment:
    """
    Ray Serve deployment for ASR (Automatic Speech Recognition).

    This class handles audio transcription using configurable ASR models.
    It supports batch processing for improved throughput.

    The deployment is configured with:
    - GPU resources based on NUM_GPUS configuration
    - Multiple replicas for scalability (NUM_REPLICAS)
    - Maximum concurrent requests limit (MAX_ONGOING_REQUESTS)
    """

    def __init__(self):
        """
        Initialize the ASR deployment.

        Sets up the ASR model using factory pattern.
        Raises RuntimeError if model initialization fails.
        """
        # Initialize ASR model using factory pattern with configuration
        self._asr_model = RecognizerFactory.create(model_name=ASR_MODEL_NAME,
                                                   device=ASR_DEVICE)

        # Validate model initialization
        if self._asr_model is None:
            raise RuntimeError(f"Failed to initialize ASR model: {ASR_MODEL_NAME}")

    @serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE,
                 batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)
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
        # Convert audio bytes to tensors
        audio_tensors = await asyncio.gather(*[load_audio_from_bytes(audio_bytes) for audio_bytes in batch])
        # Process transcriptions
        transcriptions = process_batch_transcription(asr_model=self._asr_model,
                                                     audio_data=audio_tensors,
                                                     timestamp_granularities=timestamp_granularities)
        return transcriptions

    @serve.batch(max_batch_size=ASR_MAX_BATCH_SIZE,
                 batch_wait_timeout_s=ASR_BATCH_WAIT_TIMEOUT_S)
    async def batched_transcribe_tensors(self,
                                         batch: List[torch.Tensor],
                                         timestamp_granularities: List[Union[str, None]]):
        """
        Batched transcription endpoint for pre-loaded audio tensors.

        Optimized for VAD+ASR chaining where audio is already loaded as tensors.
        Skips the bytes-to-tensor conversion step for improved latency.

        Args:
            batch: List of audio data as torch.Tensor
            timestamp_granularities: Timestamp requirements for each audio file

        Returns:
            List of transcription results
        """
        # Process transcriptions directly on tensors (no conversion needed)
        transcriptions = process_batch_transcription(asr_model=self._asr_model,
                                                     audio_data=batch,
                                                     timestamp_granularities=timestamp_granularities)
        return transcriptions

    async def __call__(self, audio_bytes: bytes, timestamp_granularity: Union[str, None] = None) -> TranscriptionResult:
        """
        Transcribe audio bytes.

        Args:
            audio_bytes: Audio data as bytes
            timestamp_granularity: Timestamp requirement (word/segment/None)

        Returns:
            TranscriptionResult object
        """
        return await self.batched_transcribe.remote(audio_bytes, timestamp_granularity)

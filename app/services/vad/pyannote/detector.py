"""
Pyannote-based Voice Activity Detection (VAD) detector implementation.

This module provides a VAD detector using pyannote.audio's pre-trained segmentation
models. It supports detecting speech segments in audio files with configurable
parameters for minimum speech/non-speech durations.
"""
# Pyannote components
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
# Schema
from app.schema.vad import BaseVADDetector
from app.schema.segment import VADSegment
# Typing
from typing import Literal, List, Union
# Other components
import torch, logging
# Utils
from app.utils.audio.io import load_audio_from_bytes
# Logger
logger = logging.getLogger("ray.serve")

# List of supported Pyannote VAD models
SUPPORTED_VAD_MODELS = [
    "pyannote/segmentation-3.0"
]

class PyannoteVADDetector(BaseVADDetector):
    """
    Voice Activity Detection (VAD) detector using Pyannote.audio models.
    
    This class provides voice activity detection capabilities using the
    pyannote.audio library and pre-trained segmentation models.
    """
    
    def __init__(self,
                 token: str,
                 model_name: str = "pyannote/segmentation-3.0",
                 device: Literal["cuda", "cpu", "auto"] = "auto",
                 min_duration_on :float = 0.0,
                 min_duration_off :float = 0.0):
        """
        Initialize the Pyannote VAD detector.
        
        Args:
            token (str): Hugging Face token for accessing private models.
                        Required for pyannote models.
            model_name (str): Name of the pretrained model to load.
                             Defaults to pyannote/segmentation-3.0.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
            min_duration_on (float): Minimum duration (in seconds) for speech regions.
                                    Shorter speech regions will be removed. Defaults to 0.0.
            min_duration_off (float): Minimum duration (in seconds) for non-speech regions.
                                     Shorter non-speech regions will be filled. Defaults to 0.0.
        """
        # Initialize parent class with model name
        super().__init__(model_name = model_name)
        
        # Configure CUDA settings for reproducibility across different hardware
        self._configure_cuda_settings()
        
        # Validate model name is supported
        if model_name not in SUPPORTED_VAD_MODELS:
            # raise ValueError(f"Model '{model_name}' is not supported. Supported models: {SUPPORTED_VAD_MODELS}")
            logger.error(f"Model '{model_name}' is not supported. Supported models: {SUPPORTED_VAD_MODELS}")
            # Fallback to default model name
            self._model_name = SUPPORTED_VAD_MODELS[0]
            logger.warning(f"Using default VAD model: {self._model_name}")
        else:
            logger.info(f"Started VAD model:'{model_name}'")

        # Store authentication token for Hugging Face model access
        self._token = token
        # Store VAD post-processing parameters
        self._min_duration_on = min_duration_on
        self._min_duration_off = min_duration_off
        
        # Determine compute device: auto-detect CUDA if "auto" is specified
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        # Validate device availability and fallback to CPU if CUDA unavailable
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self._device = "cpu"
        
        # Load the pre-trained model and initialize the pipeline
        self.load_model()

    def _configure_cuda_settings(self):
        """
        Configure CUDA backend settings for reproducibility across different CUDA versions.
        
        Disables TF32 to ensure consistent results across different CUDA/cuDNN versions.
        See: https://github.com/pyannote/pyannote-audio/issues/1370
        """
        # Disable TF32 in matrix multiplication for consistent precision
        torch.backends.cuda.matmul.allow_tf32 = False
        # Disable TF32 in cuDNN operations for consistent precision
        torch.backends.cudnn.allow_tf32 = False

    def load_model(self):
        """
        Load the Pyannote model and initialize the VAD pipeline.
        
        This method loads the pre-trained segmentation model from Hugging Face,
        moves it to the specified device, and configures the VAD pipeline
        with hyperparameters for post-processing.
        
        Returns:
            VoiceActivityDetection: The configured VAD pipeline instance.
        """
        # Load pre-trained segmentation model from Hugging Face
        model = Model.from_pretrained(self._model_name,
                                      token = self._token)
        # Move model to the specified device (CPU or CUDA)
        model.to(self._device)

        # Initialize VAD pipeline with the segmentation model
        self._pipeline = VoiceActivityDetection(segmentation=model)
        
        # Configure hyperparameters for post-processing
        HYPER_PARAMETERS = {
            # Remove speech regions shorter than this duration (seconds)
            "min_duration_on": self._min_duration_on,
            # Fill non-speech regions shorter than this duration (seconds)
            "min_duration_off": self._min_duration_off
        }
        self._pipeline.instantiate(HYPER_PARAMETERS)
        return self._pipeline

    @property
    def model_name(self) -> str:
        """
        Get the name of the loaded model.
        
        Returns:
            str: The model name
        """
        return self._model_name

    @property
    def supported_models(self) -> List[str]:
        """
        Get the name of the loaded model.

        Returns:
            str: The model name
        """
        return SUPPORTED_VAD_MODELS
    
    async def detect(self,
               audio: Union[str, bytes],
               precision: int = 3) -> List[VADSegment]:
        """
        Detect voice activity in an audio file or bytes.

        This method processes the audio through the VAD pipeline to identify
        speech segments and returns them as a list of VADSegment objects.

        Args:
            audio (Union[str, bytes]): Path to the audio file or raw audio bytes.
            precision (int): Number of decimal places for timestamp rounding.
                           Defaults to 3.

        Returns:
            List[VADSegment]: List of detected speech segments, where each segment
                            contains start and end timestamps in seconds.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            Exception: For other VAD processing errors.
        """
        # Handle different input types: bytes or file path
        if isinstance(audio, bytes):
            # Convert raw audio bytes to normalized waveform tensor (16kHz mono)
            waveform = await load_audio_from_bytes(audio)
            # Run VAD pipeline on the waveform tensor
            result = self._pipeline({"waveform": waveform.unsqueeze(0) ,
                                     "sample_rate": 16000})
        else:
            # Run VAD pipeline directly on the audio file path
            result = self._pipeline(audio)

        # Convert pyannote Segments to VADSegment objects with rounded timestamps
        # itertracks yields (segment, track, label) tuples; we only need the segment
        segments = []
        for segment, _, _ in result.itertracks(yield_label=True):
            segments.append(VADSegment(
                start=round(segment.start, precision),
                end=round(segment.end, precision)
            ))
        return segments


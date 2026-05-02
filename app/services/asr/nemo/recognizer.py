# BaseRecognizer
from app.schema.transcription.base import (BaseRecognizer,
                                           TranscribedWord,
                                           TranscribedSegment)
from app.schema.transcription.response import TranscriptionResult
# Model
import nemo.collections.asr as nemo_asr
# Typing
from typing import Literal, Union, List
# Dependencies
import torch, logging
logger = logging.getLogger("ray.serve")

# List of supported NVIDIA Parakeet models for ASR
SUPPORTED_MODELS = [
    "nvidia/parakeet-ctc-0.6b-vi",
    "nvidia/parakeet-tdt-0.6b-v3"
]


def convert_ts_to_float(value):
    """
    Convert timestamp string to float seconds.
    
    Args:
        value (str|int): Timestamp in format "HH:MM:SS:ms" or already as float
        
    Returns:
        float: Timestamp in seconds
    """
    if isinstance(value, str):
        h, m, s, ms = map(int, value.split(":"))
        return h * 3600 + m * 60 + s + ms / 1000
    return value

class ParakeetRecognizer(BaseRecognizer):
    """
    NVIDIA NeMo Parakeet ASR model recognizer implementation.

    The model automatically handles device selection (CUDA/CPU) and provides
    word-level and segment-level timestamp information when requested.
    """

    def __init__(self,
                 model_name :str = "nvidia/parakeet-ctc-0.6b-vi",
                 device :Literal["cuda","cpu","auto"] = "auto"):
        """
        Initialize the Parakeet recognizer with a pretrained model.

        Args:
            model_name (str): Name of the pretrained model to load. 
                             Defaults to Vietnamese Parakeet CTC model.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
        """
        # Initialize parent class with model name
        super().__init__(model_name = model_name)
        
        # Validate model name is supported
        if model_name not in SUPPORTED_MODELS:
            logger.error(f"Model '{model_name}' is not supported. Supported models: {SUPPORTED_MODELS}")
            # Fallback to default model name
            self._model_name = SUPPORTED_MODELS[0]
            logger.warning(f"Using default ASR model: {self._model_name}")
        else:
            logger.info(f"Started ASR model:'{model_name}'")
        
        # Define device - auto-detect CUDA availability if "auto" is specified
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        
        # Validate device availability and fallback to CPU if CUDA unavailable
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self._device = "cpu"
        
        # Initialize the pretrained NeMo ASR model
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name= self._model_name)
        
        # Move model to appropriate device for inference
        if self._device == "cuda":
            self.model = self.model.cuda()
        
        # Set model to evaluation mode for consistent inference
        self.model.eval()

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
        Get the list of supported models.

        Returns:
            List[str]: The list of supported model names
        """
        return SUPPORTED_MODELS

    def transcribe(self,
                   audio :Union[str,bytes,List[str],torch.Tensor,List[torch.Tensor]],
                   enable_timestamps :bool = False,
                   precision: int = 3) -> List[TranscriptionResult]:
        """
        Transcribe audio files to text using the Parakeet model.
        
        Args:
            audio (Union[str,bytes,List[str],torch.Tensor,List[torch.Tensor]]): Audio file path(s) or tensor(s) to transcribe.
                                               Can be a single path/tensor or list of paths/tensors.
            enable_timestamps (bool): Whether to include word and segment timestamps.
                                    When True, provides detailed timing information.
            precision (int): Number of decimal places to round timestamps to.
                           Defaults to 3.
        
        Returns:
            List[TranscriptionResult]: List of transcription results, one per audio file.
                                     Each result contains text and optional timestamps.
        
        Raises:
            FileNotFoundError: If any audio file path doesn't exist.
            Exception: For other transcription errors.
        """
        # Normalize input to list format for consistent processing
        if isinstance(audio, (str, torch.Tensor)): audio = [audio]

        # Perform transcription with error handling
        try:
            results = self.model.transcribe(audio, timestamps=enable_timestamps)

            # Extract text from all results
            transcriptions = [result.text for result in results]

            # Handle simple transcription without timestamps
            if not enable_timestamps:
                return [TranscriptionResult(text=transcription) for transcription in transcriptions]

            # Process detailed timestamps when enabled
            detailed_word_timestamps = []
            detailed_segment_timestamps = []
            
            # Extract word and segment timestamps from model results
            batched_word_timestamps = [result.timestamp['word'] for result in results]
            batched_segment_timestamps = [result.timestamp['segment'] for result in results]

            # Convert raw timestamp data to structured objects
            for (batched_word, batched_segment) in zip(batched_word_timestamps, batched_segment_timestamps):
                # Convert word timestamps to TranscribedWord objects with rounded timestamps
                detailed_word_timestamps.append([TranscribedWord(start=round(object.get("start"), precision),
                                                                 end=round(object.get("end"), precision),
                                                                 word=object.get("word")) for object in batched_word])
                # Convert segment timestamps to TranscribedSegment objects
                detailed_segment_timestamps.append([TranscribedSegment(start=round(object.get("start"), precision),
                                                                       end=round(object.get("end"), precision),
                                                                       text=object.get("segment")) for object in
                                                    batched_segment])

            # Return comprehensive transcription results with timestamps
            return [TranscriptionResult(text=transcriptions[index],
                                        segments=detailed_segment_timestamps[index],
                                        words=detailed_word_timestamps[index]) for index in range(len(transcriptions))]
        except Exception as e:
            # Log detailed error information for debugging
            logger.error(f"Error during transcription: {str(e)}")
            logger.error(f"Audio paths: {audio}")
            logger.error(f"Device: {self._device}")
            raise


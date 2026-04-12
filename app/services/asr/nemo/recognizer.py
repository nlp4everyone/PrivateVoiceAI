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
import torch, logging, os
logger = logging.getLogger("ray.serve")


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
    def __init__(self,
                 model_name :str = "nvidia/parakeet-ctc-0.6b-vi",
                 device :Literal["cuda","cpu","auto"] = "auto"):
        """
        Initialize the Parakeet recognizer.
        
        Args:
            model_name (str): Name of the pretrained model to load. 
                             Defaults to Vietnamese Parakeet CTC model.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
        """
        super().__init__(model_name = model_name)
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
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name= model_name)
        
        # Move model to appropriate device for inference
        if self._device == "cuda":
            self.model = self.model.cuda()
        
        # Set model to evaluation mode for consistent inference
        self.model.eval()

    @property
    def model_name(self):
        """
        Get the name of the loaded model.
        
        Returns:
            str: The model name
        """
        return self._model_name

    def transcribe_audio(self,
                         audio :Union[str,bytes,List[str]],
                         enable_timestamps :bool = False) -> List[TranscriptionResult]:
        """
        Transcribe audio files to text using the Parakeet model.
        
        Args:
            audio (Union[str,bytes,List[str]]): Audio file path(s) to transcribe.
                                               Can be a single path or list of paths.
            enable_timestamps (bool): Whether to include word and segment timestamps.
                                    When True, provides detailed timing information.
        
        Returns:
            List[TranscriptionResult]: List of transcription results, one per audio file.
                                     Each result contains text and optional timestamps.
        
        Raises:
            FileNotFoundError: If any audio file path doesn't exist.
            Exception: For other transcription errors.
        """
        # Normalize input to list format for consistent processing
        if isinstance(audio,str): audio = [audio]
        
        # Validate all audio file paths exist before processing
        for audio_path in audio:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Perform transcription with error handling
        try:
            # Call the NeMo model for transcription
            results = self.model.transcribe(audio, timestamps = enable_timestamps)

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
                # Convert word timestamps to TranscribedWord objects
                detailed_word_timestamps.append([TranscribedWord.model_validate(object) for object in batched_word])
                # Convert segment timestamps to TranscribedSegment objects
                detailed_segment_timestamps.append([TranscribedSegment(start=object.get("start"),
                                                                       end=object.get("end"),
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


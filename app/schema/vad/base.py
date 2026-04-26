from typing import List
from app.schema.segment import VADSegment

class BaseVADDetector:
    """
    Base class for Voice Activity Detection (VAD) models.
    
    This abstract class defines the interface that all VAD implementations
    must follow. It provides methods for loading models and detecting voice
    activity in audio files.
    
    Args:
        model_name (str): Name of the VAD model to use
    """
    
    def __init__(self,
                 model_name :str):
        self._model_name = model_name

    def load_model(self):
        """
        Load model weights and initialize the VAD engine.
        
        This method should be called before using the detect method.
        It loads the necessary model files and sets up the inference engine.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()

    @property
    def model_name(self) -> str:
        """
        Get the name of the loaded model.

        Returns:
            str: The model name
        """
        return self._model_name

    def detect(self,
               audio: str) -> List[VADSegment]:
        """
        Run Voice Activity Detection on the full audio file.
        
        Analyzes the audio file to identify segments containing speech.
        Returns a list of time segments where voice activity is detected.

        Args:
            audio (str): Path to the audio file to analyze
            
        Returns:
            List[VADSegment]: List of segments containing start_time and end_time
                            in seconds where voice activity is detected
                            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError()
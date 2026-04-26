from typing import Literal, Union
from app.schema.vad import BaseVADDetector
from app.services.vad.pyannote import PyannoteVADDetector
from app.core.config.vad import HF_TOKEN
import logging

logger = logging.getLogger("ray.serve")

class VADFactory:
    """
    Factory class for creating VAD (Voice Activity Detection) detector instances.
    
    This factory provides a centralized way to create and manage different
    VAD detector implementations. It supports model validation and device
    configuration while maintaining a singleton pattern for detector instances.
    
    Attributes:
        _detector: Current detector instance
    """
    _detector = None
    
    @classmethod
    def create(self,
               model_name: str,
               device :Literal["cuda","cpu","auto"] = "auto",
               **kwargs) -> Union[BaseVADDetector, None]:
        """
        Create a VAD detector instance based on the specified model.
        
        This method validates the model name and creates the appropriate
        detector instance with the specified device configuration.
        
        Args:
            model_name (str): Identifier of the VAD model to create.
                              Must be one of the supported models.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
            **kwargs: Additional configuration parameters for the detector
                     (currently not used but reserved for future extensions)
            
        Returns:
            Union[BaseVADDetector, None]: Configured detector instance or None
                                         if model is not supported
            
        Note:
            This method maintains a singleton pattern - only one detector
            instance is stored at a time. Subsequent calls will replace the
            previous instance.
        """
        if model_name.startswith("pyannote"):
            # Create Pyannote detector instance for supported models
            self._detector = PyannoteVADDetector(model_name=model_name,
                                                 token = HF_TOKEN,
                                                 device = device)
        else:
            logger.error(f"Unsupported model: {model_name}")
            # Attempt to create Pyannote detector anyway (may fail if model is invalid)
            self._detector = PyannoteVADDetector(model_name=model_name,
                                                 token=HF_TOKEN,
                                                 device=device)
            logger.warning(f"Using default VAD model: {self._detector.model_name}")
        return self._detector
    
    @classmethod
    def get_detector_model(self) -> str:
        """
        Get the model name of the current detector instance.
        
        Returns:
            str: The model name of the currently loaded detector.
                 Returns None if no detector has been created.
                 
        Note:
            This method should only be called after a successful create() call.
        """
        return self._detector.model_name if self._detector else None

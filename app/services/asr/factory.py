from typing import Literal, Union
from app.schema.transcription.base.recognizer import BaseRecognizer
from app.services.asr.nemo import ParakeetRecognizer
import logging

logger = logging.getLogger("ray.serve")

class RecognizerFactory:
    """
    Factory class for creating ASR recognizer instances.
    
    This factory provides a centralized way to create and manage different
    ASR recognizer implementations. It supports model validation and device
    configuration while maintaining a singleton pattern for recognizer instances.

    """

    _recognizer = None
    
    @classmethod
    def create(self,
               model_name: str,
               device :Literal["cuda","cpu","auto"] = "auto",
               **kwargs) -> Union[BaseRecognizer, None]:
        """
        Create a recognizer instance based on the specified model.
        
        This method creates the appropriate recognizer instance with the
        specified device configuration. Model validation is handled by the
        recognizer class with fallback to a default model if needed.
        
        Args:
            model_name (str): Identifier of the recognizer model to create.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
            **kwargs: Additional configuration parameters for the recognizer
                     (currently not used but reserved for future extensions)
            
        Returns:
            Union[BaseRecognizer, None]: Configured recognizer instance.
                                         Returns None only if model is not supported.
            
        Note:
            This method maintains a singleton pattern - only one recognizer
            instance is stored at a time. Subsequent calls will replace the
            previous instance.
        """
        if not model_name.startswith("nvidia/parakeet"):
            logger.error(f"Unsupported model: '{model_name}', falling back to default")

        logger.info(f"Loading ASR model '{model_name}' on {device.upper()} ...")
        self._recognizer = ParakeetRecognizer(model_name=model_name, device=device)
        logger.info(f"ASR model '{self._recognizer.model_name}' loaded successfully on {self._recognizer._device.upper()}")
        return self._recognizer
    
    @classmethod
    def get_recognizer_model(self) -> str:
        """
        Get the model name of the current recognizer instance.
        
        Returns:
            str: The model name of the currently loaded recognizer.
                 Returns None if no recognizer has been created.
                 
        Note:
            This method should only be called after a successful create() call.
        """
        return self._recognizer.model_name if self._recognizer else None


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
    
    Attributes:
        _supported_models (List[str]): List of supported model identifiers
        _recognizer (BaseRecognizer): Current recognizer instance
    """

    # List of supported NVIDIA Parakeet models for ASR
    _supported_models = ["nvidia/parakeet-ctc-0.6b-vi",
                         "nvidia/parakeet-tdt-0.6b-v3"]
    _recognizer = None
    
    @classmethod
    def create(self,
               model_name: str,
               device :Literal["cuda","cpu","auto"] = "auto",
               **kwargs) -> Union[BaseRecognizer, None]:
        """
        Create a recognizer instance based on the specified model.
        
        This method validates the model name and creates the appropriate
        recognizer instance with the specified device configuration.
        
        Args:
            model_name (str): Identifier of the recognizer model to create.
                              Must be one of the supported models.
            device (Literal["cuda","cpu","auto"]): Device to use for inference.
                                                   "auto" automatically selects CUDA if available.
            **kwargs: Additional configuration parameters for the recognizer
                     (currently not used but reserved for future extensions)
            
        Returns:
            Union[BaseRecognizer, None]: Configured recognizer instance or None
                                       if model is not supported
            
        Note:
            This method maintains a singleton pattern - only one recognizer
            instance is stored at a time. Subsequent calls will replace the
            previous instance.
        """
        # Validate that the requested model is supported
        if model_name not in self._supported_models:
            error_msg = f"Model '{model_name}' not found. Available: {self._supported_models}"
            logger.error(error_msg)
            return None

        # Create Parakeet recognizer instance for supported models
        # Both supported models use the same ParakeetRecognizer class
        if model_name == self._supported_models[0]:
            # Create CTC-based Parakeet recognizer
            self._recognizer = ParakeetRecognizer(model_name = model_name,
                                                  device = device)
        elif model_name == self._supported_models[1]:
            # Create TDT-based Parakeet recognizer
            self._recognizer = ParakeetRecognizer(model_name = model_name,
                                                  device = device)
        else:
            # This should not be reached due to earlier validation, but included for safety
            error_msg = f"Unsupported model: {model_name}"
            logger.error(error_msg)
        
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


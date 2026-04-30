# Base exception
from ..base_exception import BaseException, BaseResponse
# Typing
from typing import Any
# FastAPI
from fastapi import status

class VADModelNotFoundException(BaseException):
    """
    Exception raised when a requested VAD model is not found or inaccessible.
    
    This exception is used when a client requests a voice activity detection model that
    either doesn't exist in the system or the user doesn't have access to.
    It returns a standardized error response following OpenAI-style API format.
    
    Args:
        model (str): The name of the model that was requested but not found
        type (str): Error type, defaults to "invalid_request_error"
        params (Any, optional): Additional error parameters, defaults to None
        code (Any, optional): Error code, defaults to "model_not_found"
    """
    
    def __init__(self,
                 model:str,
                 type: str = "invalid_request_error",
                 params: Any = None,
                 code: Any = "model_not_found"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            response=BaseResponse(
                message=f"The VAD model `{model}` does not exist",
                type=type,
                params=params,
                code=code
            )
        )

class UnsupportedFileFormatException(BaseException):
    """
    Exception raised when an unsupported file format is provided.
    
    This exception is used when a client provides a file format that is not supported
    by the service. It returns a standardized error response following OpenAI-style API format.
    
    Args:
        file_format (str): The unsupported file format that was provided
        param (str): The parameter name that caused the error, defaults to "file"
        type (str): Error type, defaults to "invalid_request_error"
        code (str): Error code, defaults to "unsupported_value"
    """
    
    def __init__(self,
                 file_format: str,
                 param: str = "file",
                 type: str = "invalid_request_error",
                 code: str = "unsupported_value"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            response=BaseResponse(
                message=f"Unsupported file format {file_format}",
                type=type,
                params=param,
                code=code
            )
        )

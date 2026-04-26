# Base exception
from ..base_exception import (BaseException,
                              BaseResponse)
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

# Base exception
from ..base_exception import (BaseException,
                              BaseResponse)
# FastAPI
from fastapi import status


class UnsupportedAudioFormatException(BaseException):
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
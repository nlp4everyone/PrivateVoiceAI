from typing import Any
from pydantic import BaseModel
from fastapi.responses import JSONResponse

class BaseResponse(BaseModel):
    message: str
    type: str
    params: Any = None
    code: Any = None

class BaseException(Exception):
    def __init__(self, status_code: int, response: BaseResponse):
        self.status_code = status_code
        self.response = response

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status_code,
            content=self.response.model_dump()
        )
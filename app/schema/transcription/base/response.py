from pydantic import BaseModel
from .usage import DurationUsage
from typing import Literal

class BaseResponse(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: str
    duration: float
    text: str
    usage: DurationUsage
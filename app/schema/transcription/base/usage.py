from pydantic import BaseModel
from typing import Literal

class InputTokenDetails(BaseModel):
    text_tokens: int
    audio_tokens: int

class Usage(BaseModel):
    type: Literal["tokens"] = "tokens"
    input_tokens: int
    input_token_details: InputTokenDetails
    output_tokens: int
    total_tokens: int

class DurationUsage(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: int

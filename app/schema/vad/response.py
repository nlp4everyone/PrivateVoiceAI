from pydantic import BaseModel
from typing import Literal, List, Optional
from .segments import VoiceActivitySegment

class VoiceActivityResponse(BaseModel):
    task: Literal["voice_activity_detection"] = "voice_activity_detection"
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[VoiceActivitySegment]] = None

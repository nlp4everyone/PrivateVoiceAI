from pydantic import BaseModel
from .base import (Usage,
                   TranscribedWord,
                   TranscribedSegment,
                   AdvancedTranscribedSegment)
from .base.response import BaseResponse
from typing import List, Optional

class TranscriptionResponse(BaseModel):
    text: str
    usage: Usage

class WordResponse(BaseResponse):
    words: List[TranscribedWord]

class SegmentResponse(BaseResponse):
    segments: List[AdvancedTranscribedSegment]

class TranscriptionResult(BaseModel):
    text :str
    words :Optional[List[TranscribedWord]] = None
    segments :Optional[List[TranscribedSegment]] = None
    duration :Optional[float] = None
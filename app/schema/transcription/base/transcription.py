from .timespan import TimeSpan
from typing import Optional, List

class TranscribedSegment(TimeSpan):
    text: str

class TranscribedWord(TimeSpan):
    word: str

class AdvancedTranscribedSegment(TimeSpan):
    id: Optional[int] = None
    seek: Optional[int] = None
    text: Optional[str] = None
    tokens: List[int] = []
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None

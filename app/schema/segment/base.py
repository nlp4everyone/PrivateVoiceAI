from pydantic import BaseModel
from typing import Optional

class TimeSegment(BaseModel):
    start: float
    end: float
    confidence: Optional[float] = None

class VADSegment(TimeSegment):
    pass
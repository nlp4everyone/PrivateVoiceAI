from pydantic import BaseModel

class TimeSpan(BaseModel):
    start: float
    end: float

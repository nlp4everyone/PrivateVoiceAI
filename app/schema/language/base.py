from pydantic import BaseModel
from typing import List

class LanguageDetail(BaseModel):
    language: str   # e.g. "English", "French", "Unknown"
    lang_code: str   # ISO 639-1 code, e.g. "en", "fr", "un"
    percent: int         # % of text in this language
    score: float

class LanguageProperties(BaseModel):
    language: str
    lang_code: str
    percent: int
    details: List[LanguageDetail]

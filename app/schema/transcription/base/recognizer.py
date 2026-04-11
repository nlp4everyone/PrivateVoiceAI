from typing import Union
from app.schema.transcription.response import TranscriptionResult

class BaseRecognizer:
    def __init__(self,
                 model_name :str):
        self._model_name = model_name

    def transcribe_audio(self,
                         audio :Union[str,bytes]) -> TranscriptionResult:
        raise NotImplementedError()
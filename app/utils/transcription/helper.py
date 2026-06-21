from typing import List, Union, Tuple, Any
from pathlib import Path
import torch
from app.schema.transcription import TranscriptionType
from app.schema.transcription.response import TranscriptionResult


def get_timestamp_indices(timestamp_granularities: List[Union[str, None]]) -> Tuple[List[int], List[int]]:
    """
    Partition timestamp granularities into two groups: with and without timestamps.
    
    Args:
        timestamp_granularities: List of timestamp granularity settings
        
    Returns:
        Tuple of (ts_indices, no_ts_indices) where:
        - ts_indices: indices of items with timestamps
        - no_ts_indices: indices of items without timestamps
    """
    timestamps_flags: List[bool] = []
    # Decide timestamps flag per item
    for cfg in timestamp_granularities:
        timestamps_flags.append(False if cfg is None else True)

    # Partition indices into two groups: no_ts and ts
    no_ts_indices: List[int] = []
    ts_indices: List[int] = []
    for i, ts in enumerate(timestamps_flags):
        if ts:
            ts_indices.append(i)
        else:
            no_ts_indices.append(i)
    return ts_indices, no_ts_indices


def get_transcription_type(type: Union[str, None]) -> TranscriptionType:
    """
    Convert string type to TranscriptionType enum.
    
    Args:
        type: String representation of transcription type
        
    Returns:
        TranscriptionType enum value
    """
    if type is None:
        return TranscriptionType.Text
    elif type == "word":
        return TranscriptionType.Word
    elif type == "segment":
        return TranscriptionType.Segment
    else:
        return TranscriptionType.Text


def process_batch_transcription(asr_model: Any,
                                audio_data: Union[List[Path], List[torch.Tensor]],
                                timestamp_granularities: List[Union[str, None]]) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio files in batch using a single GPU call.

    - Pure no-ts batch  → timestamps=False (no overhead)
    - Pure ts batch     → timestamps=True
    - Mixed batch       → timestamps=True + strip unused data (1 call instead of 2)

    Args:
        asr_model: ASR model instance to use for transcription
        audio_data: List of paths to audio files or list of audio tensors to transcribe
        timestamp_granularities: List specifying timestamp requirements for each file

    Returns:
        List of transcription results corresponding to input files
    """
    if not isinstance(audio_data[0], torch.Tensor):
        audio_data = [str(path) for path in audio_data]

    ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

    # Pure no-ts: single GPU call without timestamps
    if not ts_indices:
        return asr_model.transcribe(audio=audio_data, enable_timestamps=False)

    # Pure ts or mixed: single GPU call with timestamps
    transcriptions = asr_model.transcribe(audio=audio_data, enable_timestamps=True)

    # Mixed: strip timestamp data for requests that don't need it
    if no_ts_indices:
        no_ts_set = set(no_ts_indices)
        for i in no_ts_set:
            transcriptions[i].words = None
            transcriptions[i].segments = None

    return transcriptions

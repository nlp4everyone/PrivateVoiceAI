from typing import List, Union, Tuple
from app.schema.transcription import TranscriptionType


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

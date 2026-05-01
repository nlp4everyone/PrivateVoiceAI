from typing import List, Union, Tuple, Any
from pathlib import Path
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
                                audio_paths: List[Path],
                                timestamp_granularities: List[Union[str, None]]) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio files in batch.

    Optimizes performance by grouping requests with similar timestamp requirements
    and processing them together. Supports both timestamped and non-timestamped
    transcriptions in the same batch.

    Args:
        asr_model: ASR model instance to use for transcription
        audio_paths: List of paths to audio files to transcribe
        timestamp_granularities: List specifying timestamp requirements for each file

    Returns:
        List of transcription results corresponding to input files
    """
    # Convert Path objects to strings for ASR model compatibility
    audio_paths = [str(path) for path in audio_paths]

    # Separate requests by timestamp requirements for optimization
    ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

    # Initialize output array
    outputs: List[Union[Any]] = [None] * len(timestamp_granularities)

    # Process files without timestamps (more efficient)
    if no_ts_indices:
        transcriptions = asr_model.transcribe_audio(
            audio=audio_paths,
            enable_timestamps=False
        )
        # Map results back to original request order
        for local_idx, global_idx in enumerate(no_ts_indices):
            outputs[global_idx] = transcriptions[local_idx]

    # Process files with timestamps (word/segment level)
    if ts_indices:
        transcriptions = asr_model.transcribe_audio(
            audio=audio_paths,
            enable_timestamps=True
        )
        # Map results back to original request order
        for local_idx, global_idx in enumerate(ts_indices):
            outputs[global_idx] = transcriptions[local_idx]

    return outputs

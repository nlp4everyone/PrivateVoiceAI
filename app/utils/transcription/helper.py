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
                                timestamp_granularities: List[Union[str, None]],
                                split_mixed_batch: bool = True) -> List[TranscriptionResult]:
    """
    Transcribe multiple audio files in batch using a single GPU call.

    - Pure no-ts batch  → 1 GPU call, timestamps=False
    - Pure ts batch     → 1 GPU call, timestamps=True
    - Mixed batch       → 2 GPU sub-calls (ts + no-ts) when split_mixed_batch=True,
                          else 1 GPU call with timestamps=True and strip unused data

    Args:
        asr_model: ASR model instance to use for transcription
        audio_data: List of paths to audio files or list of audio tensors to transcribe
        timestamp_granularities: List specifying timestamp requirements for each file
        split_mixed_batch: When True, mixed batches are split into two sub-calls to
                           avoid GPU→CPU logit transfer and CPU alignment for no-ts items

    Returns:
        List of transcription results corresponding to input files
    """
    if not isinstance(audio_data[0], torch.Tensor):
        audio_data = [str(path) for path in audio_data]

    ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

    # Pure no-ts: single GPU call without timestamps
    if not ts_indices:
        return asr_model.transcribe(audio=audio_data, enable_timestamps=False)

    # Pure ts: single GPU call with timestamps
    if not no_ts_indices:
        return asr_model.transcribe(audio=audio_data, enable_timestamps=True)

    # Mixed: split into two sub-batches so no-ts items skip GPU→CPU logit
    # transfer and CPU alignment entirely
    if split_mixed_batch:
        results: List[TranscriptionResult] = [None] * len(audio_data)

        ts_audio = [audio_data[i] for i in ts_indices]
        for i, r in zip(ts_indices, asr_model.transcribe(audio=ts_audio, enable_timestamps=True)):
            results[i] = r

        no_ts_audio = [audio_data[i] for i in no_ts_indices]
        for i, r in zip(no_ts_indices, asr_model.transcribe(audio=no_ts_audio, enable_timestamps=False)):
            results[i] = r

        return results

    # Fallback: single GPU call with timestamps, strip unused data afterwards
    transcriptions = asr_model.transcribe(audio=audio_data, enable_timestamps=True)
    for i in no_ts_indices:
        transcriptions[i].words = None
        transcriptions[i].segments = None
    return transcriptions

import soundfile as sf
import io, torch, torchaudio, magic, asyncio
magic_mime = magic.Magic(mime=True)

_resamplers = {}

def get_resampler(sr, target_sr):
    key = (sr, target_sr)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(sr, target_sr)
    return _resamplers[key]

def estimate_audio_duration(audio_bytes: bytes) -> float:
    """
    Estimate the duration of audio data in seconds.
    
    Args:
        audio_bytes: Raw audio data bytes
        
    Returns:
        Duration in seconds as a float
    """
    with io.BytesIO(audio_bytes) as f:
        info = sf.info(f)
        return info.frames / info.samplerate

async def load_audio_from_bytes(audio_bytes: bytes,
                                target_sr: int = 16000,
                                normalize: bool = False) -> torch.Tensor:
    """
    Load audio from bytes and convert to mono tensor at target sample rate.

    Args:
        audio_bytes: Raw audio data bytes
        target_sr: Target sample rate
        normalize: Whether to normalize waveform to [-1, 1]

    Returns:
        torch.Tensor: 1D waveform
    """

    buffer = io.BytesIO(audio_bytes)

    try:
        waveform, sr = await asyncio.to_thread(torchaudio.load, buffer)
    except Exception as e:
        raise ValueError(f"Invalid audio bytes: {e}")

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = get_resampler(sr, target_sr)
        waveform = resampler(waveform)

    # Normalize (optional)
    if normalize:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

    return waveform.squeeze(0)

def is_audio_file(data: bytes,
                  buffer_size: int = 2048) -> bool:
    """
    Check if the given bytes represent an audio file.

    Args:
        data: Raw bytes to check for audio content
        buffer_size: Number of bytes to read from the data for MIME detection (default: 2048)

    Returns:
        True if the data appears to be an audio file, False otherwise
    """
    try:
        if not data:
            return False

        # Only need a small prefix
        mime = magic_mime.from_buffer(data[:buffer_size])

        return mime.startswith("audio/") or mime in {
            "application/ogg",  # common edge case
        }

    except Exception:
        return False
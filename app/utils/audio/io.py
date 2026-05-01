from pathlib import Path
from typing import List, Union
import soundfile as sf
import uuid, os, subprocess, io, torch, torchaudio

def save_temp_audio(audio_bytes: bytes,
                    sample_rate :int = 16000) -> Union[Path,None]:
    """
    Convert arbitrary audio bytes (mp3, wav, m4a, etc.) to a normalized WAV file.

    - Output: 16kHz, mono WAV (ASR-friendly)
    - Uses /dev/shm for fast in-memory disk (Linux)
    """

    tmp_input = Path(f"/dev/shm/{uuid.uuid4().hex}.input")
    tmp_output = Path(f"/dev/shm/{uuid.uuid4().hex}.wav")

    try:
        # Write raw input bytes
        with open(tmp_input, "wb") as f:
            f.write(audio_bytes)

        # Convert to WAV using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # overwrite output
                "-i", str(tmp_input),  # input file
                "-ar", str(sample_rate),  # sample rate (16kHz)
                "-ac", "1",  # mono
                "-f", "wav",  # force WAV format
                str(tmp_output),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # Validate output
        if not tmp_output.exists():
            raise RuntimeError(f"WAV file not created: {tmp_output}")

        if tmp_output.stat().st_size == 0:
            raise RuntimeError(f"WAV file is empty: {tmp_output}")

        return tmp_output

    finally:
        # Always clean up input file
        if tmp_input.exists():
            tmp_input.unlink(missing_ok=True)

def clean_up_temp_audio(audio_paths: List[Path]) -> None:
    """
    Remove temporary audio files from disk.
    
    Args:
        audio_paths: List of Path objects pointing to temporary audio files
    """
    for path in audio_paths:
        os.remove(path)

def estimate_audio_duration(audio_bytes: bytes) -> float:
    """
    Estimate the duration of audio data in seconds.
    
    Args:
        audio_bytes: Raw audio data bytes
        
    Returns:
        Duration in seconds as a float
    """
    with io.BytesIO(audio_bytes) as f:
        data, samplerate = sf.read(f)
        duration = len(data) / samplerate
    return duration

def load_audio_from_bytes(audio_bytes: bytes,
                          target_sr :int = 16000) -> torch.Tensor:
    """
    Load audio from bytes and convert to mono tensor at target sample rate.
    
    Args:
        audio_bytes: Raw audio data bytes
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        torch.Tensor: Audio waveform as 1D tensor
    """
    # Create in-memory buffer from bytes
    buffer = io.BytesIO(audio_bytes)
    # Load audio as tensor (channels, time)
    waveform, sr = torchaudio.load(buffer)

   # Convert to mono by averaging channels if stereo/multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)
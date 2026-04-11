from pathlib import Path
from typing import List
import uuid, os, subprocess, io
import soundfile as sf

def save_temp_audio(audio_bytes: bytes,
                    sample_rate :int = 16000) -> Path:
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
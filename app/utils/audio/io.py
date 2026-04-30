from pathlib import Path
from typing import List
import uuid, os, subprocess, io, magic
import soundfile as sf
magic_mime = magic.Magic(mime=True)

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
        info = sf.info(f)
        return info.frames / info.samplerate


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
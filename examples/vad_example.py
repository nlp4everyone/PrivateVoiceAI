# Example: Send voice activity detection request to VAD service
import requests

def send_vad_request(audio_path: str,
                     model: str = "pyannote/segmentation-3.0"):
    """
    Send a voice activity detection request to the VAD service.

    Args:
        audio_path: Path to the audio file to analyze
        model: VAD model name (default: pyannote/segmentation-3.0)

    Returns:
        The JSON response from the service
    """
    # VAD service endpoint URL
    url = "http://localhost:8005/v1/audio/activity_detections"

    # Prepare the multipart form data with the audio file
    files = {
        "file": (audio_path, open(audio_path, "rb"), "audio/wav"),
    }

    # Additional parameters for the VAD model
    data = {
        "model": model
    }

    # Send POST request with the audio file and model parameter
    response = requests.post(url, files=files, data=data)

    return response.json()


if __name__ == "__main__":
    # Path to the sample audio file (Vietnamese audio)
    audio_path = "resources/sample_vi.wav"
    # Send VAD request and get the response
    res = send_vad_request(audio_path = audio_path)
    # Print the voice activity detection results
    print(f"Vietnamese sample response: {res}")
    audio_path = "resources/silent.wav"
    # Send VAD request and get the response
    res = send_vad_request(audio_path=audio_path)
    # Print the voice activity detection results
    print(f"Silent sample response: {res}")


from openai import OpenAI
import time

# Initialize OpenAI client with local server configuration
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="",
)

# Path to the audio file for transcription
audio_path = "resources/sample_vi.mp3"

# Fallback to audio transcription endpoint
try:
    # Record start time for processing measurement
    start_time = time.time()
    
    # Open audio file and send to transcription service
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="Qwen/Qwen3-ASR-1.7B",
            file=audio_file,
            language="vi",
        )
    
    # Calculate and print processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print transcription results and processing metrics
    print("=== Audio Transcription Results ===")
    print(f"Transcribed Text: {transcription.text}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print("=" * 35)
    
except Exception as e2:
    print("Audio transcription also failed:", e2)


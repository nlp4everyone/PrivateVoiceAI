from openai import OpenAI
import json, os, requests

def stream_api_response(audio_path: str,
                        model: str,
                        openai_api_base: str,
                        language :str = "en",
                        chunk_size: int = 8192):
    """
    Perform streaming transcription using raw HTTP requests to the vLLM API server.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        chunk_size (int): Size of chunks to read from streaming response (default: 8192)
    
    Returns:
        None: Prints transcription results to stdout as they stream in
    """

    # Construct the API endpoint URL for audio transcription
    api_url = f"{openai_api_base}/audio/transcriptions"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Transcription-Client"}
    # Open the audio file in binary mode for upload
    with open(audio_path, "rb") as f:
        # Prepare the file for multipart form data upload
        files = {"file": (os.path.basename(audio_path), f)}
        # Set up the request parameters for streaming transcription
        data = {
            "stream": "true",           # Enable streaming response
            "model": model,              # Specify the model to use
            "language": language,       # Set transcription language
            "response_format": "json",  # Request JSON response format
        }

        # Print header for streaming output
        print("\ntranscription result [stream]:", end=" ")
        # Make POST request with streaming enabled
        response = requests.post(api_url,
                                 headers=headers,
                                 files=files,
                                 data=data,
                                 stream=True)
        # Process the streaming response line by line
        for chunk in response.iter_lines(chunk_size = chunk_size,
                                         decode_unicode = False,
                                         delimiter = b"\n"):
            if chunk:
                # Remove the "data: " prefix from Server-Sent Events format
                data = chunk[len("data: ") :]
                # Parse the JSON data from the chunk
                data = json.loads(data.decode("utf-8"))
                # Extract the choice data from OpenAI-style response
                data = data["choices"][0]
                # Get the delta content (transcribed text)
                delta = data["delta"]["content"]
                # Print the transcribed text immediately without newline
                print(delta, end="", flush=True)

                # Check if the stream has finished
                finish_reason = data.get("finish_reason")
                if finish_reason is not None:
                    # Print the completion reason and exit the loop
                    print(f"\n[Stream finished reason: {finish_reason}]")
                    break

def main():
    """
    Main function to demonstrate streaming audio transcription.
    Sets up the OpenAI client and initiates streaming transcription.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8001/v1"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Get the first available model from the server
    model_name = "Qwen/Qwen3-ASR-1.7B"

    # Get the first available model from the server
    print(f"Using model: {model_name}")

    # Start streaming transcription of the audio file
    stream_api_response(audio_path = audio_path,
                        model = model_name,
                        openai_api_base = openai_api_base)

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
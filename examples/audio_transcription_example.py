from openai import AsyncOpenAI
import asyncio, os, time
import httpx

async def transcribe_with_openai(audio_path: str,
                                 model: str,
                                 openai_api_base: str,
                                 language: str = "en",
                                 api_key: str = "EMPTY") -> str:
    """
    Perform asynchronous audio transcription using the official OpenAI client interface.

    Args:
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        api_key (str): API key for the server (default: "EMPTY")

    Returns:
        str: The transcribed text
    """

    # Initialize the async OpenAI client with local server configuration
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        # Open audio file and send to transcription service
        with open(audio_path, "rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                model = model,
                file = audio_file,
                language = language,
            )

    # The OpenAI client returns a parsed object exposing the text attribute
    return transcription.text

async def transcribe_with_httpx(audio_path: str,
                                model: str,
                                openai_api_base: str,
                                language: str = "en",
                                timeout: float = 60.0) -> str:
    """
    Perform asynchronous audio transcription using raw HTTP requests via the `httpx` library.

    Args:
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        timeout (float): Maximum time in seconds to wait for the response (default: 60.0)

    Returns:
        str: The transcribed text
    """

    # Construct the API endpoint URL for audio transcription
    api_url = f"{openai_api_base}/audio/transcriptions"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Transcription-Client"}
    # Open the audio file in binary mode for upload
    with open(audio_path, "rb") as f:
        # Prepare the file for multipart form data upload
        files = {"file": (os.path.basename(audio_path), f)}
        # Set up the request parameters for non-streaming transcription
        data = {
            "model": model,                 # Specify the model to use
            "language": language,           # Set transcription language
            "response_format": "json",      # Request JSON response format
        }

        # Reuse a single async client so the connection is closed deterministically
        async with httpx.AsyncClient(timeout = timeout) as client:
            response = await client.post(api_url,
                                         headers = headers,
                                         files = files,
                                         data = data)

    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    # Extract the transcribed text from the JSON payload
    return response.json()["text"]

async def run_transcriber(label: str,
                          transcribe,
                          audio_path: str,
                          model: str,
                          openai_api_base: str,
                          language: str = "en"):
    """
    Await a single async transcription client and report its result and timing.

    Args:
        label (str): Human readable name of the client being exercised
        transcribe (Callable): Async transcription function to await
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")

    Returns:
        None: Prints transcription results to stdout
    """

    try:
        # Record start time for processing measurement
        start_time = time.perf_counter()

        # Transcribe the audio file with the current client
        text = await transcribe(audio_path = audio_path,
                                model = model,
                                openai_api_base = openai_api_base,
                                language = language)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        # Print transcription results and processing metrics
        print(f"=== Audio Transcription Results [{label}] ===")
        print(f"Transcribed Text: {text}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print("=" * 35)

    except Exception as e:
        # Report the failure but let the other clients finish
        print(f"Audio transcription failed [{label}]:", e)

async def main():
    """
    Main coroutine to demonstrate asynchronous audio transcription through two
    different clients: the OpenAI SDK and the `httpx` library.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8001/v1"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Model name for audio transcription
    model_name = "Qwen/Qwen3-ASR-1.7B"
    # Language code of the audio content
    language = "vi"

    # Map each client label to its corresponding async transcription function
    transcribers = {
        "OpenAI SDK": transcribe_with_openai,
        "httpx": transcribe_with_httpx,
    }

    # Fire both clients concurrently and wait until all of them complete
    await asyncio.gather(*[
        run_transcriber(label = label,
                        transcribe = transcribe,
                        audio_path = audio_path,
                        model = model_name,
                        openai_api_base = openai_api_base,
                        language = language)
        for label, transcribe in transcribers.items()
    ])

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())

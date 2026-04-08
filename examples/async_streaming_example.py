from openai import AsyncOpenAI
import asyncio

async def stream_openai_response(client: AsyncOpenAI,
                                 audio_path: str,
                                 model: str,
                                 language :str = "en",
                                 temperature :float = 0.0,
                                 seed :int = 420,
                                 top_p :float = 0.6):
    """
    Perform asynchronous transcription using OpenAI-compatible API.
    
    Args:
        client (AsyncOpenAI): Async OpenAI client instance
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        language (str): Language code for transcription (default: "en")
        temperature (float): Sampling temperature for transcription (default: 0.0)
        seed (int): Random seed for reproducible results (default: 420)
        top_p (float): Nucleus sampling parameter (default: 0.6)
    
    Returns:
        None: Prints transcription results to stdout as they stream in
    """
    # Print header for streaming output
    print("\ntranscription result [stream]:", end=" ")
    # Open the audio file in binary mode for transcription
    with open(audio_path, "rb") as f:
        # Create asynchronous transcription request with streaming enabled
        transcription = await client.audio.transcriptions.create(
            file=f,
            model=model,
            language=language,
            response_format="json",
            temperature=temperature,
            # Additional sampling params not provided by OpenAI API.
            extra_body=dict(seed=seed,
                            top_p=top_p),
            stream=True,
        )
        # Process the streaming response asynchronously
        async for chunk in transcription:
            if chunk.choices:
                # Extract the transcribed content from the chunk
                content = chunk.choices[0].get("delta", {}).get("content")
                # Print the transcribed text immediately without newline
                print(content, end="", flush=True)

    # Print final newline after streaming completes
    print()


def main():
    """
    Main function to demonstrate asynchronous streaming audio transcription.
    Sets up the AsyncOpenAI client and initiates async streaming transcription.
    """
    # Configure API connection parameters for vLLM server
    # Note: vLLM uses "EMPTY" as a placeholder API key when not requiring authentication
    openai_api_key = "EMPTY"
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8001/v1"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Get the first available model from the server
    model_name = "Qwen/Qwen3-ASR-1.7B"

    # Initialize asynchronous OpenAI client with custom base URL and empty API key
    client = AsyncOpenAI(api_key=openai_api_key,
                         base_url=openai_api_base)
    # Get the model
    print(f"Using model: {model_name}")
    # Run the async streaming function
    asyncio.run(stream_openai_response(client = client,
                                       audio_path = audio_path,
                                       model = model_name))

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
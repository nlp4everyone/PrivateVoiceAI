from openai import AsyncOpenAI
import asyncio
import time
from typing import List, Dict, Any, Optional
API_KEY = "token"
MODEL_NAME = "nvidia/parakeet-ctc-0.6b-vi"

async def transcribe_audio(client: AsyncOpenAI,
                           audio_path: str,
                           model: str = MODEL_NAME,
                           timestamp_granularities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Send a single transcription request to the ASR service using AsyncOpenAI.

    Args:
        client: AsyncOpenAI client instance
        audio_path: Path to audio file
        model: Model name to use
        timestamp_granularities: List of timestamp granularities (word, segment, or None)

    Returns:
        Transcription response as dictionary
    """
    with open(audio_path, "rb") as audio_file:
        transcript = await client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            timestamp_granularities=timestamp_granularities or []
        )
        return transcript.model_dump()


async def run_concurrent_requests(audio_path: str,
                                  num_requests: int = 4,
                                  base_url: str = "http://localhost:8005/v1",
                                  model: str = "nvidia/parakeet-ctc-0.6b-vi",
                                  timestamp_granularities: Optional[List[str]] = None) -> tuple:
    """
    Run multiple concurrent transcription requests using AsyncOpenAI.

    Args:
        audio_path: Single audio file path to use for all requests
        num_requests: Number of concurrent requests to make
        base_url: Base URL for the API
        model: Model name to use
        timestamp_granularities: Timestamp granularities for requests

    Returns:
        Tuple of (successful_results, errors, latencies)
    """
    client = AsyncOpenAI(base_url=base_url, api_key=API_KEY)

    # Create tasks for concurrent requests with timing
    async def transcribe_with_timing(audio_path: str, index: int):
        start = time.perf_counter()
        try:
            result = await transcribe_audio(client, audio_path, model, timestamp_granularities)
            latency = time.perf_counter() - start
            return result, None, latency, index
        except Exception as e:
            latency = time.perf_counter() - start
            return None, str(e), latency, index

    tasks = []
    for i in range(num_requests):
        task = transcribe_with_timing(audio_path, i)
        tasks.append(task)

    # Execute all requests concurrently
    results = await asyncio.gather(*tasks)

    # Close the client
    await client.close()

    # Separate successful results from exceptions and collect latencies
    successful_results = []
    errors = []
    latencies = []

    for result, error, latency, index in results:
        latencies.append((index, latency))
        if error:
            errors.append((index, error))
        else:
            successful_results.append(result)

    return successful_results, errors, latencies


async def main():
    # Configuration (inherited from audio_transcription_example.py)
    BASE_URL = "http://localhost:8005/v1"
    AUDIO_PATH = "resources/sample_vi.wav"
    NUM_CONCURRENT_REQUESTS = 4

    """Main function to run concurrent transcription tests."""
    print(f"Starting concurrent transcription test with {NUM_CONCURRENT_REQUESTS} requests...")
    print(f"Target endpoint: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 60)

    # Test without timestamps
    print("\n[Test] No timestamps")
    start_time = time.perf_counter()
    results, errors, latencies = await run_concurrent_requests(AUDIO_PATH, NUM_CONCURRENT_REQUESTS, BASE_URL, MODEL_NAME, None)
    elapsed = time.perf_counter() - start_time

    print(f"Completed in {elapsed:.2f}s")
    print(f"Successful requests: {len(results)}/{NUM_CONCURRENT_REQUESTS}")
    print(f"Failed requests: {len(errors)}/{NUM_CONCURRENT_REQUESTS}")

    if errors:
        print("\nErrors:")
        for idx, error in errors:
            print(f"  Request {idx}: {error}")

    # Print individual latencies
    print("\nLatencies:")
    latencies_sorted = sorted(latencies, key=lambda x: x[0])
    for idx, latency in latencies_sorted:
        print(f"  Request {idx}: {latency:.3f}s")

    # Calculate and print statistics
    latency_values = [lat for _, lat in latencies]
    min_latency = min(latency_values)
    max_latency = max(latency_values)
    avg_latency = sum(latency_values) / len(latency_values)

    print(f"\nLatency Statistics:")
    print(f"  Min: {min_latency:.3f}s")
    print(f"  Max: {max_latency:.3f}s")
    print(f"  Avg: {avg_latency:.3f}s")

    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())

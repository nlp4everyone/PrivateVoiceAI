import asyncio
import time
import aiohttp

async def send_vad_request(
    session: aiohttp.ClientSession,
    audio_path: str,
    model: str = "pyannote/segmentation-3.0",
) -> dict:
    """
    Send a voice activity detection request to the VAD service.

    Args:
        session: aiohttp ClientSession for connection pooling
        audio_path: Path to the audio file to analyze
        model: VAD model name (default: pyannote/segmentation-3.0)

    Returns:
        Dict containing response data and latency information
    """
    url = "http://localhost:8005/v1/audio/activity_detections"

    form_data = aiohttp.FormData()
    form_data.add_field("model", model)
    with open(audio_path, "rb") as f:
        form_data.add_field("file", f, filename=audio_path, content_type="audio/wav")

        start_time = time.perf_counter()
        async with session.post(url, data=form_data) as response:
            await response.json()
            end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    return {"latency_ms": latency_ms, "status": response.status}


async def run_concurrent_requests(
    audio_path: str, num_requests: int = 10
) -> list[dict]:
    """
    Send multiple concurrent VAD requests and measure latency.

    Args:
        audio_path: Path to the audio file to analyze
        num_requests: Number of concurrent requests to send

    Returns:
        List of latency results from each request
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_vad_request(session, audio_path) for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


if __name__ == "__main__":
    audio_path = "resources/sample_vi.wav"
    num_requests = 10

    print(f"Sending {num_requests} concurrent requests to VAD service...")
    print(f"Audio file: {audio_path}")
    print("-" * 50)

    overall_start = time.perf_counter()
    results = asyncio.run(run_concurrent_requests(audio_path, num_requests))
    overall_end = time.perf_counter()

    latencies = []
    errors = 0

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"Request {i:2d}: ERROR - {result}")
            errors += 1
        else:
            latency = result["latency_ms"]
            latencies.append(latency)
            print(f"Request {i:2d}: {latency:8.2f} ms (status: {result['status']})")

    print("-" * 50)
    print(f"Total time: {(overall_end - overall_start) * 1000:.2f} ms")
    print(f"Successful: {len(latencies)}/{num_requests}, Errors: {errors}")

    if latencies:
        print(f"\nLatency Statistics:")
        print(f"  Min:    {min(latencies):8.2f} ms")
        print(f"  Max:    {max(latencies):8.2f} ms")
        print(f"  Avg:    {sum(latencies) / len(latencies):8.2f} ms")

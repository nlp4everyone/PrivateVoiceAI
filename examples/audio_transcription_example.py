from openai import OpenAI
import time
client = OpenAI(base_url="http://localhost:8005/v1",
                api_key="token")

audio_file = open("resources/sample_vi.wav", "rb")
begin = time.perf_counter()
transcript = client.audio.transcriptions.create(
  model="nvidia/parakeet-ctc-0.6b-vi",
  file=audio_file,
    timestamp_granularities=["word"]
)
print(transcript)
print(time.perf_counter() - begin)
"""Isolated Silero VAD test on a 16kHz mono WAV file."""
import os
import torch

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "audio_samples", "test_fr_16k.wav")

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
)
get_speech_timestamps, _, read_audio, _, _ = utils

wav = read_audio(AUDIO_PATH, sampling_rate=16000)
timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

print(f"Audio length: {len(wav)/16000:.2f}s")
print(f"Speech segments ({len(timestamps)}):")
for ts in timestamps:
    start = ts["start"] / 16000
    end = ts["end"] / 16000
    print(f"  {start:.2f}s – {end:.2f}s  ({end-start:.2f}s)")

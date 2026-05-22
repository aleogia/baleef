import queue

import numpy as np
import resampy
import sounddevice as sd
import torch

from config import VAD_CHUNK_S, VAD_MAX_PHRASE_S, VAD_SILENCE_CHUNKS
from inference import _run_audio_inference
from models import _infer_lock, get_speech_timestamps, vad_model


def find_loopback_device() -> int:
    """Return the index of the PipeWire loopback input device."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and d["name"] == "pipewire":
            print(f"[init] Loopback device: {i} ({d['name']})")
            return i
    raise RuntimeError("PipeWire input device not found. Is PipeWire running?")


def pipeline_worker(side: str, mic_device: int, sample_rate: int = 48000, channels: int = 1, fixed_src_lang: str | None = None):
    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(VAD_CHUNK_S * 16000)

    def callback(indata, frames, time_info, status):
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        audio_q.put(mono.copy())

    buffer = np.array([], dtype=np.float32)
    speech_buffer = np.array([], dtype=np.float32)
    in_speech = False
    silence_count = 0
    max_phrase_samples = int(VAD_MAX_PHRASE_S * 16000)

    with sd.InputStream(
        device=mic_device, channels=channels, samplerate=sample_rate,
        callback=callback, blocksize=int(sample_rate * 0.1),
    ):
        print(f"[side {side}] Listening on device {mic_device} ({sample_rate}Hz, {channels}ch)...")
        while True:
            raw = audio_q.get().flatten().astype(np.float32)
            chunk_16k = resampy.resample(raw, sample_rate, 16000)
            buffer = np.concatenate([buffer, chunk_16k])

            if len(buffer) < chunk_samples:
                continue

            window = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            with _infer_lock:
                speech_ts = get_speech_timestamps(
                    torch.from_numpy(window), vad_model, sampling_rate=16000
                )

            if speech_ts:
                speech_buffer = np.concatenate([speech_buffer, window])
                in_speech = True
                silence_count = 0
                if len(speech_buffer) >= max_phrase_samples:
                    chunk = speech_buffer
                    speech_buffer = np.array([], dtype=np.float32)
                    try:
                        _run_audio_inference(chunk, side, fixed_src_lang)
                    except Exception as exc:
                        print(f"[side {side}] inference error: {exc}")
            elif in_speech:
                speech_buffer = np.concatenate([speech_buffer, window])
                silence_count += 1
                if silence_count >= VAD_SILENCE_CHUNKS:
                    in_speech = False
                    silence_count = 0
                    if len(speech_buffer) < 16000:
                        speech_buffer = np.array([], dtype=np.float32)
                        continue
                    chunk = speech_buffer
                    speech_buffer = np.array([], dtype=np.float32)
                    try:
                        _run_audio_inference(chunk, side, fixed_src_lang)
                    except Exception as exc:
                        print(f"[side {side}] inference error: {exc}")

import queue

import numpy as np
import resampy
import sounddevice as sd
import torch

from config import VAD_CHUNK_S
from inference import _run_audio_inference
from models import _infer_lock, get_speech_timestamps, vad_model


def find_usb_mics() -> tuple[int, int]:
    """Return (device_A, device_B) indices for the two USB PnP Audio Devices."""
    usb_devices = [
        i for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
        and ("USB PnP Audio Device" in d["name"] or "USB_PnP_Audio_Device" in d["name"])
    ]
    if len(usb_devices) < 2:
        raise RuntimeError(
            f"Expected 2 USB PnP Audio Devices, found {len(usb_devices)}: "
            + str([sd.query_devices(i)["name"] for i in usb_devices])
        )
    print(f"[init] USB mics: A={usb_devices[0]} ({sd.query_devices(usb_devices[0])['name']}), "
          f"B={usb_devices[1]} ({sd.query_devices(usb_devices[1])['name']})")
    return usb_devices[0], usb_devices[1]


def pipeline_worker(side: str, mic_device: int, sample_rate: int = 48000, channels: int = 1, fixed_src_lang: str | None = None):
    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(VAD_CHUNK_S * 16000)

    def callback(indata, frames, time_info, status):
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        audio_q.put(mono.copy())

    buffer = np.array([], dtype=np.float32)
    speech_buffer = np.array([], dtype=np.float32)
    in_speech = False

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
            elif in_speech:
                in_speech = False
                if len(speech_buffer) < 16000:
                    speech_buffer = np.array([], dtype=np.float32)
                    continue
                chunk = speech_buffer
                speech_buffer = np.array([], dtype=np.float32)
                try:
                    _run_audio_inference(chunk, side, fixed_src_lang)
                except Exception as exc:
                    print(f"[side {side}] inference error: {exc}")

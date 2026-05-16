"""Real-time pipeline — mic input, terminal output."""
import os
import queue
import threading
import time

import noisereduce as nr
import numpy as np
import resampy
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

DEVICE_INDEX = 12   # run: python -c "import sounddevice as sd; print(sd.query_devices())"
SAMPLE_RATE = 48000
CHANNELS = 1
VAD_CHUNK_S = 1.0   # seconds to accumulate before each VAD check
TARGET_LANG = "eng_Latn"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")

NLLB_LANG_MAP = {
    "fr": "fra_Latn", "en": "eng_Latn", "es": "spa_Latn",
    "de": "deu_Latn", "ar": "arb_Arab", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "pt": "por_Latn", "ru": "rus_Cyrl",
    "it": "ita_Latn", "hi": "hin_Deva",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
print(f"Device: {device}")

print("Loading Whisper medium...")
whisper = WhisperModel(
    "medium", device=device,
    compute_type="float16" if device == "cuda" else "int8",
)

print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
)
get_speech_timestamps, _, _, _, _ = vad_utils

print("Loading NLLB...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
nmt = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, provider=provider)

audio_queue: queue.Queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def translate(text: str, src_nllb: str, tgt_nllb: str) -> str:
    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt")
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    out = nmt.generate(**inputs, forced_bos_token_id=tgt_id)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def process_loop():
    buffer = np.array([], dtype=np.float32)
    speech_buffer = np.array([], dtype=np.float32)
    in_speech = False
    chunk_samples = int(VAD_CHUNK_S * 16000)

    while True:
        raw = audio_queue.get().flatten().astype(np.float32)
        chunk_16k = resampy.resample(raw, SAMPLE_RATE, 16000)
        buffer = np.concatenate([buffer, chunk_16k])

        if len(buffer) < chunk_samples:
            continue

        window = buffer[:chunk_samples]
        buffer = buffer[chunk_samples:]

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

            audio_16k = nr.reduce_noise(y=speech_buffer, sr=16000, stationary=False)
            speech_buffer = np.array([], dtype=np.float32)

            t0 = time.time()
            segments, info = whisper.transcribe(audio_16k, beam_size=5)
            transcript = " ".join(s.text for s in segments).strip()
            if not transcript:
                continue

            detected = info.language
            src_nllb = NLLB_LANG_MAP.get(detected, "fra_Latn")
            translation = translate(transcript, src_nllb, TARGET_LANG)
            elapsed = int((time.time() - t0) * 1000)
            print(f"[{elapsed}ms] [{detected}] {transcript}  →  {translation}")


print(f"\nListening on device {DEVICE_INDEX} — Ctrl+C to stop\n")
threading.Thread(target=process_loop, daemon=True).start()

with sd.InputStream(
    device=DEVICE_INDEX, channels=CHANNELS, samplerate=SAMPLE_RATE,
    callback=audio_callback, blocksize=int(SAMPLE_RATE * 0.1),
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")

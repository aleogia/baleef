"""File-based full pipeline: VAD → denoise → Whisper STT → NLLB NMT."""
import os
import time

import noisereduce as nr
import numpy as np
import resampy
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "audio_samples", "test_fr_16k.wav")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")
TARGET_LANG = "eng_Latn"

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
nmt = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, provider=provider, use_merged=False)

# Load and resample audio
audio, sr = sf.read(AUDIO_PATH)
if sr != 16000:
    print(f"Resampling {sr}Hz → 16kHz...")
    audio = resampy.resample(audio.astype(np.float32), sr, 16000)
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)/16000:.2f}s at 16kHz")

# VAD — filter silent regions
print("\n[VAD]")
t0 = time.time()
speech_ts = get_speech_timestamps(
    torch.from_numpy(audio), vad_model, sampling_rate=16000
)
print(f"  {len(speech_ts)} speech segment(s) in {int((time.time()-t0)*1000)}ms")
if not speech_ts:
    print("No speech detected.")
    raise SystemExit

speech_audio = np.concatenate([
    audio[ts["start"]:ts["end"]] for ts in speech_ts
])

# Noise suppression
print("\n[Denoise]")
t0 = time.time()
speech_audio = nr.reduce_noise(y=speech_audio, sr=16000, stationary=False)
print(f"  noisereduce: {int((time.time()-t0)*1000)}ms")

# STT
print("\n[STT]")
t0 = time.time()
segments, info = whisper.transcribe(speech_audio, beam_size=5)
transcript = " ".join(s.text for s in segments).strip()
detected_lang = info.language
print(f"  Whisper: {int((time.time()-t0)*1000)}ms  [{detected_lang}] \"{transcript}\"")

# NMT
print("\n[NMT]")
src_nllb = NLLB_LANG_MAP.get(detected_lang, "fra_Latn")
tokenizer.src_lang = src_nllb
inputs = tokenizer(transcript, return_tensors="pt")
tgt_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
t0 = time.time()
out = nmt.generate(**inputs, forced_bos_token_id=tgt_id)
translation = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
print(f"  NLLB: {int((time.time()-t0)*1000)}ms  [{TARGET_LANG}] \"{translation}\"")

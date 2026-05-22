import threading

import ctranslate2
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from config import CT2_MODEL_DIR, MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={device}")

print("[init] Loading Whisper large-v3...")
whisper_model = WhisperModel(
    "large-v3", device=device,
    compute_type="float16" if device == "cuda" else "int8",
)

print("[init] Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
)
get_speech_timestamps, _, _, _, _ = vad_utils

print("[init] Loading NLLB-200...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
nmt_model = ctranslate2.Translator(CT2_MODEL_DIR, device=device, compute_type="int8")
print("[init] Ready.\n")

# VAD, Whisper, and NLLB share the same CUDA context — serialize all GPU calls
_infer_lock = threading.Lock()

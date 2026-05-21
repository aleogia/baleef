import threading

import torch
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

from config import MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
print(f"[init] device={device}")

print("[init] Loading Whisper medium...")
whisper_model = WhisperModel(
    "medium", device=device,
    compute_type="float16" if device == "cuda" else "int8",
)

print("[init] Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
)
get_speech_timestamps, _, _, _, _ = vad_utils

print("[init] Loading NLLB-200...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
nmt_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, provider=provider, use_merged=False)
print("[init] Ready.\n")

# VAD, Whisper, and NLLB share the same CUDA context — serialize all GPU calls
_infer_lock = threading.Lock()

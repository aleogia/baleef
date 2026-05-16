"""One-time export of NLLB-200 distilled-600M to ONNX INT8."""
import os
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_ID = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Exporting {MODEL_ID} to ONNX (this downloads ~2GB and takes 10-20 min)...")
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done — model saved to {OUTPUT_DIR}")

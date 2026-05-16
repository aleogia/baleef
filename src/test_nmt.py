"""Isolated NLLB-200 ONNX translation test."""
import os
import time
import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")

provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
print(f"Loading NLLB from {MODEL_DIR} ({provider})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, provider=provider, use_merged=False)


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    t0 = time.time()
    output = model.generate(**inputs, forced_bos_token_id=tgt_id)
    elapsed = int((time.time() - t0) * 1000)
    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(f"  [{elapsed}ms] {src_lang} → {tgt_lang}: {result}")
    return result


print("\nTranslation tests:")
translate("Bonjour, comment allez-vous ?", "fra_Latn", "eng_Latn")
translate("Hello, how are you today?", "eng_Latn", "fra_Latn")
translate("Buenos días, ¿cómo estás?", "spa_Latn", "eng_Latn")
translate("Guten Morgen, wie geht es Ihnen?", "deu_Latn", "fra_Latn")

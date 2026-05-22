"""Compare ONNX Runtime vs CTranslate2 for NLLB translation — quality + speed."""
import sys
import time

sys.path.insert(0, "src")

import ctranslate2
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

from config import MODEL_DIR

CT2_MODEL_DIR = "models/nllb-600M-ct2"
device = "cuda" if torch.cuda.is_available() else "cpu"
provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"

PHRASES = [
    ("Bonjour, comment allez-vous ?",           "fra_Latn", "eng_Latn"),
    ("Le médecin est en retard.",                "fra_Latn", "eng_Latn"),
    ("J'ai besoin d'un interprète.",             "fra_Latn", "eng_Latn"),
    ("Pouvez-vous répéter plus lentement ?",     "fra_Latn", "eng_Latn"),
    ("Je ne comprends pas ce document.",         "fra_Latn", "eng_Latn"),
    ("Good morning, please take a seat.",        "eng_Latn", "fra_Latn"),
    ("Do you have any allergies?",               "eng_Latn", "fra_Latn"),
    ("The appointment is at three o'clock.",     "eng_Latn", "fra_Latn"),
    ("Please sign here.",                        "eng_Latn", "fra_Latn"),
    ("I don't understand, can you help me?",     "eng_Latn", "fra_Latn"),
    ("Este documento es muy importante.",        "spa_Latn", "fra_Latn"),
    ("Ich spreche kein Französisch.",            "deu_Latn", "eng_Latn"),
]


def translate_onnx(text, src_lang, tgt_lang, tokenizer, model):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    out = model.generate(**inputs, forced_bos_token_id=tgt_id, num_beams=4)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def translate_ct2(text, src_lang, tgt_lang, tokenizer, translator):
    tokenizer.src_lang = src_lang
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids)
    result = translator.translate_batch(
        [tokens],
        target_prefix=[[tgt_lang]],
        beam_size=4,
    )
    output_tokens = result[0].hypotheses[0]
    # strip the target language prefix token
    if output_tokens and output_tokens[0] == tgt_lang:
        output_tokens = output_tokens[1:]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Loading ONNX model...")
t = time.time()
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, provider=provider, use_merged=False)
onnx_load_ms = int((time.time() - t) * 1000)

print("Loading CTranslate2 model...")
t = time.time()
ct2_model = ctranslate2.Translator(CT2_MODEL_DIR, device=device, compute_type="int8")
ct2_load_ms = int((time.time() - t) * 1000)

print(f"\nChargement — ONNX: {onnx_load_ms}ms  |  CT2: {ct2_load_ms}ms\n")
print(f"{'Texte source':<45} {'Sens':<15} {'ONNX':<40} {'CT2':<40} {'Match'}")
print("-" * 160)

onnx_times, ct2_times = [], []
mismatches = 0

for text, src, tgt in PHRASES:
    t = time.time()
    r_onnx = translate_onnx(text, src, tgt, tokenizer, onnx_model)
    onnx_times.append(int((time.time() - t) * 1000))

    t = time.time()
    r_ct2 = translate_ct2(text, src, tgt, tokenizer, ct2_model)
    ct2_times.append(int((time.time() - t) * 1000))

    match = "OK" if r_onnx.strip().lower() == r_ct2.strip().lower() else "DIFF"
    if match == "DIFF":
        mismatches += 1
    label = f"{src[:3]}→{tgt[:3]}"
    print(f"{text:<45} {label:<15} {r_onnx:<40} {r_ct2:<40} {match}")

print("-" * 160)
print(f"\nVitesse moy. — ONNX: {sum(onnx_times)//len(onnx_times)}ms  |  CT2: {sum(ct2_times)//len(ct2_times)}ms")
print(f"Différences de traduction: {mismatches}/{len(PHRASES)}")
if mismatches == 0:
    print("\nResultat: traductions identiques, migration sans risque.")
else:
    print("\nResultat: certaines phrases different — verifier manuellement.")

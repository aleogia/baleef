import json
import os
import re
import threading
import time
import unicodedata

from config import CACHE_PATH
from models import tokenizer, nmt_model, _infer_lock

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_cache_dirty = False
vocab_hints: list[str] = []


def _cache_key(text_norm: str, src: str, tgt: str) -> str:
    return f"{src}|{tgt}|{text_norm}"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def _load_cache():
    global _cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            print(f"[cache] Loaded {len(_cache)} entries from disk.")
        except Exception as e:
            print(f"[cache] Failed to load cache: {e}")


def _save_cache():
    global _cache_dirty
    with _cache_lock:
        if not _cache_dirty:
            return
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_cache, f, ensure_ascii=False, indent=None)
            _cache_dirty = False
        except Exception as e:
            print(f"[cache] Failed to save cache: {e}")


def _cache_saver():
    while True:
        time.sleep(30)
        _save_cache()


def translate(text: str, src_nllb: str, tgt_nllb: str) -> tuple[str, bool]:
    """Returns (translation, from_cache)."""
    global _cache_dirty
    key = _cache_key(_normalize(text), src_nllb, tgt_nllb)
    with _cache_lock:
        if key in _cache:
            return _cache[key], True

    with _infer_lock:
        tokenizer.src_lang = src_nllb
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids)
        output = nmt_model.translate_batch([tokens], target_prefix=[[tgt_nllb]], beam_size=4)
    output_tokens = output[0].hypotheses[0]
    if output_tokens and output_tokens[0] == tgt_nllb:
        output_tokens = output_tokens[1:]
    result = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)

    with _cache_lock:
        _cache[key] = result
        _cache_dirty = True
    return result, False


def add_entry(src_text: str, src_nllb: str, tgt_nllb: str, translation: str):
    """Add an entry directly to the cache (e.g. from glossary import)."""
    global _cache_dirty
    key = _cache_key(_normalize(src_text), src_nllb, tgt_nllb)
    with _cache_lock:
        _cache[key] = translation
        _cache_dirty = True

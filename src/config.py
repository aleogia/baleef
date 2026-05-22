import os

LOOPBACK_SAMPLERATE = 48000
VAD_CHUNK_S = 0.4
VAD_SILENCE_CHUNKS = 3   # 3 × 0.4s = 1.2s de silence pour terminer une phrase
VAD_MAX_PHRASE_S = 8     # durée max avant envoi forcé (même si parole continue)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")
CT2_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-ct2")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "translation_cache.json")
FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "fonts")

TARGET_LANG: dict[str, str] = {
    "A": "eng_Latn",
}

NLLB_LANG_MAP = {
    "fr": "fra_Latn", "en": "eng_Latn", "es": "spa_Latn",
    "de": "deu_Latn", "ar": "arb_Arab", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "pt": "por_Latn", "ru": "rus_Cyrl",
    "it": "ita_Latn", "hi": "hin_Deva",
}

LANG_NAMES = {
    "fra_Latn": "French", "eng_Latn": "English", "spa_Latn": "Spanish",
    "deu_Latn": "German", "arb_Arab": "Arabic", "zho_Hans": "Chinese",
    "jpn_Jpan": "Japanese", "por_Latn": "Portuguese", "rus_Cyrl": "Russian",
    "ita_Latn": "Italian", "hin_Deva": "Hindi",
}

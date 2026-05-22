import asyncio
import time

import noisereduce as nr
import numpy as np

import state
from cache import translate
from config import LANG_NAMES, NLLB_LANG_MAP, TARGET_LANG
from models import _infer_lock, whisper_model


def _run_audio_inference(audio_16k: np.ndarray, side: str, fixed_src_lang: str | None) -> dict | None:
    """Denoise → Whisper → NLLB → broadcast. Returns result dict, or None if no speech detected."""
    audio_16k = nr.reduce_noise(y=audio_16k, sr=16000, stationary=False)
    t0 = time.time()
    with _infer_lock:
        segments, info = whisper_model.transcribe(
            audio_16k, beam_size=2,
            language=fixed_src_lang,
            condition_on_previous_text=False,
        )
        transcript = " ".join(s.text for s in segments).strip()
    if not transcript:
        return None

    detected = fixed_src_lang or info.language
    src_nllb = NLLB_LANG_MAP.get(detected, "fra_Latn")
    tgt_lang = TARGET_LANG[side]
    translation, cached = translate(transcript, src_nllb, tgt_lang)
    elapsed = int((time.time() - t0) * 1000)

    msg = {
        "original": transcript,
        "translated": translation,
        "src_lang": LANG_NAMES.get(src_nllb, detected),
        "tgt_lang": LANG_NAMES.get(tgt_lang, tgt_lang),
        "ms": elapsed,
    }
    print(f"[side {side}] {elapsed}ms {'(cache)' if cached else ''} [{detected}→{tgt_lang}] {transcript!r} → {translation!r}")
    state.log_event("translation",
        side=side, ms=elapsed, cached=cached,
        src_lang=msg["src_lang"], tgt_lang=msg["tgt_lang"],
        original=transcript, translated=translation,
    )
    if state.event_loop and side in state.broadcast_queues:
        asyncio.run_coroutine_threadsafe(state.broadcast_queues[side].put(msg), state.event_loop)
    return msg

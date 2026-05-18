"""Web interface — FastAPI + WebSocket, dual-sided real-time translation kiosk."""
import asyncio
import csv
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import unicodedata
from typing import Dict, Set

import noisereduce as nr
import numpy as np
import resampy
import sounddevice as sd
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
# USB mic indices are resolved dynamically at startup (see find_usb_mics())
DEVICE_A: int = -1
DEVICE_B: int = -1
USB_MIC_SAMPLERATE = 48000
VAD_CHUNK_S = 0.4
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "nllb-600M-onnx-int8")

# Side A hears Person B → translate to Person A's language
# Side B hears Person A → translate to Person B's language
TARGET_LANG: Dict[str, str] = {
    "A": "eng_Latn",   # Person A reads English
    "B": "fra_Latn",   # Person B reads French
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

# ── Model loading ─────────────────────────────────────────────────────────────
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

# ── App & state ───────────────────────────────────────────────────────────────
FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

app = FastAPI()
app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")

display_config: Dict = {
    "bg_color": "#000000",
    "text_color": "#ffffff",
    "font_size": 32,
    "max_phrases": 4,
    "font_family": "'Segoe UI', system-ui, sans-serif",
    "custom_fonts": [],
}
config_clients: Set[WebSocket] = set()

# Source terms from imported glossaries — injected into Whisper initial_prompt
vocab_hints: list[str] = []

ws_clients: Dict[str, Set[WebSocket]] = {"A": set(), "B": set()}
broadcast_queues: Dict[str, asyncio.Queue] = {}
recent_messages: Dict[str, list] = {"A": [], "B": []}  # last 4 per side
event_loop: asyncio.AbstractEventLoop = None
MAX_RECENT = 4

# ── Admin log ─────────────────────────────────────────────────────────────────
admin_log: list = []          # last 200 entries
admin_clients: Set[WebSocket] = set()
MAX_LOG = 200

def log_event(kind: str, **kwargs):
    entry = {"ts": time.strftime("%H:%M:%S"), "kind": kind, **kwargs}
    admin_log.append(entry)
    if len(admin_log) > MAX_LOG:
        admin_log.pop(0)
    if event_loop:
        asyncio.run_coroutine_threadsafe(_broadcast_log(entry), event_loop)

async def _broadcast_log(entry: dict):
    payload = json.dumps(entry)
    dead = set()
    for ws in admin_clients.copy():
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    admin_clients.difference_update(dead)

async def _broadcast_config_ws():
    payload = json.dumps({"kind": "config_update", **display_config})
    dead = set()
    for ws in config_clients.copy():
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    config_clients.difference_update(dead)

# VAD, Whisper, and NLLB share the same CUDA context — serialize all GPU calls
_infer_lock = threading.Lock()

# ── Translation cache ─────────────────────────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "translation_cache.json")
_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()
_cache_dirty = False

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
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(nmt_model.device) for k, v in inputs.items()}
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
        out = nmt_model.generate(**inputs, forced_bos_token_id=tgt_id)
    result = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    with _cache_lock:
        _cache[key] = result
        _cache_dirty = True
    return result, False


def _run_audio_inference(audio_16k: np.ndarray, side: str, fixed_src_lang: str | None) -> dict | None:
    """Denoise → Whisper → NLLB → broadcast. Returns result dict, or None if no speech detected."""
    audio_16k = nr.reduce_noise(y=audio_16k, sr=16000, stationary=False)
    t0 = time.time()
    if fixed_src_lang == "fr":
        hints = ", ".join(vocab_hints[:20])
        prompt = "Conversation en français." + (f" {hints}." if hints else "")
    else:
        prompt = None
    with _infer_lock:
        segments, info = whisper_model.transcribe(
            audio_16k, beam_size=2,
            language=fixed_src_lang,
            initial_prompt=prompt,
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
    log_event("translation",
        side=side, ms=elapsed, cached=cached,
        src_lang=msg["src_lang"], tgt_lang=msg["tgt_lang"],
        original=transcript, translated=translation,
    )
    if event_loop and side in broadcast_queues:
        asyncio.run_coroutine_threadsafe(broadcast_queues[side].put(msg), event_loop)
    return msg


def pipeline_worker(side: str, mic_device: int, sample_rate: int = 48000, channels: int = 1, fixed_src_lang: str | None = None):
    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(VAD_CHUNK_S * 16000)

    def callback(indata, frames, time_info, status):
        # Average channels if multichannel capture
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        audio_q.put(mono.copy())

    buffer = np.array([], dtype=np.float32)
    speech_buffer = np.array([], dtype=np.float32)
    in_speech = False

    with sd.InputStream(
        device=mic_device, channels=channels, samplerate=sample_rate,
        callback=callback, blocksize=int(sample_rate * 0.1),
    ):
        print(f"[side {side}] Listening on device {mic_device} ({sample_rate}Hz, {channels}ch)...")
        while True:
            raw = audio_q.get().flatten().astype(np.float32)
            chunk_16k = resampy.resample(raw, sample_rate, 16000)
            buffer = np.concatenate([buffer, chunk_16k])

            if len(buffer) < chunk_samples:
                continue

            window = buffer[:chunk_samples]
            buffer = buffer[chunk_samples:]

            with _infer_lock:
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
                chunk = speech_buffer
                speech_buffer = np.array([], dtype=np.float32)
                try:
                    _run_audio_inference(chunk, side, fixed_src_lang)
                except Exception as exc:
                    print(f"[side {side}] inference error: {exc}")


# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>lingua_display</title>
  <style>
    :root {
      --bg: #000000;
      --text: #ffffff;
      --font-size: 32px;
      --font-family: 'Segoe UI', system-ui, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-family);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    .side {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 28px 36px;
      gap: 16px;
      min-width: 0;
    }
    .side-a { border-right: 1px solid #1e1e1e; }

    /* ── Top bar ── */
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .side-label {
      font-size: 10px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #383838;
    }
    .lang-fixed {
      font-size: 14px;
      color: #555;
      padding: 7px 12px;
    }
    .lang-select {
      background: #111;
      color: #aaa;
      border: 1px solid #252525;
      border-radius: 8px;
      padding: 7px 32px 7px 12px;
      font-size: 14px;
      cursor: pointer;
      outline: none;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
      transition: border-color 0.2s;
    }
    .lang-select:hover { border-color: #444; color: #fff; }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #1e1e1e;
      transition: background 0.3s;
      flex-shrink: 0;
    }
    .dot.active { background: #22c55e; }

    /* ── Feed: 4 phrases, newest at bottom ── */
    .feed {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      gap: 0;
      overflow: hidden;
    }
    .feed-item {
      padding: 14px 0;
      border-top: 1px solid #141414;
      animation: slideUp 0.3s ease;
    }
    .feed-item:first-child { border-top: none; }
    /* Newest item always full opacity, older items progressively dimmed */
    .feed-item { opacity: 0.15; }
    .feed-item:nth-last-child(4) { opacity: 0.2; }
    .feed-item:nth-last-child(3) { opacity: 0.4; }
    .feed-item:nth-last-child(2) { opacity: 0.7; }
    .feed-item:nth-last-child(1) { opacity: 1; }
    .feed-item .f-translated {
      font-size: var(--font-size);
      font-weight: 300;
      line-height: 1.35;
      word-break: break-word;
    }
    .feed-item .f-original {
      font-size: 13px;
      color: #555;
      font-style: italic;
      margin-top: 4px;
    }
    .feed-item .f-meta {
      font-size: 11px;
      color: #282828;
      margin-top: 3px;
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { transform: translateY(0); }
    }
    #page-controls {
      position: fixed; bottom: 14px; right: 14px;
      display: flex; gap: 6px; opacity: 0.08; transition: opacity 0.25s; z-index: 99;
    }
    #page-controls:hover { opacity: 1; }
    .page-btn {
      background: #111; color: #777; border: 1px solid #2a2a2a;
      border-radius: 5px; padding: 5px 11px; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer;
    }
    .page-btn:hover { color: #ddd; border-color: #444; }
    .page-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="side side-a">
    <div class="topbar">
      <div class="side-label">Side A</div>
      <select class="lang-select" onchange="setLang('A', this.value)">
        <option value="fra_Latn">French</option>
        <option value="eng_Latn" selected>English</option>
        <option value="spa_Latn">Spanish</option>
        <option value="deu_Latn">German</option>
        <option value="arb_Arab">Arabic</option>
        <option value="zho_Hans">Chinese</option>
        <option value="jpn_Jpan">Japanese</option>
        <option value="por_Latn">Portuguese</option>
        <option value="rus_Cyrl">Russian</option>
        <option value="ita_Latn">Italian</option>
        <option value="hin_Deva">Hindi</option>
      </select>
      <div class="dot" id="dotA"></div>
    </div>
    <div class="feed" id="feedA"></div>
  </div>

  <div class="side side-b">
    <div class="topbar">
      <div class="side-label">Side B</div>
      <span class="lang-fixed">Français</span>
      <div class="dot" id="dotB"></div>
    </div>
    <div class="feed" id="feedB"></div>
  </div>

  <script>
    let MAX = 4;

    function applyConfig(cfg) {
      const r = document.documentElement.style;
      if (cfg.bg_color   !== undefined) r.setProperty('--bg',          cfg.bg_color);
      if (cfg.text_color !== undefined) r.setProperty('--text',        cfg.text_color);
      if (cfg.font_size  !== undefined) r.setProperty('--font-size',   cfg.font_size + 'px');
      if (cfg.font_family!== undefined) r.setProperty('--font-family', cfg.font_family);
      if (cfg.max_phrases !== undefined) {
        MAX = cfg.max_phrases;
        ['A','B'].forEach(s => {
          const f = document.getElementById('feed' + s);
          if (f) while (f.children.length > MAX) f.removeChild(f.firstChild);
        });
      }
      if (cfg.custom_fonts) cfg.custom_fonts.forEach(fn => {
        if (document.querySelector('[data-font="' + fn + '"]')) return;
        const nm = fn.replace(/\.[^.]+$/, '').replace(/_/g, ' ');
        const st = document.createElement('style');
        st.setAttribute('data-font', fn);
        st.textContent = "@font-face{font-family:'" + nm + "';src:url('/fonts/" + fn + "')}";
        document.head.appendChild(st);
      });
    }

    function connectConfig() {
      const ws = new WebSocket('ws://' + location.host + '/ws/config');
      ws.onmessage = e => applyConfig(JSON.parse(e.data));
      ws.onclose = () => setTimeout(connectConfig, 2000);
    }

    function setLang(side, code) {
      fetch('/lang/' + side + '?code=' + code, { method: 'POST' });
    }

    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function addItem(side, d) {
      const feed = document.getElementById('feed' + side);

      const item = document.createElement('div');
      item.className = 'feed-item';
      item.innerHTML =
        '<div class="f-translated">' + esc(d.translated) + '</div>' +
        '<div class="f-original">' + esc(d.original) + '</div>' +
        '<div class="f-meta">' + esc(d.src_lang) + ' → ' + esc(d.tgt_lang) + ' · ' + d.ms + 'ms</div>';

      feed.appendChild(item);

      while (feed.children.length > MAX) feed.removeChild(feed.firstChild);
    }

    function connect(side) {
      const dot = document.getElementById('dot' + side);
      const ws  = new WebSocket('ws://' + location.host + '/ws/' + side);

      ws.onmessage = e => {
        const d = JSON.parse(e.data);
        addItem(side, d);
        dot.classList.add('active');
        setTimeout(() => dot.classList.remove('active'), 600);
      };
      ws.onclose = () => setTimeout(() => connect(side), 1500);
    }

    connect('A');
    connect('B');
    connectConfig();

    function hardRefresh() {
      location.replace(location.pathname + '?_=' + Date.now());
    }

    async function restartServer() {
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try { await fetch('/restart', { method: 'POST' }); } catch(e) {}
      const poll = setInterval(() => {
        fetch('/').then(r => { if (r.ok) { clearInterval(poll); location.reload(); } }).catch(() => {});
      }, 1000);
    }
  </script>
  <div id="page-controls">
    <button class="page-btn" onclick="hardRefresh()">Refresh</button>
    <button class="page-btn" id="restart-btn" onclick="restartServer()">Restart</button>
  </div>
</body>
</html>
"""


def make_single_side_html(side: str) -> str:
    lang_ctrl = (
        f"""<select class="lang-select" onchange="setLang('{side}', this.value)">
        {"".join(f'<option value="{c}"{" selected" if c == TARGET_LANG[side] else ""}>{n}</option>' for c, n in LANG_NAMES.items())}
      </select>"""
        if side == "A"
        else '<span class="lang-fixed">Français</span>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>lingua_display — Side {side}</title>
  <style>
    :root {{
      --bg: #000000;
      --text: #ffffff;
      --font-size: 32px;
      --font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: var(--font-family);
      display: flex; flex-direction: column;
      height: 100vh; overflow: hidden;
      padding: 28px 36px; gap: 16px;
    }}
    .topbar {{
      display: flex; align-items: center;
      justify-content: space-between; flex-shrink: 0;
    }}
    .side-label {{
      font-size: 10px; letter-spacing: 3px;
      text-transform: uppercase; color: #383838;
    }}
    .lang-select {{
      background: #111; color: #aaa; border: 1px solid #252525;
      border-radius: 8px; padding: 7px 32px 7px 12px; font-size: 14px;
      cursor: pointer; outline: none; appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center;
    }}
    .lang-fixed {{ font-size: 14px; color: #555; padding: 7px 12px; }}
    .dot {{
      width: 8px; height: 8px; border-radius: 50%;
      background: #1e1e1e; transition: background 0.3s; flex-shrink: 0;
    }}
    .dot.active {{ background: #22c55e; }}
    .feed {{
      flex: 1; display: flex; flex-direction: column;
      justify-content: flex-end; overflow: hidden;
    }}
    .feed-item {{
      padding: 14px 0; border-top: 1px solid #141414;
      animation: slideUp 0.3s ease;
    }}
    .feed-item:first-child {{ border-top: none; }}
    .feed-item {{ opacity: 0.15; }}
    .feed-item:nth-last-child(4) {{ opacity: 0.2; }}
    .feed-item:nth-last-child(3) {{ opacity: 0.4; }}
    .feed-item:nth-last-child(2) {{ opacity: 0.7; }}
    .feed-item:nth-last-child(1) {{ opacity: 1; }}
    .feed-item .f-translated {{
      font-size: var(--font-size); font-weight: 300;
      line-height: 1.35; word-break: break-word;
    }}
    .feed-item .f-original {{ font-size: 13px; color: #555; font-style: italic; margin-top: 4px; }}
    .feed-item .f-meta {{ font-size: 11px; color: #282828; margin-top: 3px; }}
    @keyframes slideUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to   {{ transform: translateY(0); }}
    }}
    #page-controls {{
      position: fixed; bottom: 14px; right: 14px;
      display: flex; gap: 6px; opacity: 0.08; transition: opacity 0.25s; z-index: 99;
    }}
    #page-controls:hover {{ opacity: 1; }}
    .page-btn {{
      background: #111; color: #777; border: 1px solid #2a2a2a;
      border-radius: 5px; padding: 5px 11px; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer;
    }}
    .page-btn:hover {{ color: #ddd; border-color: #444; }}
    .page-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="side-label">Side {side}</div>
    {lang_ctrl}
    <div class="dot" id="dot{side}"></div>
  </div>
  <div class="feed" id="feed{side}"></div>
  <script>
    let MAX = 4;

    function applyConfig(cfg) {{
      const r = document.documentElement.style;
      if (cfg.bg_color    !== undefined) r.setProperty('--bg',          cfg.bg_color);
      if (cfg.text_color  !== undefined) r.setProperty('--text',        cfg.text_color);
      if (cfg.font_size   !== undefined) r.setProperty('--font-size',   cfg.font_size + 'px');
      if (cfg.font_family !== undefined) r.setProperty('--font-family', cfg.font_family);
      if (cfg.max_phrases !== undefined) {{
        MAX = cfg.max_phrases;
        const f = document.getElementById('feed{side}');
        if (f) while (f.children.length > MAX) f.removeChild(f.firstChild);
      }}
      if (cfg.custom_fonts) cfg.custom_fonts.forEach(fn => {{
        if (document.querySelector('[data-font="' + fn + '"]')) return;
        const nm = fn.replace(/\.[^.]+$/, '').replace(/_/g, ' ');
        const st = document.createElement('style');
        st.setAttribute('data-font', fn);
        st.textContent = "@font-face{{font-family:'" + nm + "';src:url('/fonts/" + fn + "')}}";
        document.head.appendChild(st);
      }});
    }}

    function connectConfig() {{
      const ws = new WebSocket('ws://' + location.host + '/ws/config');
      ws.onmessage = e => applyConfig(JSON.parse(e.data));
      ws.onclose = () => setTimeout(connectConfig, 2000);
    }}

    function setLang(side, code) {{
      fetch('/lang/' + side + '?code=' + code, {{ method: 'POST' }});
    }}
    function esc(s) {{
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}
    function addItem(d) {{
      const feed = document.getElementById('feed{side}');
      const item = document.createElement('div');
      item.className = 'feed-item';
      item.innerHTML =
        '<div class="f-translated">' + esc(d.translated) + '</div>' +
        '<div class="f-original">' + esc(d.original) + '</div>' +
        '<div class="f-meta">' + esc(d.src_lang) + ' → ' + esc(d.tgt_lang) + ' · ' + d.ms + 'ms</div>';
      feed.appendChild(item);
      while (feed.children.length > MAX) feed.removeChild(feed.firstChild);
    }}
    function connect() {{
      const dot = document.getElementById('dot{side}');
      const ws = new WebSocket('ws://' + location.host + '/ws/{side}');
      ws.onmessage = e => {{
        addItem(JSON.parse(e.data));
        dot.classList.add('active');
        setTimeout(() => dot.classList.remove('active'), 600);
      }};
      ws.onclose = () => setTimeout(connect, 1500);
    }}
    connect();
    connectConfig();

    function hardRefresh() {{
      location.replace(location.pathname + '?_=' + Date.now());
    }}

    async function restartServer() {{
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try {{ await fetch('/restart', {{ method: 'POST' }}); }} catch(e) {{}}
      const poll = setInterval(() => {{
        fetch('/').then(r => {{ if (r.ok) {{ clearInterval(poll); location.reload(); }} }}).catch(() => {{}});
      }}, 1000);
    }}
  </script>
  <div id="page-controls">
    <button class="page-btn" onclick="hardRefresh()">Refresh</button>
    <button class="page-btn" id="restart-btn" onclick="restartServer()">Restart</button>
  </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.get("/side/{side}", response_class=HTMLResponse)
async def side_view(side: str):
    if side not in ("A", "B"):
        raise HTTPException(status_code=404)
    return make_single_side_html(side)


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>lingua_display — Admin</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f0f; color: #e0e0e0; font-family: 'Segoe UI', monospace, sans-serif; font-size: 13px; }
    header {
      position: sticky; top: 0; background: #0f0f0f;
      border-bottom: 1px solid #222; padding: 14px 24px;
      display: flex; align-items: center; gap: 16px; z-index: 10;
    }
    header h1 { font-size: 14px; font-weight: 600; letter-spacing: 2px; color: #fff; text-transform: uppercase; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; background: #1a1a1a; color: #555; }
    .badge.live { background: #052e16; color: #22c55e; }
    .controls { margin-left: auto; display: flex; gap: 8px; }
    button {
      background: #1a1a1a; color: #888; border: 1px solid #2a2a2a;
      padding: 5px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;
    }
    button:hover { background: #252525; color: #ccc; }
    #log { padding: 12px 0; }
    .row {
      display: grid;
      grid-template-columns: 52px 26px 90px 90px 52px 48px 1fr;
      gap: 0 12px;
      align-items: baseline;
      padding: 7px 24px;
      border-bottom: 1px solid #161616;
      transition: background 0.15s;
    }
    .row:hover { background: #161616; }
    .row.translation { }
    .row.connection { color: #444; }
    .row.lang-change { color: #555; }
    .ts { color: #333; font-variant-numeric: tabular-nums; }
    .side-a { color: #60a5fa; font-weight: 600; }
    .side-b { color: #f472b6; font-weight: 600; }
    .src { color: #888; }
    .tgt { color: #888; }
    .ms { color: #555; font-variant-numeric: tabular-nums; text-align: right; }
    .cache { color: #22c55e; font-size: 11px; }
    .original { color: #555; font-style: italic; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .translated { color: #e0e0e0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .full-span { grid-column: 3 / -1; }
    #stats {
      position: sticky; bottom: 0; background: #0a0a0a;
      border-top: 1px solid #1e1e1e; padding: 8px 24px;
      display: flex; gap: 24px; font-size: 12px; color: #444;
    }
    #stats span { color: #666; }
    #stats b { color: #888; }
    #upload-bar {
      background: #0d0d0d; border-bottom: 1px solid #1e1e1e;
      padding: 10px 24px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    #upload-bar label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
    #upload-side {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      padding: 4px 8px; border-radius: 5px; font-size: 12px; cursor: pointer;
    }
    #upload-file {
      font-size: 12px; color: #777;
      background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 5px;
      padding: 4px 8px; cursor: pointer; flex: 1; min-width: 0;
    }
    #upload-file::file-selector-button {
      background: #252525; color: #888; border: none; border-radius: 4px;
      padding: 3px 8px; font-size: 11px; cursor: pointer; margin-right: 8px;
    }
    #upload-btn {
      background: #1a3a2a; color: #22c55e; border: 1px solid #1e5c3a;
      padding: 5px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; white-space: nowrap;
    }
    #upload-btn:hover { background: #1e4a33; }
    #upload-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    #upload-status { font-size: 12px; color: #555; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #upload-status.ok { color: #22c55e; }
    #upload-status.err { color: #f87171; }

    /* ── Display settings ── */
    #display-bar {
      background: #080808; border-bottom: 1px solid #1a1a1a;
      padding: 8px 24px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    }
    .bar-label { font-size: 10px; color: #3a3a3a; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
    .ctrl-grp { display: flex; align-items: center; gap: 5px; }
    .ctrl-grp > label { font-size: 11px; color: #4a4a4a; white-space: nowrap; }
    .ctrl-sep { width: 1px; height: 18px; background: #1e1e1e; margin: 0 2px; }
    input[type="color"] {
      width: 28px; height: 22px; border: 1px solid #2a2a2a; border-radius: 3px;
      cursor: pointer; background: none; padding: 1px 2px;
    }
    input[type="range"]#cfg-size { width: 72px; accent-color: #22c55e; cursor: pointer; }
    #cfg-size-val { font-size: 11px; color: #555; min-width: 30px; }
    input[type="number"]#cfg-phrases {
      width: 44px; background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 5px; font-size: 12px; text-align: center;
    }
    #cfg-font {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 8px; font-size: 12px; cursor: pointer;
    }
    #font-upload-btn {
      background: #1a1a1a; color: #666; border: 1px solid #2a2a2a;
      padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer;
    }
    #font-upload-btn:hover { color: #aaa; }
    #font-status { font-size: 11px; color: #22c55e; }

    /* ── Glossary ── */
    #glossary-bar {
      background: #070707; border-bottom: 1px solid #161616;
      padding: 8px 24px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    }
    #gls-src, #gls-tgt {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 8px; font-size: 12px; cursor: pointer;
    }
    #gls-import-btn {
      background: #1a1a1a; color: #666; border: 1px solid #2a2a2a;
      padding: 4px 12px; border-radius: 5px; font-size: 12px; cursor: pointer;
    }
    #gls-import-btn:hover { color: #aaa; }
    #gls-status { font-size: 12px; }
    #gls-status.ok { color: #22c55e; }
    #gls-status.err { color: #f87171; }
    .gls-hint { font-size: 10px; color: #2a2a2a; font-style: italic; }
  </style>
</head>
<body>
  <header>
    <h1>lingua_display</h1>
    <div class="badge" id="status">connecting…</div>
    <div class="controls">
      <button onclick="clearLog()">Effacer</button>
      <button onclick="togglePause()" id="pauseBtn">Pause</button>
      <button onclick="hardRefresh()">Refresh</button>
      <button id="admin-restart-btn" onclick="restartServer()">Restart</button>
    </div>
  </header>
  <div id="display-bar">
    <span class="bar-label">Affichage</span>
    <div class="ctrl-grp">
      <label>Fond</label>
      <input type="color" id="cfg-bg" value="#000000" oninput="pushConfig('bg_color', this.value)">
    </div>
    <div class="ctrl-grp">
      <label>Texte</label>
      <input type="color" id="cfg-text" value="#ffffff" oninput="pushConfig('text_color', this.value)">
    </div>
    <div class="ctrl-sep"></div>
    <div class="ctrl-grp">
      <label>Taille</label>
      <input type="range" id="cfg-size" min="10" max="120" value="32"
             oninput="document.getElementById('cfg-size-val').textContent=this.value+'px'; pushConfig('font_size', +this.value)">
      <span id="cfg-size-val">32px</span>
    </div>
    <div class="ctrl-grp">
      <label>Phrases</label>
      <input type="number" id="cfg-phrases" min="1" max="12" value="4"
             onchange="pushConfig('max_phrases', +this.value)">
    </div>
    <div class="ctrl-sep"></div>
    <div class="ctrl-grp">
      <label>Police</label>
      <select id="cfg-font" onchange="pushConfig('font_family', this.value)">
        <option value="'Segoe UI', system-ui, sans-serif">Segoe UI</option>
        <option value="system-ui, sans-serif">System UI</option>
        <option value="Georgia, serif">Georgia</option>
        <option value="'Courier New', monospace">Courier New</option>
        <option value="Arial, sans-serif">Arial</option>
        <option value="'Times New Roman', serif">Times New Roman</option>
      </select>
      <input type="file" id="font-file" accept=".ttf,.otf,.woff,.woff2" style="display:none" onchange="uploadFont(this)">
      <button id="font-upload-btn" onclick="document.getElementById('font-file').click()">+ Police</button>
      <span id="font-status"></span>
    </div>
  </div>

  <div id="glossary-bar">
    <span class="bar-label">Glossaire</span>
    <div class="ctrl-grp">
      <label>Source</label>
      <select id="gls-src">
        <option value="fr">Français</option>
        <option value="en">English</option>
        <option value="es">Español</option>
        <option value="de">Deutsch</option>
        <option value="ar">العربية</option>
        <option value="zh">中文</option>
        <option value="ja">日本語</option>
        <option value="pt">Português</option>
        <option value="ru">Русский</option>
        <option value="it">Italiano</option>
        <option value="hi">हिन्दी</option>
      </select>
    </div>
    <div class="ctrl-grp">
      <label>Traduction vers</label>
      <select id="gls-tgt">
        <option value="eng_Latn">English</option>
        <option value="fra_Latn">Français</option>
        <option value="spa_Latn">Español</option>
        <option value="deu_Latn">Deutsch</option>
        <option value="arb_Arab">العربية</option>
        <option value="zho_Hans">中文</option>
        <option value="jpn_Jpan">日本語</option>
        <option value="por_Latn">Português</option>
        <option value="rus_Cyrl">Русский</option>
        <option value="ita_Latn">Italiano</option>
        <option value="hin_Deva">हिन्दी</option>
      </select>
    </div>
    <input type="file" id="gls-file" accept=".csv,.tsv,.txt" style="display:none" onchange="importGlossary(this)">
    <button id="gls-import-btn" onclick="document.getElementById('gls-file').click()">Importer CSV / TSV</button>
    <span id="gls-status"></span>
    <span class="gls-hint">2 col : source, traduction — 4 col : source, src_lang, tgt_lang, traduction — # = commentaire</span>
  </div>

  <div id="upload-bar">
    <label>Test audio</label>
    <select id="upload-side">
      <option value="A">Side A (source : français)</option>
      <option value="B">Side B (source : auto)</option>
    </select>
    <input type="file" id="upload-file" accept=".wav,.flac,.ogg,.opus">
    <button id="upload-btn" onclick="uploadAudio()">Envoyer</button>
    <span id="upload-status"></span>
  </div>
  <div id="log"></div>
  <div id="stats">
    <span>Traductions : <b id="st-total">0</b></span>
    <span>Cache hits : <b id="st-cache">0</b></span>
    <span>Side A : <b id="st-a">0</b></span>
    <span>Side B : <b id="st-b">0</b></span>
    <span>Latence moy. : <b id="st-avg">—</b></span>
  </div>
  <script>
    let paused = false, total = 0, cacheHits = 0, sideA = 0, sideB = 0, totalMs = 0;

    function togglePause() {
      paused = !paused;
      document.getElementById('pauseBtn').textContent = paused ? 'Reprendre' : 'Pause';
    }
    function clearLog() {
      document.getElementById('log').innerHTML = '';
      total = cacheHits = sideA = sideB = totalMs = 0;
      updateStats();
    }
    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function updateStats() {
      document.getElementById('st-total').textContent = total;
      document.getElementById('st-cache').textContent = cacheHits + (total ? ' (' + Math.round(cacheHits/total*100) + '%)' : '');
      document.getElementById('st-a').textContent = sideA;
      document.getElementById('st-b').textContent = sideB;
      document.getElementById('st-avg').textContent = total ? Math.round(totalMs/total) + ' ms' : '—';
    }
    function addRow(e) {
      if (paused) return;
      const log = document.getElementById('log');
      const row = document.createElement('div');

      if (e.kind === 'translation') {
        total++; totalMs += e.ms;
        if (e.cached) cacheHits++;
        if (e.side === 'A') sideA++; else sideB++;
        updateStats();
        row.className = 'row translation';
        row.innerHTML =
          '<span class="ts">' + esc(e.ts) + '</span>' +
          '<span class="side-' + e.side.toLowerCase() + '">' + esc(e.side) + '</span>' +
          '<span class="src">' + esc(e.src_lang) + '</span>' +
          '<span class="tgt">' + esc(e.tgt_lang) + '</span>' +
          '<span class="ms">' + e.ms + ' ms</span>' +
          '<span class="cache">' + (e.cached ? '⚡cache' : '') + '</span>' +
          '<span class="translated">' + esc(e.translated) + ' <span class="original">← ' + esc(e.original) + '</span></span>';
      } else {
        row.className = 'row ' + (e.kind || 'info');
        row.innerHTML =
          '<span class="ts">' + esc(e.ts) + '</span>' +
          '<span></span>' +
          '<span class="full-span" style="color:#333">' + esc(JSON.stringify(e)) + '</span>';
      }

      log.appendChild(row);
      log.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'end' });

      // Keep max 500 rows in DOM
      while (log.children.length > 500) log.removeChild(log.firstChild);
    }

    async function importGlossary(input) {
      const file = input.files[0];
      if (!file) return;
      const status = document.getElementById('gls-status');
      const src = document.getElementById('gls-src').value;
      const tgt = document.getElementById('gls-tgt').value;
      status.textContent = 'Import…'; status.className = '';
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch('/upload/glossary?src_lang=' + src + '&tgt_lang=' + tgt, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.added !== undefined) {
        let msg = '✓ ' + d.added + ' entrée' + (d.added > 1 ? 's' : '') + ' ajoutée' + (d.added > 1 ? 's' : '');
        if (d.skipped) msg += ' · ' + d.skipped + ' ignorée' + (d.skipped > 1 ? 's' : '');
        if (d.sample && d.sample.length)
          msg += ' — ex : "' + esc(d.sample[0].source) + '" → "' + esc(d.sample[0].translation) + '"';
        status.textContent = msg; status.className = 'ok';
      } else {
        status.textContent = d.detail || 'Erreur'; status.className = 'err';
      }
      input.value = '';
    }

    async function pushConfig(key, value) {
      await fetch('/config/display', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value })
      });
    }

    async function uploadFont(input) {
      const file = input.files[0];
      if (!file) return;
      const status = document.getElementById('font-status');
      status.textContent = '…';
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch('/upload/font', { method: 'POST', body: fd });
      const d = await r.json();
      const sel = document.getElementById('cfg-font');
      const opt = document.createElement('option');
      const ff = "'" + d.font_name + "', sans-serif";
      opt.value = ff; opt.textContent = d.font_name + ' ✓'; opt.selected = true;
      sel.appendChild(opt);
      pushConfig('font_family', ff);
      status.textContent = d.font_name + ' chargée';
      setTimeout(() => status.textContent = '', 3000);
      input.value = '';
    }

    // Init display controls from server
    fetch('/config/display').then(r => r.json()).then(cfg => {
      document.getElementById('cfg-bg').value = cfg.bg_color;
      document.getElementById('cfg-text').value = cfg.text_color;
      document.getElementById('cfg-size').value = cfg.font_size;
      document.getElementById('cfg-size-val').textContent = cfg.font_size + 'px';
      document.getElementById('cfg-phrases').value = cfg.max_phrases;
      const sel = document.getElementById('cfg-font');
      (cfg.custom_fonts || []).forEach(fn => {
        const nm = fn.replace(/\.[^.]+$/, '').replace(/_/g, ' ');
        const opt = document.createElement('option');
        opt.value = "'" + nm + "', sans-serif"; opt.textContent = nm + ' ✓';
        sel.appendChild(opt);
      });
      for (const opt of sel.options) {
        if (opt.value === cfg.font_family) { opt.selected = true; break; }
      }
    });

    async function uploadAudio() {
      const side = document.getElementById('upload-side').value;
      const fileInput = document.getElementById('upload-file');
      const status = document.getElementById('upload-status');
      const btn = document.getElementById('upload-btn');
      if (!fileInput.files.length) { status.textContent = 'Choisir un fichier'; status.className = 'err'; return; }
      btn.disabled = true;
      status.textContent = 'Traitement…';
      status.className = '';
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      try {
        const r = await fetch('/upload/' + side, { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok || d.error) {
          status.textContent = d.detail || d.error || 'Erreur';
          status.className = 'err';
        } else {
          status.textContent = '✓ "' + d.original + '" → "' + d.translated + '" (' + d.ms + ' ms)';
          status.className = 'ok';
        }
      } catch(e) {
        status.textContent = 'Erreur réseau : ' + e.message;
        status.className = 'err';
      } finally {
        btn.disabled = false;
      }
    }

    function hardRefresh() {
      location.replace(location.pathname + '?_=' + Date.now());
    }

    async function restartServer() {
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('admin-restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try { await fetch('/restart', { method: 'POST' }); } catch(e) {}
      const poll = setInterval(() => {
        fetch('/').then(r => { if (r.ok) { clearInterval(poll); location.reload(); } }).catch(() => {});
      }, 1000);
    }

    function connect() {
      const ws = new WebSocket('ws://' + location.host + '/ws/admin');
      const status = document.getElementById('status');
      ws.onopen = () => { status.textContent = 'live'; status.className = 'badge live'; };
      ws.onmessage = e => addRow(JSON.parse(e.data));
      ws.onclose = () => {
        status.textContent = 'reconnecting…'; status.className = 'badge';
        setTimeout(connect, 1500);
      };
    }
    connect();
  </script>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    return ADMIN_HTML


@app.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    await websocket.accept()
    admin_clients.add(websocket)
    for entry in admin_log:
        try:
            await websocket.send_text(json.dumps(entry))
        except Exception:
            break
    try:
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception):
        admin_clients.discard(websocket)


@app.post("/restart")
async def restart_server():
    def _do_restart():
        time.sleep(0.6)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do_restart, daemon=False).start()
    return JSONResponse({"status": "restarting"})


@app.get("/config/display")
async def get_display_config():
    return JSONResponse(display_config)


@app.post("/config/display")
async def set_display_config(body: dict):
    global MAX_RECENT
    for k, v in body.items():
        if k == "font_size":
            display_config[k] = max(10, min(120, int(v)))
        elif k == "max_phrases":
            display_config[k] = max(1, min(20, int(v)))
            MAX_RECENT = display_config[k]
        elif k in {"bg_color", "text_color", "font_family"}:
            display_config[k] = str(v)
    await _broadcast_config_ws()
    return JSONResponse(display_config)


@app.post("/upload/font")
async def upload_font(file: UploadFile = File(...)):
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename or "font")
    path = os.path.join(FONTS_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    if name not in display_config["custom_fonts"]:
        display_config["custom_fonts"].append(name)
    font_name = os.path.splitext(name)[0].replace("_", " ")
    await _broadcast_config_ws()
    return JSONResponse({"filename": name, "url": f"/fonts/{name}", "font_name": font_name})


@app.websocket("/ws/config")
async def ws_config(websocket: WebSocket):
    await websocket.accept()
    config_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"kind": "config_update", **display_config}))
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception):
        config_clients.discard(websocket)


@app.post("/upload/glossary")
async def upload_glossary(file: UploadFile = File(...), src_lang: str = "fr", tgt_lang: str = "eng_Latn"):
    """Import a CSV/TSV glossary into the translation cache.

    2-column format : source, translation
    4-column format : source, src_lang_whisper, tgt_lang_nllb, translation
    Lines starting with # are comments. First row may be a header.
    """
    global _cache_dirty
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    filename = file.filename or ""

    # Detect delimiter
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel

    src_nllb_default = NLLB_LANG_MAP.get(src_lang, "fra_Latn")
    added, skipped, samples = 0, 0, []
    first_data_row = True

    for row in csv.reader(io.StringIO(raw), dialect=dialect):
        row = [c.strip() for c in row]
        if not row or row[0].startswith("#"):
            continue
        if first_data_row:
            first_data_row = False
            if row[0].lower() in ("source", "src", "phrase", "terme", "term", "mot"):
                continue  # skip header

        if len(row) >= 4:
            src_text, r_src, r_tgt, translation = row[0], row[1], row[2], row[3]
            src_nllb = NLLB_LANG_MAP.get(r_src, src_nllb_default)
            tgt_nllb = r_tgt if r_tgt in LANG_NAMES else tgt_lang
        elif len(row) == 2:
            src_text, translation = row[0], row[1]
            src_nllb, tgt_nllb = src_nllb_default, tgt_lang
        else:
            skipped += 1
            continue

        if not src_text or not translation:
            skipped += 1
            continue

        key = _cache_key(_normalize(src_text), src_nllb, tgt_nllb)
        with _cache_lock:
            _cache[key] = translation
            _cache_dirty = True

        if src_text not in vocab_hints:
            vocab_hints.append(src_text)

        if len(samples) < 5:
            samples.append({"source": src_text, "translation": translation})
        added += 1

    log_event("glossary", added=added, skipped=skipped, file=filename)
    return JSONResponse({"added": added, "skipped": skipped, "sample": samples})


@app.post("/upload/{side}")
async def upload_audio(side: str, file: UploadFile = File(...)):
    """Upload an audio file and run it through the full pipeline for the given side."""
    if side not in ("A", "B"):
        raise HTTPException(status_code=404)

    data = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read audio file: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)

    fixed = "fr" if side == "A" else None
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run_audio_inference, audio, side, fixed)

    if result is None:
        return JSONResponse({"error": "no_speech"})
    return JSONResponse(result)


@app.post("/lang/{side}")
async def set_language(side: str, code: str):
    if side not in ("A", "B"):
        raise HTTPException(status_code=404)
    if side == "B":
        raise HTTPException(status_code=403, detail="Side B language is fixed to French.")
    if code not in LANG_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown language code: {code}")
    TARGET_LANG[side] = code
    print(f"[config] side {side} target language → {code} ({LANG_NAMES[code]})")
    return JSONResponse({"side": side, "lang": code, "name": LANG_NAMES[code]})


@app.websocket("/ws/{side}")
async def ws_endpoint(websocket: WebSocket, side: str):
    if side not in ("A", "B"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    ws_clients[side].add(websocket)
    # Replay recent translations so the display isn't blank on connect
    for msg in recent_messages[side]:
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            break
    try:
        while True:
            await websocket.receive()  # handles text, binary, ping/pong without crashing
    except (WebSocketDisconnect, Exception):
        ws_clients[side].discard(websocket)


async def broadcast_worker(side: str):
    q = broadcast_queues[side]
    while True:
        msg = await q.get()
        recent_messages[side].append(msg)
        if len(recent_messages[side]) > MAX_RECENT:
            recent_messages[side].pop(0)
        payload = json.dumps(msg)
        dead: Set[WebSocket] = set()
        for ws in ws_clients[side].copy():
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        ws_clients[side] -= dead


def find_usb_mics() -> tuple[int, int]:
    """Return (device_A, device_B) indices for the two USB PnP Audio Devices."""
    usb_devices = [
        i for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
        and ("USB PnP Audio Device" in d["name"] or "USB_PnP_Audio_Device" in d["name"])
    ]
    if len(usb_devices) < 2:
        raise RuntimeError(
            f"Expected 2 USB PnP Audio Devices, found {len(usb_devices)}: "
            + str([sd.query_devices(i)["name"] for i in usb_devices])
        )
    print(f"[init] USB mics: A={usb_devices[0]} ({sd.query_devices(usb_devices[0])['name']}), "
          f"B={usb_devices[1]} ({sd.query_devices(usb_devices[1])['name']})")
    return usb_devices[0], usb_devices[1]


@app.on_event("startup")
async def startup():
    global event_loop, DEVICE_A, DEVICE_B
    for cmd in ["xset s off", "xset -dpms", "xset s noblank"]:
        subprocess.run(cmd.split(), check=False)
    event_loop = asyncio.get_event_loop()
    broadcast_queues["A"] = asyncio.Queue()
    broadcast_queues["B"] = asyncio.Queue()
    asyncio.create_task(broadcast_worker("A"))
    asyncio.create_task(broadcast_worker("B"))
    _load_cache()
    threading.Thread(target=_cache_saver, daemon=True).start()
    try:
        DEVICE_A, DEVICE_B = find_usb_mics()
        threading.Thread(target=pipeline_worker, args=("A", DEVICE_A, USB_MIC_SAMPLERATE, 1, "fr"), daemon=True).start()
        threading.Thread(target=pipeline_worker, args=("B", DEVICE_B, USB_MIC_SAMPLERATE, 1), daemon=True).start()
    except RuntimeError as e:
        print(f"[init] WARNING: mic pipeline disabled — {e}")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    finally:
        _save_cache()

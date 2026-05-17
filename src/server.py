"""Web interface — FastAPI + WebSocket, dual-sided real-time translation kiosk."""
import asyncio
import json
import os
import queue
import re
import subprocess
import threading
import time
import unicodedata
from typing import Dict, Set

import noisereduce as nr
import numpy as np
import resampy
import sounddevice as sd
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
app = FastAPI()
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

    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(nmt_model.device) for k, v in inputs.items()}
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    with _infer_lock:
        out = nmt_model.generate(**inputs, forced_bos_token_id=tgt_id)
    result = tokenizer.batch_decode(out, skip_special_tokens=True)[0]

    with _cache_lock:
        _cache[key] = result
        _cache_dirty = True
    return result, False


def pipeline_worker(side: str, mic_device: int, sample_rate: int = 48000, channels: int = 1, fixed_src_lang: str | None = None):
    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(VAD_CHUNK_S * 16000)

    def callback(indata, frames, time_info, status):
        # Average channels if multichannel capture
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        audio_q.put(mono.copy())

    def push(msg: dict):
        if event_loop and side in broadcast_queues:
            asyncio.run_coroutine_threadsafe(
                broadcast_queues[side].put(msg), event_loop
            )

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

                audio_16k = nr.reduce_noise(y=speech_buffer, sr=16000, stationary=False)
                speech_buffer = np.array([], dtype=np.float32)

                t0 = time.time()
                prompt = "Conversation en français." if fixed_src_lang == "fr" else None
                with _infer_lock:
                    segments, info = whisper_model.transcribe(
                        audio_16k, beam_size=2,
                        language=fixed_src_lang,
                        initial_prompt=prompt,
                        condition_on_previous_text=False,
                    )
                transcript = " ".join(s.text for s in segments).strip()
                if not transcript:
                    continue

                detected = fixed_src_lang or info.language
                src_nllb = NLLB_LANG_MAP.get(detected, "fra_Latn")
                tgt_lang = TARGET_LANG[side]  # read dynamically — may change via UI
                translation, cached = translate(transcript, src_nllb, tgt_lang)
                elapsed = int((time.time() - t0) * 1000)

                print(f"[side {side}] {elapsed}ms {'(cache)' if cached else ''} [{detected}→{tgt_lang}] {transcript!r} → {translation!r}")
                log_event("translation",
                    side=side, ms=elapsed, cached=cached,
                    src_lang=LANG_NAMES.get(src_nllb, detected),
                    tgt_lang=LANG_NAMES.get(tgt_lang, tgt_lang),
                    original=transcript, translated=translation,
                )
                push({
                    "original": transcript,
                    "translated": translation,
                    "src_lang": LANG_NAMES.get(src_nllb, detected),
                    "tgt_lang": LANG_NAMES.get(tgt_lang, tgt_lang),
                    "ms": elapsed,
                })


# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>lingua_display</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #000000;
      color: #ffffff;
      font-family: 'Segoe UI', system-ui, sans-serif;
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
    /* Oldest → newest opacity */
    .feed-item:nth-child(1) { opacity: 0.2; }
    .feed-item:nth-child(2) { opacity: 0.4; }
    .feed-item:nth-child(3) { opacity: 0.7; }
    .feed-item:nth-child(4) { opacity: 1; }
    .feed-item .f-translated {
      font-size: clamp(20px, 2.8vw, 40px);
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
    const MAX = 4;

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

      // Keep only last MAX items
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
  </script>
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
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #ffffff; color: #111111;
      font-family: 'Segoe UI', system-ui, sans-serif;
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
    .feed-item:nth-child(1) {{ opacity: 0.2; }}
    .feed-item:nth-child(2) {{ opacity: 0.4; }}
    .feed-item:nth-child(3) {{ opacity: 0.7; }}
    .feed-item:nth-child(4) {{ opacity: 1; }}
    .feed-item .f-translated {{
      font-size: clamp(24px, 4vw, 64px); font-weight: 300;
      line-height: 1.35; word-break: break-word;
    }}
    .feed-item .f-original {{ font-size: 13px; color: #555; font-style: italic; margin-top: 4px; }}
    .feed-item .f-meta {{ font-size: 11px; color: #282828; margin-top: 3px; }}
    @keyframes slideUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to   {{ transform: translateY(0); }}
    }}
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
    const MAX = 4;
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
  </script>
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
  </style>
</head>
<body>
  <header>
    <h1>lingua_display</h1>
    <div class="badge" id="status">connecting…</div>
    <div class="controls">
      <button onclick="clearLog()">Effacer</button>
      <button onclick="togglePause()" id="pauseBtn">Pause</button>
    </div>
  </header>
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
    DEVICE_A, DEVICE_B = find_usb_mics()
    threading.Thread(target=pipeline_worker, args=("A", DEVICE_A, USB_MIC_SAMPLERATE, 1, "fr"), daemon=True).start()
    threading.Thread(target=pipeline_worker, args=("B", DEVICE_B, USB_MIC_SAMPLERATE, 1), daemon=True).start()


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    finally:
        _save_cache()

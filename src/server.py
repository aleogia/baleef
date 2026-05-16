"""Web interface — FastAPI + WebSocket, dual-sided real-time translation kiosk."""
import asyncio
import json
import os
import queue
import threading
import time
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
DEVICE_A = 12           # Mic for side A — find index: python -c "import sounddevice as sd; print(sd.query_devices())"
DEVICE_B = 12           # Mic for side B — set to a second device index when two mics are connected
SAMPLE_RATE = 48000
VAD_CHUNK_S = 1.0
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
event_loop: asyncio.AbstractEventLoop = None

# VAD, Whisper, and NLLB share the same CUDA context — serialize all GPU calls
_infer_lock = threading.Lock()


def translate(text: str, src_nllb: str, tgt_nllb: str) -> str:
    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(nmt_model.device) for k, v in inputs.items()}
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    with _infer_lock:
        out = nmt_model.generate(**inputs, forced_bos_token_id=tgt_id)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def pipeline_worker(side: str, mic_device: int):
    audio_q: queue.Queue = queue.Queue()
    chunk_samples = int(VAD_CHUNK_S * 16000)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    def push(msg: dict):
        if event_loop and side in broadcast_queues:
            asyncio.run_coroutine_threadsafe(
                broadcast_queues[side].put(msg), event_loop
            )

    buffer = np.array([], dtype=np.float32)
    speech_buffer = np.array([], dtype=np.float32)
    in_speech = False

    with sd.InputStream(
        device=mic_device, channels=1, samplerate=SAMPLE_RATE,
        callback=callback, blocksize=int(SAMPLE_RATE * 0.1),
    ):
        print(f"[side {side}] Listening on device {mic_device}...")
        while True:
            raw = audio_q.get().flatten().astype(np.float32)
            chunk_16k = resampy.resample(raw, SAMPLE_RATE, 16000)
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
                with _infer_lock:
                    segments, info = whisper_model.transcribe(audio_16k, beam_size=5)
                transcript = " ".join(s.text for s in segments).strip()
                if not transcript:
                    continue

                detected = info.language
                src_nllb = NLLB_LANG_MAP.get(detected, "fra_Latn")
                tgt_lang = TARGET_LANG[side]  # read dynamically — may change via UI
                translation = translate(transcript, src_nllb, tgt_lang)
                elapsed = int((time.time() - t0) * 1000)

                print(f"[side {side}] {elapsed}ms [{detected}→{tgt_lang}] {transcript!r} → {translation!r}")
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
      background: #0a0a0a;
      color: #f0f0f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    .side {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      padding: 48px 40px;
      gap: 20px;
    }
    .side-a { border-right: 1px solid #222; }
    .side-label {
      font-size: 11px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #444;
    }
    .lang-select {
      background: #111;
      color: #ccc;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 15px;
      cursor: pointer;
      outline: none;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%23666' d='M6 8L0 0h12z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      padding-right: 32px;
      transition: border-color 0.2s;
    }
    .lang-select:hover, .lang-select:focus { border-color: #555; color: #fff; }
    .lang-indicator {
      font-size: 12px;
      color: #444;
      min-height: 18px;
    }
    .translated {
      font-size: clamp(28px, 4vw, 56px);
      font-weight: 300;
      text-align: center;
      line-height: 1.4;
      min-height: 80px;
      max-width: 520px;
      word-break: break-word;
    }
    .original {
      font-size: 16px;
      color: #555;
      text-align: center;
      max-width: 480px;
      min-height: 24px;
      font-style: italic;
    }
    .latency { font-size: 11px; color: #2a2a2a; }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #1a1a1a;
      transition: background 0.3s;
    }
    .dot.active { background: #22c55e; }
  </style>
</head>
<body>
  <div class="side side-a">
    <div class="side-label">Side A</div>
    <select class="lang-select" id="selectA" onchange="setLang('A', this.value)">
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
    <div class="lang-indicator" id="langA"></div>
    <div class="translated" id="textA">—</div>
    <div class="original" id="origA"></div>
    <div class="latency" id="msA"></div>
    <div class="dot" id="dotA"></div>
  </div>
  <div class="side side-b">
    <div class="side-label">Side B</div>
    <select class="lang-select" id="selectB" onchange="setLang('B', this.value)">
      <option value="fra_Latn" selected>French</option>
      <option value="eng_Latn">English</option>
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
    <div class="lang-indicator" id="langB"></div>
    <div class="translated" id="textB">—</div>
    <div class="original" id="origB"></div>
    <div class="latency" id="msB"></div>
    <div class="dot" id="dotB"></div>
  </div>
  <script>
    function setLang(side, code) {
      fetch('/lang/' + side + '?code=' + code, { method: 'POST' });
    }

    function connect(side) {
      const text = document.getElementById('text' + side);
      const orig = document.getElementById('orig' + side);
      const ms   = document.getElementById('ms' + side);
      const lang = document.getElementById('lang' + side);
      const dot  = document.getElementById('dot' + side);

      const ws = new WebSocket('ws://' + location.host + '/ws/' + side);
      ws.onmessage = e => {
        const d = JSON.parse(e.data);
        text.textContent = d.translated;
        orig.textContent = d.original;
        ms.textContent   = d.ms + 'ms';
        lang.textContent = d.src_lang + ' → ' + d.tgt_lang;
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/lang/{side}")
async def set_language(side: str, code: str):
    if side not in ("A", "B"):
        raise HTTPException(status_code=404)
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
    try:
        while True:
            await websocket.receive_text()  # keep connection alive, client sends nothing
    except (WebSocketDisconnect, Exception):
        ws_clients[side].discard(websocket)


async def broadcast_worker(side: str):
    q = broadcast_queues[side]
    while True:
        msg = await q.get()
        payload = json.dumps(msg)
        dead: Set[WebSocket] = set()
        for ws in ws_clients[side].copy():
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        ws_clients[side] -= dead


@app.on_event("startup")
async def startup():
    global event_loop
    event_loop = asyncio.get_event_loop()
    broadcast_queues["A"] = asyncio.Queue()
    broadcast_queues["B"] = asyncio.Queue()
    asyncio.create_task(broadcast_worker("A"))
    asyncio.create_task(broadcast_worker("B"))
    threading.Thread(target=pipeline_worker, args=("A", DEVICE_A), daemon=True).start()
    threading.Thread(target=pipeline_worker, args=("B", DEVICE_B), daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

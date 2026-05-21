import asyncio
import csv
import io
import json
import os
import re
import sys
import threading
import time

import numpy as np
import resampy
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import state
from cache import add_entry, vocab_hints
from config import FONTS_DIR, LANG_NAMES, NLLB_LANG_MAP, TARGET_LANG
from html import ADMIN_HTML, HTML, make_single_side_html
from inference import _run_audio_inference

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@router.get("/side/{side}", response_class=HTMLResponse)
async def side_view(side: str):
    if side not in ("A", "B"):
        raise HTTPException(status_code=404)
    return make_single_side_html(side)


@router.get("/admin", response_class=HTMLResponse)
async def admin():
    return ADMIN_HTML


@router.websocket("/ws/admin")
async def ws_admin(websocket: WebSocket):
    await websocket.accept()
    state.admin_clients.add(websocket)
    for entry in state.admin_log:
        try:
            await websocket.send_text(json.dumps(entry))
        except Exception:
            break
    try:
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception):
        state.admin_clients.discard(websocket)


@router.post("/restart")
async def restart_server():
    def _do_restart():
        time.sleep(0.6)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do_restart, daemon=False).start()
    return JSONResponse({"status": "restarting"})


@router.get("/config/display")
async def get_display_config():
    return JSONResponse(state.display_config)


@router.post("/config/display")
async def set_display_config(body: dict):
    for k, v in body.items():
        if k == "font_size":
            state.display_config[k] = max(10, min(120, int(v)))
        elif k == "max_phrases":
            state.display_config[k] = max(1, min(20, int(v)))
            state.MAX_RECENT = state.display_config[k]
        elif k in {"bg_color", "text_color", "font_family"}:
            state.display_config[k] = str(v)
    await state._broadcast_config_ws()
    return JSONResponse(state.display_config)


@router.post("/upload/font")
async def upload_font(file: UploadFile = File(...)):
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename or "font")
    path = os.path.join(FONTS_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    if name not in state.display_config["custom_fonts"]:
        state.display_config["custom_fonts"].append(name)
    font_name = os.path.splitext(name)[0].replace("_", " ")
    await state._broadcast_config_ws()
    return JSONResponse({"filename": name, "url": f"/fonts/{name}", "font_name": font_name})


@router.websocket("/ws/config")
async def ws_config(websocket: WebSocket):
    await websocket.accept()
    state.config_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"kind": "config_update", **state.display_config}))
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception):
        state.config_clients.discard(websocket)


@router.post("/upload/glossary")
async def upload_glossary(file: UploadFile = File(...), src_lang: str = "fr", tgt_lang: str = "eng_Latn"):
    """Import a CSV/TSV glossary into the translation cache.

    2-column format : source, translation
    4-column format : source, src_lang_whisper, tgt_lang_nllb, translation
    Lines starting with # are comments. First row may be a header.
    """
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    filename = file.filename or ""

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
                continue

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

        add_entry(src_text, src_nllb, tgt_nllb, translation)

        if src_text not in vocab_hints:
            vocab_hints.append(src_text)

        if len(samples) < 5:
            samples.append({"source": src_text, "translation": translation})
        added += 1

    state.log_event("glossary", added=added, skipped=skipped, file=filename)
    return JSONResponse({"added": added, "skipped": skipped, "sample": samples})


@router.post("/upload/{side}")
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


@router.post("/lang/{side}")
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


@router.websocket("/ws/{side}")
async def ws_endpoint(websocket: WebSocket, side: str):
    if side not in ("A", "B"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    state.ws_clients[side].add(websocket)
    for msg in state.recent_messages[side]:
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            break
    try:
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception):
        state.ws_clients[side].discard(websocket)

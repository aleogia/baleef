"""baleef — real-time translation of system audio loopback (dual-sided kiosk)."""
import asyncio
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import state
from cache import _cache_saver, _load_cache, _save_cache
from config import FONTS_DIR, LOOPBACK_SAMPLERATE
from pipeline import find_loopback_device, pipeline_worker
from routes import router


@asynccontextmanager
async def lifespan(app_: FastAPI):
    for cmd in ["xset s off", "xset -dpms", "xset s noblank"]:
        subprocess.run(cmd.split(), check=False)
    state.event_loop = asyncio.get_event_loop()
    state.broadcast_queues["A"] = asyncio.Queue()
    asyncio.create_task(state.broadcast_worker("A"))
    _load_cache()
    threading.Thread(target=_cache_saver, daemon=True).start()
    try:
        device = find_loopback_device()
        threading.Thread(target=pipeline_worker, args=("A", device, LOOPBACK_SAMPLERATE, 2), daemon=True).start()
    except RuntimeError as e:
        print(f"[init] WARNING: loopback pipeline disabled — {e}")
    yield
    _save_cache()


app = FastAPI(lifespan=lifespan)
os.makedirs(FONTS_DIR, exist_ok=True)
app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

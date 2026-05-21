import asyncio
import json
import time
from typing import Dict, Set

from fastapi import WebSocket

admin_log: list = []
admin_clients: Set[WebSocket] = set()
MAX_LOG = 200

ws_clients: Dict[str, Set[WebSocket]] = {"A": set(), "B": set()}
broadcast_queues: Dict[str, asyncio.Queue] = {}
recent_messages: Dict[str, list] = {"A": [], "B": []}
event_loop: asyncio.AbstractEventLoop | None = None
MAX_RECENT = 4

display_config: Dict = {
    "bg_color": "#000000",
    "text_color": "#ffffff",
    "font_size": 32,
    "max_phrases": 4,
    "font_family": "'Segoe UI', system-ui, sans-serif",
    "custom_fonts": [],
}
config_clients: Set[WebSocket] = set()


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

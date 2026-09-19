"""
SQLite-backed response cache, keyed by a hash of (model, system, prompt,
tools). Makes reruns free: if you re-run the same case against the same
model after a crash, a prompt tweak on a *different* case, or just to
regenerate a report, you don't pay for it twice. This is the single
biggest practical cost-control mechanism in the harness.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from harness.core.models import ProviderResponse


def _cache_key(model: str, system: str | None, prompt: str, tools: list | None) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "tools": tools},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def get(self, model: str, system: str | None, prompt: str, tools: list | None) -> ProviderResponse | None:
        key = _cache_key(model, system, prompt, tools)
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return ProviderResponse(**data)

    def put(self, model: str, system: str | None, prompt: str, tools: list | None, response: ProviderResponse) -> None:
        key = _cache_key(model, system, prompt, tools)
        # `raw` holds the SDK's native response object and isn't JSON-safe/needed once cached
        cacheable = asdict(response)
        cacheable["raw"] = None
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, model, response_json) VALUES (?, ?, ?)",
                (key, model, json.dumps(cacheable)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

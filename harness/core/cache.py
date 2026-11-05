"""
SQLite-backed response cache, keyed by a hash of (model, system, prompt,
tools, variant). Makes reruns free: if you re-run the same case against
the same model after a crash, a prompt tweak on a *different* case, or just
to regenerate a report, you don't pay for it twice. This is the single
biggest practical cost-control mechanism in the harness.

`variant` is an opaque string folding in everything else that changes what
the model would say: the sampling settings (temperature, max tokens) and
the repeat-sample index. Without it, a cached temperature=0 answer would be
served for a temperature=0.8 request, and --repeats N would replay the same
cached sample N times instead of drawing N.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, fields
from pathlib import Path

from harness.core.models import ProviderResponse

_RESPONSE_FIELDS = {f.name for f in fields(ProviderResponse)}


def _cache_key(model: str, system: str | None, prompt: str, tools: list | None, variant: str = "") -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "tools": tools, "variant": variant},
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

    def get(
        self, model: str, system: str | None, prompt: str, tools: list | None, variant: str = "",
    ) -> ProviderResponse | None:
        key = _cache_key(model, system, prompt, tools, variant)
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM response_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        # Tolerate rows written by an older/newer version of ProviderResponse.
        return ProviderResponse(**{k: v for k, v in data.items() if k in _RESPONSE_FIELDS})

    def put(
        self, model: str, system: str | None, prompt: str, tools: list | None,
        response: ProviderResponse, variant: str = "",
    ) -> None:
        key = _cache_key(model, system, prompt, tools, variant)
        cacheable = asdict(response)
        # `raw` is normally either None (single-turn generate()) or a
        # JSON-safe list of content-block dicts (generate_messages(), so a
        # tool-calling loop can replay a cached turn correctly). Anything
        # else (e.g. a raw SDK object slipping through) isn't cacheable --
        # drop it rather than let json.dumps blow up the whole write.
        try:
            json.dumps(cacheable)
        except TypeError:
            cacheable["raw"] = None
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, model, response_json) VALUES (?, ?, ?)",
                (key, model, json.dumps(cacheable)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

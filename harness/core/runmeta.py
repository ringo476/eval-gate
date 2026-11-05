"""Provenance for a run: which code, which dataset, which settings.

Without this, a regression in the history table is just a number going
down -- nothing says whether the prompt, the dataset, the judge, or the
sampling settings changed. Recording it makes a regression attributable.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

from harness import __version__


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if sha else None
    return {
        "git_sha": sha,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") if sha else None,
        "git_dirty": bool(status) if status is not None else None,
    }


def collect_meta(**run_settings: Any) -> dict[str, Any]:
    """Run settings (provider, judge, temperature, repeats, dataset info, ...)
    plus environment provenance."""
    meta: dict[str, Any] = {
        "harness_version": __version__,
        "python": platform.python_version(),
        **git_info(),
        **run_settings,
    }
    return meta

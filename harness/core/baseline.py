"""
Blessed baselines: the committed record of "what good looks like".

Why this exists
---------------
Diffing against "the previous run in runs.db" has three failure modes:
  * CI has no runs.db (every runner starts empty), so nothing is ever compared;
  * the previous run may be a --smoke subset, or a degraded run, so the bar
    silently drifts down ("ratchets") instead of holding;
  * the baseline isn't reviewable -- nobody approved it.

A baseline file is a small, deterministic JSON snapshot of a known-good full
run, stored in the repo under baselines/. It survives CI, it only changes
when someone deliberately runs `harness baseline set` and commits the
result, and the change shows up in code review as a normal diff.

One file per (task, model): baselines/<task>__<model>.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.core.store import RunStore, now_iso

BASELINE_SCHEMA = 1
_KEPT_META = (
    "git_sha", "dataset_path", "dataset_sha256", "provider", "judge_provider", "judge_model",
    "temperature", "repeats", "harness_version",
)


class BaselineError(ValueError):
    pass


@dataclass
class Baseline:
    task: str
    model: str
    created_at: str
    source_run_id: str
    summary: dict[str, Any]
    cases: dict[str, dict[str, Any]]  # case_id -> {"pass_rate", "n_samples", "score"}
    meta: dict[str, Any] = field(default_factory=dict)
    path: str | None = None


def slugify(text: str) -> str:
    """Filesystem-safe: model names like 'llama3.1:8b' contain characters
    (':') that are illegal in Windows filenames."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in text).strip("_")
    return out or "unnamed"


def baseline_path(baselines_dir: str | Path, task: str, model: str) -> Path:
    return Path(baselines_dir) / f"{slugify(task)}__{slugify(model)}.json"


def baseline_from_store(store: RunStore, run_id: str, allow_errors: bool = False) -> Baseline:
    run = store.get_run(run_id)
    if run is None:
        raise BaselineError(f"run '{run_id}' not found")
    if run["is_aborted"]:
        raise BaselineError(f"run '{run_id}' was aborted (e.g. budget cap) and cannot be a baseline")
    if run["is_partial"]:
        raise BaselineError(
            f"run '{run_id}' is a partial (--smoke) run covering {run['case_count']} cases; a baseline must come "
            "from a full run, otherwise it can't vouch for the cases it skipped"
        )
    if run["error_count"] and not allow_errors:
        raise BaselineError(
            f"run '{run_id}' had {run['error_count']} errored sample(s); fix them or pass --force to bless it anyway"
        )
    cases = store.case_results_for_run(run_id)
    if not cases:
        raise BaselineError(f"run '{run_id}' has no case results")
    meta = {k: run["meta"][k] for k in _KEPT_META if k in run["meta"]}
    return Baseline(
        task=run["task"],
        model=run["model"],
        created_at=now_iso(),
        source_run_id=run_id,
        summary={
            "mean_score": round(run["mean_score"], 4),
            "pass_rate": round(run["pass_rate"], 4),
            "n_cases": run["case_count"],
            "repeats": run["repeats"],
        },
        cases={
            cid: {"pass_rate": round(v["pass_rate"], 4), "n_samples": v["n_samples"], "score": round(v["score"], 4)}
            for cid, v in sorted(cases.items())
        },
        meta=meta,
    )


def write_baseline(baseline: Baseline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BASELINE_SCHEMA,
        "task": baseline.task,
        "model": baseline.model,
        "created_at": baseline.created_at,
        "source_run_id": baseline.source_run_id,
        "meta": baseline.meta,
        "summary": baseline.summary,
        "cases": baseline.cases,
    }
    # sorted keys + trailing newline => stable, reviewable git diffs
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_baseline(path: str | Path) -> Baseline:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise BaselineError(f"baseline file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise BaselineError(f"{path}: invalid JSON ({e})") from e

    if data.get("schema") != BASELINE_SCHEMA:
        raise BaselineError(f"{path}: unsupported baseline schema {data.get('schema')!r} (expected {BASELINE_SCHEMA})")
    for key in ("task", "model", "source_run_id", "summary", "cases"):
        if key not in data:
            raise BaselineError(f"{path}: missing required field '{key}'")
    for cid, entry in data["cases"].items():
        if "pass_rate" not in entry:
            raise BaselineError(f"{path}: case '{cid}' has no pass_rate")

    return Baseline(
        task=data["task"], model=data["model"], created_at=data.get("created_at", ""),
        source_run_id=data["source_run_id"], summary=data["summary"], cases=data["cases"],
        meta=data.get("meta", {}), path=str(path),
    )


def find_baseline(baselines_dir: str | Path, task: str, model: str) -> Baseline | None:
    path = baseline_path(baselines_dir, task, model)
    return load_baseline(path) if path.exists() else None

"""Loading and validating golden JSONL datasets."""
from __future__ import annotations

import json
from pathlib import Path

from harness.core.models import EvalCase


def load_jsonl(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e

            for required in ("id", "task", "input", "expected"):
                if required not in row:
                    raise ValueError(f"{path}:{line_no}: missing required field '{required}'")

            if row["id"] in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate case id '{row['id']}'")
            seen_ids.add(row["id"])

            cases.append(EvalCase(
                id=row["id"],
                task=row["task"],
                input=row["input"],
                expected=row["expected"],
                tags=row.get("tags", []),
                human_label=row.get("human_label"),
            ))

    if not cases:
        raise ValueError(f"{path}: dataset is empty")

    return cases


def filter_smoke(cases: list[EvalCase], n: int = 10) -> list[EvalCase]:
    """Fast/cheap subset for iteration: pinned regression cases always
    included, then filled up to n with the rest, in file order (stable,
    not random -- so a smoke run is reproducible run to run)."""
    pinned = [c for c in cases if c.is_pinned]
    rest = [c for c in cases if not c.is_pinned]
    return (pinned + rest)[:max(n, len(pinned))]

"""Loading and validating golden JSONL datasets."""
from __future__ import annotations

import json
from pathlib import Path

from harness.core.models import EvalCase


def load_jsonl(path: str | Path, expected_task: str | None = None) -> list[EvalCase]:
    """Load and strictly validate a golden dataset.

    expected_task: if given, every row's `task` must equal it. Catches a
    case copy-pasted into the wrong file, or `--dataset` pointed at another
    task's data, before it silently runs through the wrong adapter.
    """
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
            where = f"{path}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON ({e})") from e
            if not isinstance(row, dict):
                raise ValueError(f"{where}: each line must be a JSON object")

            for required in ("id", "task", "input", "expected"):
                if required not in row:
                    raise ValueError(f"{where}: missing required field '{required}'")

            if not isinstance(row["id"], str) or not row["id"].strip():
                raise ValueError(f"{where}: 'id' must be a non-empty string")
            if not isinstance(row["input"], dict):
                raise ValueError(f"{where}: 'input' must be an object")
            if not isinstance(row["expected"], dict):
                raise ValueError(f"{where}: 'expected' must be an object")
            skip = row["expected"].get("skip_scorers", [])
            if not isinstance(skip, list) or not all(isinstance(s, str) for s in skip):
                raise ValueError(f"{where}: 'expected.skip_scorers' must be a list of scorer names")
            for entry in row["expected"].get("answer_contains", []):
                alternatives = entry if isinstance(entry, list) else [entry]
                if not alternatives or not all(isinstance(a, str) and a.strip() for a in alternatives):
                    raise ValueError(
                        f"{where}: 'expected.answer_contains' entries must be strings or lists of alternative strings"
                    )
            tags = row.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                raise ValueError(f"{where}: 'tags' must be a list of strings")
            label = row.get("human_label")
            if label is not None and (
                isinstance(label, bool) or not isinstance(label, (int, float)) or not 0.0 <= label <= 1.0
            ):
                raise ValueError(f"{where}: 'human_label' must be a number between 0 and 1")
            if expected_task is not None and row["task"] != expected_task:
                raise ValueError(
                    f"{where}: case '{row['id']}' has task '{row['task']}', expected '{expected_task}'"
                )

            if row["id"] in seen_ids:
                raise ValueError(f"{where}: duplicate case id '{row['id']}'")
            seen_ids.add(row["id"])

            cases.append(EvalCase(
                id=row["id"],
                task=row["task"],
                input=row["input"],
                expected=row["expected"],
                tags=tags,
                human_label=label,
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

"""
Regression detection: compare a run against a baseline for the same
(task, model). This is the actual point of the harness -- an
average-score-went-up-by-2% summary hides a prompt change that broke 3
specific cases while improving 5 others. Diffing surfaces exactly which.

Which baseline? In order of preference:
  1. an explicit file (--baseline PATH),
  2. the committed, blessed baseline  baselines/<task>__<model>.json
     -- the only kind that works in CI, where there is no run history;
  3. fallback for local iteration: the previous *full* run in runs.db
     (never a --smoke subset or an aborted run).
The result records which kind was used, so a report can't quietly present
"vs. last run" as if it were "vs. the approved baseline".

What counts as a regression? Cases are compared by pass *rate*, not a single
pass/fail flip. With one sample per case (the default) the rate is 0 or 1 and
this is the familiar "was passing, now fails". With --repeats N it separates
real breakage from noise: a case dropping from 5/5 to 4/5 is `degraded`
(reported, not gating by default); 5/5 to 1/5 is a `regression`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.core.baseline import find_baseline, load_baseline
from harness.core.store import RunStore

DEFAULT_REGRESSION_THRESHOLD = 0.5


@dataclass
class DiffResult:
    baseline_run_id: str | None
    current_run_id: str
    regressions: list[str]     # pass rate dropped by >= threshold
    improvements: list[str]    # pass rate rose by >= threshold
    unchanged_pass: int
    unchanged_fail: int
    new_cases: list[str]       # in current but not in baseline
    degraded: list[str] = field(default_factory=list)       # dropped, but by less than the threshold
    missing_cases: list[str] = field(default_factory=list)  # in baseline but absent from current
    baseline_kind: str = "none"                             # "blessed" | "previous_run" | "none"
    baseline_label: str | None = None                       # file path or run id, for display
    baseline_mean_score: float | None = None
    baseline_pass_rate: float | None = None
    threshold: float = DEFAULT_REGRESSION_THRESHOLD


@dataclass
class BaselineRef:
    kind: str
    label: str
    run_id: str | None
    cases: dict[str, dict[str, Any]]
    mean_score: float | None = None
    pass_rate: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def resolve_baseline(
    store: RunStore,
    task: str,
    model: str,
    current_run_id: str,
    baselines_dir: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> BaselineRef | None:
    blessed = None
    if baseline_path is not None:
        blessed = load_baseline(baseline_path)  # explicit path: missing/invalid is an error, not a fallback
    elif baselines_dir is not None:
        blessed = find_baseline(baselines_dir, task, model)

    if blessed is not None:
        return BaselineRef(
            kind="blessed", label=blessed.path or blessed.source_run_id, run_id=blessed.source_run_id,
            cases=blessed.cases, mean_score=blessed.summary.get("mean_score"),
            pass_rate=blessed.summary.get("pass_rate"), meta=blessed.meta,
        )

    prev = store.latest_run_id(task, model, before_run_id=current_run_id, full_only=True)
    if prev is None:
        return None
    run = store.get_run(prev) or {}
    return BaselineRef(
        kind="previous_run", label=prev, run_id=prev, cases=store.case_results_for_run(prev),
        mean_score=run.get("mean_score"), pass_rate=run.get("pass_rate"), meta=run.get("meta", {}),
    )


def diff_cases(
    ref: BaselineRef | None,
    current_run_id: str,
    current: dict[str, dict[str, Any]],
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> DiffResult:
    if ref is None:
        return DiffResult(
            baseline_run_id=None, current_run_id=current_run_id,
            regressions=[], improvements=[],
            unchanged_pass=sum(1 for v in current.values() if v["passed"]),
            unchanged_fail=sum(1 for v in current.values() if not v["passed"]),
            new_cases=sorted(current), threshold=threshold,
        )

    regressions, improvements, degraded, new_cases = [], [], [], []
    unchanged_pass = unchanged_fail = 0

    for case_id, cur in current.items():
        base = ref.cases.get(case_id)
        if base is None:
            new_cases.append(case_id)
            continue
        drop = base["pass_rate"] - cur["pass_rate"]
        if drop >= threshold:
            regressions.append(case_id)
        elif -drop >= threshold:
            improvements.append(case_id)
        else:
            if drop > 0:
                degraded.append(case_id)
            if cur["passed"]:
                unchanged_pass += 1
            else:
                unchanged_fail += 1

    return DiffResult(
        baseline_run_id=ref.run_id,
        current_run_id=current_run_id,
        regressions=sorted(regressions),
        improvements=sorted(improvements),
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        new_cases=sorted(new_cases),
        degraded=sorted(degraded),
        missing_cases=sorted(set(ref.cases) - set(current)),
        baseline_kind=ref.kind,
        baseline_label=ref.label,
        baseline_mean_score=ref.mean_score,
        baseline_pass_rate=ref.pass_rate,
        threshold=threshold,
    )


def diff_runs(
    store: RunStore,
    task: str,
    model: str,
    current_run_id: str,
    *,
    baselines_dir: str | Path | None = None,
    baseline_path: str | Path | None = None,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> DiffResult:
    ref = resolve_baseline(store, task, model, current_run_id, baselines_dir, baseline_path)
    return diff_cases(ref, current_run_id, store.case_results_for_run(current_run_id), threshold)

"""
Regression detection: compare a run against the run before it for the
same (task, model). This is the actual point of the harness -- an
average-score-went-up-by-2% summary hides a prompt change that broke 3
specific cases while improving 5 others. Diffing surfaces exactly which.
"""
from __future__ import annotations

from dataclasses import dataclass

from harness.core.store import RunStore


@dataclass
class DiffResult:
    baseline_run_id: str | None
    current_run_id: str
    regressions: list[str]     # case ids that passed before, fail now
    improvements: list[str]    # case ids that failed before, pass now
    unchanged_pass: int
    unchanged_fail: int
    new_cases: list[str]       # in current but not in baseline


def diff_runs(store: RunStore, task: str, model: str, current_run_id: str) -> DiffResult:
    baseline_run_id = store.latest_run_id(task, model, before_run_id=current_run_id)

    current = store.case_results_for_run(current_run_id)
    if baseline_run_id is None:
        return DiffResult(
            baseline_run_id=None,
            current_run_id=current_run_id,
            regressions=[],
            improvements=[],
            unchanged_pass=sum(1 for v in current.values() if v["passed"]),
            unchanged_fail=sum(1 for v in current.values() if not v["passed"]),
            new_cases=list(current.keys()),
        )

    baseline = store.case_results_for_run(baseline_run_id)

    regressions, improvements, new_cases = [], [], []
    unchanged_pass = unchanged_fail = 0

    for case_id, cur in current.items():
        if case_id not in baseline:
            new_cases.append(case_id)
            continue
        was_passed = baseline[case_id]["passed"]
        now_passed = cur["passed"]
        if was_passed and not now_passed:
            regressions.append(case_id)
        elif not was_passed and now_passed:
            improvements.append(case_id)
        elif now_passed:
            unchanged_pass += 1
        else:
            unchanged_fail += 1

    return DiffResult(
        baseline_run_id=baseline_run_id,
        current_run_id=current_run_id,
        regressions=sorted(regressions),
        improvements=sorted(improvements),
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        new_cases=sorted(new_cases),
    )

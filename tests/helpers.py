"""Shared builders for tests."""
from __future__ import annotations

from harness.core.models import CaseResult, EvalCase, RunSummary, RunTrace, ScoreResult


def make_sample(case_id: str, passed: bool, sample_idx: int = 0, error: str | None = None) -> CaseResult:
    case = EvalCase(id=case_id, task="t", input={}, expected={})
    trace = RunTrace(case_id=case_id, output_text="x", error=error)
    scores = [] if error else [ScoreResult("dummy", score=1.0 if passed else 0.0, passed=passed)]
    return CaseResult(case=case, trace=trace, scores=scores, sample_idx=sample_idx)


def make_summary(
    run_id: str,
    task: str,
    model: str,
    timestamp: str,
    case_outcomes: dict[str, bool | list[bool]],
    meta: dict | None = None,
    aborted: bool = False,
) -> RunSummary:
    """case_outcomes: {case_id: passed} for one sample, or {case_id: [passed, ...]} for repeats."""
    results = []
    for case_id, outcome in case_outcomes.items():
        outcomes = outcome if isinstance(outcome, list) else [outcome]
        for i, passed in enumerate(outcomes):
            results.append(make_sample(case_id, passed, sample_idx=i))
    return RunSummary(run_id=run_id, task=task, model=model, timestamp=timestamp,
                      case_results=results, meta=meta or {}, aborted=aborted)

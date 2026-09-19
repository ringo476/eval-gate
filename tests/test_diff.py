from harness.core.diff import diff_runs
from harness.core.models import CaseResult, EvalCase, RunSummary, RunTrace, ScoreResult
from harness.core.store import RunStore


def make_summary(run_id, task, model, timestamp, case_scores: dict[str, bool]):
    """case_scores: {case_id: passed}"""
    results = []
    for case_id, passed in case_scores.items():
        case = EvalCase(id=case_id, task=task, input={}, expected={})
        trace = RunTrace(case_id=case_id, output_text="x")
        score = ScoreResult("dummy", score=1.0 if passed else 0.0, passed=passed)
        results.append(CaseResult(case=case, trace=trace, scores=[score]))
    return RunSummary(run_id=run_id, task=task, model=model, timestamp=timestamp, case_results=results)


def test_first_run_has_no_baseline(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    summary = make_summary("run1", "t", "m", "2025-01-01T00:00:00Z", {"a": True, "b": False})
    store.save(summary)

    diff = diff_runs(store, "t", "m", "run1")
    assert diff.baseline_run_id is None
    assert set(diff.new_cases) == {"a", "b"}
    store.close()


def test_detects_regression_and_improvement(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))

    baseline = make_summary("run1", "t", "m", "2025-01-01T00:00:00Z", {"a": True, "b": False, "c": True})
    store.save(baseline)

    current = make_summary("run2", "t", "m", "2025-01-02T00:00:00Z", {"a": False, "b": True, "c": True})
    store.save(current)

    diff = diff_runs(store, "t", "m", "run2")
    assert diff.baseline_run_id == "run1"
    assert diff.regressions == ["a"]      # was passing, now fails
    assert diff.improvements == ["b"]     # was failing, now passes
    assert diff.unchanged_pass == 1       # c stayed passing
    assert diff.unchanged_fail == 0
    store.close()


def test_new_case_not_counted_as_regression(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("run1", "t", "m", "2025-01-01T00:00:00Z", {"a": True}))
    store.save(make_summary("run2", "t", "m", "2025-01-02T00:00:00Z", {"a": True, "new_case": False}))

    diff = diff_runs(store, "t", "m", "run2")
    assert diff.new_cases == ["new_case"]
    assert diff.regressions == []
    store.close()

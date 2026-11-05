"""Baseline resolution, thresholds and missing-case handling in diffing."""
import pytest

from harness.core.baseline import BaselineError, baseline_from_store, baseline_path, write_baseline
from harness.core.diff import BaselineRef, diff_cases, diff_runs
from harness.core.store import RunStore
from tests.helpers import make_summary


def _ref(cases):
    return BaselineRef(kind="blessed", label="file.json", run_id="base", cases=cases)


def _cur(rates):
    return {cid: {"passed": r > 0.5, "pass_rate": r, "score": r, "n_samples": 5} for cid, r in rates.items()}


def test_blessed_baseline_beats_previous_run(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("good", "t", "m", "2026-01-01T00:00:00Z", {"a": True, "b": True}))
    write_baseline(baseline_from_store(store, "good"), baseline_path(tmp_path / "bl", "t", "m"))
    # a later degraded run becomes "the previous run"...
    store.save(make_summary("bad", "t", "m", "2026-01-02T00:00:00Z", {"a": False, "b": False}))
    # ...and the current run is as bad as that one
    store.save(make_summary("cur", "t", "m", "2026-01-03T00:00:00Z", {"a": False, "b": False}))

    blessed = diff_runs(store, "t", "m", "cur", baselines_dir=tmp_path / "bl")
    assert blessed.baseline_kind == "blessed"
    assert blessed.regressions == ["a", "b"]  # judged against the approved bar

    rolling = diff_runs(store, "t", "m", "cur")  # no baseline file: falls back to previous run
    assert rolling.baseline_kind == "previous_run"
    assert rolling.regressions == []  # the ratchet problem a blessed baseline prevents
    store.close()


def test_previous_run_fallback_ignores_smoke_runs(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("full", "t", "m", "2026-01-01T00:00:00Z", {"a": True, "b": True}))
    store.save(make_summary("smoke", "t", "m", "2026-01-02T00:00:00Z", {"a": False}, meta={"partial": True}))
    store.save(make_summary("cur", "t", "m", "2026-01-03T00:00:00Z", {"a": True, "b": False}))
    diff = diff_runs(store, "t", "m", "cur")
    assert diff.baseline_run_id == "full"
    assert diff.regressions == ["b"]
    store.close()


def test_explicit_baseline_path_must_exist(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    store.save(make_summary("cur", "t", "m", "2026-01-03T00:00:00Z", {"a": True}))
    with pytest.raises(BaselineError):
        diff_runs(store, "t", "m", "cur", baseline_path=tmp_path / "nope.json")
    store.close()


def test_threshold_separates_regression_from_degraded():
    ref = _ref({"x": {"pass_rate": 1.0}, "y": {"pass_rate": 1.0}, "z": {"pass_rate": 0.0}})
    diff = diff_cases(ref, "cur", _cur({"x": 0.8, "y": 0.2, "z": 0.8}), threshold=0.5)
    assert diff.degraded == ["x"]  # 5/5 -> 4/5: noise-sized, reported not gating
    assert diff.regressions == ["y"]  # 5/5 -> 1/5: real
    assert diff.improvements == ["z"]


def test_single_sample_flip_is_always_a_regression():
    ref = _ref({"a": {"pass_rate": 1.0}})
    assert diff_cases(ref, "cur", _cur({"a": 0.0})).regressions == ["a"]


def test_missing_and_new_cases():
    ref = _ref({"kept": {"pass_rate": 1.0}, "dropped": {"pass_rate": 0.0}})
    diff = diff_cases(ref, "cur", _cur({"kept": 1.0, "added": 0.0}))
    assert diff.missing_cases == ["dropped"]
    assert diff.new_cases == ["added"]
    assert diff.regressions == []


def test_no_baseline_means_everything_is_new():
    diff = diff_cases(None, "cur", _cur({"a": 1.0}))
    assert diff.baseline_kind == "none" and diff.new_cases == ["a"]

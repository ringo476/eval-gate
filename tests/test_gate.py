from harness.core.diff import BaselineRef, diff_cases
from harness.core.gate import evaluate_gate

GOOD_RUN = {"is_aborted": False, "is_partial": False, "error_count": 0, "case_count": 3,
            "meta": {"dataset_sha256": "abc"}}


def results(rates):
    return {cid: {"passed": r > 0.5, "pass_rate": r, "score": r, "n_samples": 1} for cid, r in rates.items()}


def make_diff(current, baseline_rates=None, kind="blessed"):
    ref = None
    if baseline_rates is not None:
        ref = BaselineRef(kind=kind, label="b.json", run_id="base",
                          cases={c: {"pass_rate": r} for c, r in baseline_rates.items()})
    return diff_cases(ref, "cur", current)


def gate(rates, baseline_rates=None, *, run=None, pinned=(), kind="blessed", **kwargs):
    current = results(rates)
    kwargs.setdefault("min_score", 0.0)
    kwargs.setdefault("current_dataset_sha", "abc")
    return evaluate_gate(run=run or GOOD_RUN, case_results=current, pinned_ids=set(pinned),
                         diff=make_diff(current, baseline_rates, kind), **kwargs)


def test_clean_run_passes():
    v = gate({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 1}, min_score=0.9)
    assert v.ok and not v.failures


def test_mean_score_bar():
    v = gate({"a": 1, "b": 0, "c": 0}, {"a": 1, "b": 0, "c": 0}, min_score=0.75)
    assert not v.ok and any("mean_score" in f for f in v.failures)


def test_min_pass_rate():
    v = gate({"a": 1, "b": 0}, {"a": 1, "b": 0}, min_pass_rate=0.9)
    assert any("pass_rate" in f for f in v.failures)


def test_pinned_case_failing_or_flaky_fails_the_gate():
    assert not gate({"a": 0, "b": 1}, {"a": 0, "b": 1}, pinned={"a"}).ok
    v = gate({"a": 0.5, "b": 1}, {"a": 0.5, "b": 1}, pinned={"a"})  # a flaky pinned case is not good enough
    assert any("pinned" in f for f in v.failures)


def test_unevaluated_pinned_case_fails():
    v = gate({"b": 1}, {"b": 1}, pinned={"a"})
    assert any("not evaluated" in f for f in v.failures)


def test_regression_fails():
    v = gate({"a": 0, "b": 1}, {"a": 1, "b": 1})
    assert not v.ok and any("regressed" in f and "a" in f for f in v.failures)


def test_degraded_warns_unless_strict():
    base = {"a": 1.0}
    relaxed = gate({"a": 0.8}, base, run={**GOOD_RUN, "case_count": 1})
    assert relaxed.ok and any("flakier" in w for w in relaxed.warnings)
    assert not gate({"a": 0.8}, base, fail_on_degraded=True).ok


def test_deleting_a_failing_case_cannot_turn_the_gate_green():
    v = gate({"a": 1}, {"a": 1, "hard-case": 0})
    assert not v.ok and any("missing" in f and "hard-case" in f for f in v.failures)


def test_no_baseline():
    assert gate({"a": 1}).ok  # warns only
    assert gate({"a": 1}).warnings
    v = gate({"a": 1}, require_baseline=True)
    assert not v.ok and any("baseline" in f for f in v.failures)


def test_require_baseline_rejects_a_mere_previous_run():
    v = gate({"a": 1}, {"a": 1}, kind="previous_run", require_baseline=True)
    assert not v.ok and any("committed baseline" in f for f in v.failures)


def test_partial_run_is_refused_unless_allowed():
    partial = {**GOOD_RUN, "is_partial": True, "case_count": 2}
    v = gate({"a": 1}, {"a": 1, "b": 1}, run=partial)
    assert not v.ok and any("--smoke" in f for f in v.failures)
    assert not any("missing" in f for f in v.failures)  # a smoke run legitimately lacks cases
    assert gate({"a": 1}, {"a": 1, "b": 1}, run=partial, allow_partial=True).ok


def test_aborted_run_fails():
    v = gate({"a": 1}, {"a": 1}, run={**GOOD_RUN, "is_aborted": True})
    assert not v.ok and any("aborted" in f for f in v.failures)


def test_errors_fail_by_default_but_can_be_tolerated():
    erroring = {**GOOD_RUN, "error_count": 2}
    assert not gate({"a": 1}, {"a": 1}, run=erroring).ok
    assert gate({"a": 1}, {"a": 1}, run=erroring, max_errors=2).ok


def test_stale_dataset_fails():
    v = gate({"a": 1}, {"a": 1}, current_dataset_sha="different")
    assert not v.ok and any("stale" in f for f in v.failures)


def test_run_without_fingerprint_only_warns():
    v = gate({"a": 1}, {"a": 1}, run={**GOOD_RUN, "meta": {}})
    assert v.ok and any("fingerprint" in w for w in v.warnings)

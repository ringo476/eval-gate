from harness.core.models import CaseAggregate
from tests.helpers import make_sample, make_summary


def test_single_sample_semantics_unchanged():
    summary = make_summary("r", "t", "m", "ts", {"a": True, "b": False})
    assert summary.pass_rate == 0.5
    assert summary.repeats == 1
    assert summary.flaky_cases == []
    assert [c.status for c in summary.cases] == ["pass", "fail"]


def test_aggregate_status_and_majority():
    agg = CaseAggregate(case=make_sample("a", True).case,
                        samples=[make_sample("a", True, 0), make_sample("a", True, 1), make_sample("a", False, 2)])
    assert agg.n_passed == 2 and agg.n_samples == 3
    assert agg.passed is True          # strict majority
    assert agg.status == "flaky"
    assert abs(agg.pass_rate - 2 / 3) < 1e-9


def test_even_split_is_not_a_pass():
    agg = CaseAggregate(case=make_sample("a", True).case, samples=[make_sample("a", True, 0), make_sample("a", False, 1)])
    assert agg.passed is False         # 1 of 2 is not a strict majority
    assert agg.status == "flaky"


def test_run_summary_aggregates_repeats():
    summary = make_summary("r", "t", "m", "ts", {"a": [True, True, True], "b": [True, False, False], "c": [False] * 3})
    assert summary.repeats == 3
    assert [c.status for c in summary.cases] == ["pass", "flaky", "fail"]
    assert summary.flaky_cases == ["b"]
    assert abs(summary.pass_rate - 1 / 3) < 1e-9   # only "a" passes by majority


def test_errored_samples_are_counted_and_fail_the_case():
    summary = make_summary("r", "t", "m", "ts", {"a": True})
    summary.case_results.append(make_sample("b", False, error="boom"))
    assert summary.error_count == 1
    bad = next(c for c in summary.cases if c.case.id == "b")
    assert bad.passed is False and bad.mean_score == 0.0 and bad.error_count == 1


def test_representative_prefers_a_failing_sample():
    agg = CaseAggregate(case=make_sample("a", True).case,
                        samples=[make_sample("a", True, 0), make_sample("a", False, 1)])
    assert agg.representative.sample_idx == 1


def test_partial_flag_comes_from_meta():
    assert make_summary("r", "t", "m", "ts", {"a": True}, meta={"partial": True}).is_partial is True
    assert make_summary("r", "t", "m", "ts", {"a": True}).is_partial is False

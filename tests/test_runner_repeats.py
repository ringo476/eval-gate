import time

import pytest

from harness.adapters.base import Adapter
from harness.core.budget import BudgetExceededError
from harness.core.models import EvalCase, RunTrace
from harness.core.providers import MockProvider
from harness.core.runner import Runner
from harness.scorers.contains import ContainsScorer


def make_cases(n):
    return [EvalCase(id=f"c{i}", task="t", input={}, expected={"answer_contains": ["ok"]}) for i in range(n)]


class IndexedProvider(MockProvider):
    """Remembers which repeat-sample it was scoped to."""

    def __init__(self, idx=0):
        super().__init__()
        self.idx = idx

    def scoped(self, sample_idx):
        return IndexedProvider(sample_idx)


class PassOnEvenSamples(Adapter):
    task = "t"

    def run(self, case, provider):
        return RunTrace(case_id=case.id, output_text="ok" if provider.idx % 2 == 0 else "nope", model="m")


def test_repeats_draw_independent_samples_and_aggregate():
    runner = Runner(PassOnEvenSamples(), [ContainsScorer()], IndexedProvider(), max_workers=3, repeats=4)
    summary = runner.run(make_cases(2))

    assert len(summary.case_results) == 8
    assert summary.repeats == 4
    # ordered by (case, sample), not by thread completion
    assert [(cr.case.id, cr.sample_idx) for cr in summary.case_results] == [
        (f"c{c}", s) for c in range(2) for s in range(4)
    ]
    for agg in summary.cases:
        assert agg.n_passed == 2 and agg.status == "flaky" and agg.passed is False
    assert summary.flaky_cases == ["c0", "c1"]


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        Runner(PassOnEvenSamples(), [], IndexedProvider(), repeats=0)


def test_provider_without_scoped_still_works():
    class Plain:
        name = "plain"

    class A(Adapter):
        task = "t"

        def run(self, case, provider):
            return RunTrace(case_id=case.id, output_text="ok", model=provider.name)

    summary = Runner(A(), [ContainsScorer()], Plain(), repeats=2).run(make_cases(1))
    assert len(summary.case_results) == 2


def test_budget_exhaustion_aborts_the_run_instead_of_erroring_every_case():
    class BudgetAdapter(Adapter):
        task = "t"

        def run(self, case, provider):
            if case.id == "c2":
                raise BudgetExceededError("cap hit")
            time.sleep(0.05)
            return RunTrace(case_id=case.id, output_text="ok", model="m")

    summary = Runner(BudgetAdapter(), [ContainsScorer()], MockProvider(), max_workers=1).run(make_cases(8))

    assert summary.aborted is True
    assert "cap hit" in summary.abort_reason
    assert len(summary.case_results) < 8  # pending cases were cancelled, not run
    assert summary.error_count == 0  # and NOT recorded as 8 separate per-case errors


def test_budget_error_raised_by_a_scorer_also_aborts():
    class OkAdapter(Adapter):
        task = "t"

        def run(self, case, provider):
            return RunTrace(case_id=case.id, output_text="ok", model="m")

    class BudgetScorer:
        name = "budget"

        def score(self, case, trace):
            raise BudgetExceededError("judge over budget")

    summary = Runner(OkAdapter(), [BudgetScorer()], MockProvider(), max_workers=1).run(make_cases(3))
    assert summary.aborted is True and "judge over budget" in summary.abort_reason

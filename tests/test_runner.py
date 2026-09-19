from harness.adapters.base import Adapter
from harness.core.models import EvalCase, RunTrace
from harness.core.providers import MockProvider
from harness.core.runner import Runner
from harness.scorers.contains import ContainsScorer


class AlwaysSucceedsAdapter(Adapter):
    task = "fake_task"

    def run(self, case, provider):
        return RunTrace(case_id=case.id, output_text=f"answer containing {case.input['needle']}", model=provider.name)


class OneCaseAlwaysErrorsAdapter(Adapter):
    task = "fake_task"

    def run(self, case, provider):
        if case.id == "bad-case":
            return RunTrace(case_id=case.id, output_text="", model=provider.name, error="simulated failure")
        return RunTrace(case_id=case.id, output_text=f"answer containing {case.input['needle']}", model=provider.name)


def make_cases(n):
    return [
        EvalCase(id=f"c{i}", task="fake_task", input={"needle": "ok"}, expected={"answer_contains": ["ok"]})
        for i in range(n)
    ]


def test_runner_end_to_end_all_pass():
    cases = make_cases(5)
    runner = Runner(AlwaysSucceedsAdapter(), [ContainsScorer()], MockProvider(), max_workers=3)
    summary = runner.run(cases)

    assert len(summary.case_results) == 5
    assert summary.pass_rate == 1.0
    assert summary.error_count == 0
    # order must match input dataset order regardless of thread completion order
    assert [cr.case.id for cr in summary.case_results] == [c.id for c in cases]


def test_runner_isolates_one_bad_case():
    cases = make_cases(4)
    cases[2] = EvalCase(id="bad-case", task="fake_task", input={"needle": "ok"}, expected={"answer_contains": ["ok"]})

    runner = Runner(OneCaseAlwaysErrorsAdapter(), [ContainsScorer()], MockProvider(), max_workers=2)
    summary = runner.run(cases)

    assert len(summary.case_results) == 4
    assert summary.error_count == 1
    bad = next(cr for cr in summary.case_results if cr.case.id == "bad-case")
    assert bad.passed is False
    assert bad.trace.error == "simulated failure"
    # the other three cases must be unaffected by the one failure
    assert summary.pass_rate == 0.75


def test_runner_a_broken_scorer_does_not_kill_the_run():
    class ExplodingScorer:
        name = "boom"

        def score(self, case, trace):
            raise RuntimeError("scorer bug")

    cases = make_cases(2)
    runner = Runner(AlwaysSucceedsAdapter(), [ContainsScorer(), ExplodingScorer()], MockProvider())
    summary = runner.run(cases)

    assert len(summary.case_results) == 2
    for cr in summary.case_results:
        assert any(s.scorer_name == "boom" and "scorer_error" in s.details for s in cr.scores)

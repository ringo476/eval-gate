import json

import pytest

from harness.adapters.base import Adapter
from harness.core.dataset import load_jsonl
from harness.core.models import EvalCase, RunTrace
from harness.core.providers import MockProvider
from harness.core.runner import Runner
from harness.scorers.contains import ContainsScorer


class Fixed(Adapter):
    task = "t"

    def run(self, case, provider):
        return RunTrace(case_id=case.id, output_text="nothing relevant", model="m")


class AlwaysFails:
    name = "strict"

    def score(self, case, trace):
        from harness.core.models import ScoreResult
        return ScoreResult(self.name, 0.0, False)


def test_a_case_can_opt_out_of_a_scorer_that_does_not_apply():
    applies = EvalCase(id="a", task="t", input={}, expected={})
    skips = EvalCase(id="b", task="t", input={}, expected={"skip_scorers": ["strict"]})
    summary = Runner(Fixed(), [ContainsScorer(), AlwaysFails()], MockProvider()).run([applies, skips])

    by_id = {c.case.id: c for c in summary.cases}
    assert by_id["a"].passed is False
    assert by_id["b"].passed is True  # the strict scorer was not run for it
    assert [s.scorer_name for s in by_id["b"].representative.scores] == ["contains"]


def test_skipping_an_unknown_scorer_warns_instead_of_silently_doing_nothing(caplog):
    case = EvalCase(id="a", task="t", input={}, expected={"skip_scorers": ["typo_scorer"]})
    with caplog.at_level("WARNING", logger="harness.runner"):
        Runner(Fixed(), [ContainsScorer()], MockProvider()).run([case])
    assert "typo_scorer" in caplog.text


def test_dataset_rejects_malformed_skip_scorers(tmp_path):
    path = tmp_path / "d.jsonl"
    row = {"id": "a", "task": "t", "input": {}, "expected": {"skip_scorers": "embedding_similarity"}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="skip_scorers"):
        load_jsonl(path)


def test_shipped_behavior_description_cases_skip_embedding_similarity():
    agent = {c.id: c for c in load_jsonl("golden/tool_agent_qa.jsonl")}
    rag = {c.id: c for c in load_jsonl("golden/rag_fusion_qa.jsonl")}
    for case in (agent["ta-009"], agent["ta-010"], rag["rf-023"]):
        assert "embedding_similarity" in case.expected["skip_scorers"], case.id

import json

from harness.core.models import EvalCase, ProviderResponse, RunTrace
from harness.scorers.contains import ContainsScorer
from harness.scorers.retrieval import RetrievalHitRateScorer


def make_case(expected, input_=None):
    return EvalCase(id="c1", task="t", input=input_ or {"query": "q"}, expected=expected)


def make_trace(output_text="", retrieved_chunks=None):
    return RunTrace(case_id="c1", output_text=output_text, retrieved_chunks=retrieved_chunks or [])


class FakeJudgeProvider:
    """Returns a fixed, valid judge JSON response -- used to test the
    judge's parsing/scoring logic without any real model call."""
    name = "fake-judge"

    def __init__(self, score=8, reason="looks correct"):
        self._score = score
        self._reason = reason

    def generate(self, prompt, system=None, tools=None):
        return ProviderResponse(
            text=json.dumps({"score": self._score, "reason": self._reason}),
            input_tokens=50,
            output_tokens=10,
        )


class BrokenJudgeProvider:
    name = "broken-judge"

    def generate(self, prompt, system=None, tools=None):
        return ProviderResponse(text="not json at all", input_tokens=10, output_tokens=5)


def test_contains_scorer_passes_when_all_present():
    case = make_case({"answer_contains": ["hello", "world"]})
    trace = make_trace(output_text="Well, hello there world!")
    result = ContainsScorer().score(case, trace)
    assert result.passed is True
    assert result.score == 1.0


def test_contains_scorer_partial_miss():
    case = make_case({"answer_contains": ["hello", "world"]})
    trace = make_trace(output_text="hello only")
    result = ContainsScorer().score(case, trace)
    assert result.passed is False
    assert result.score == 0.5
    assert result.details["missing"] == ["world"]


def test_contains_scorer_no_requirement_passes():
    case = make_case({})
    trace = make_trace(output_text="anything")
    result = ContainsScorer().score(case, trace)
    assert result.passed is True
    assert result.score == 1.0


def test_retrieval_hit_rate_full_hit():
    case = make_case({"relevant_doc_ids": ["doc1", "doc2"]})
    trace = make_trace(retrieved_chunks=[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc9"}])
    result = RetrievalHitRateScorer().score(case, trace)
    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_hit_rate_partial_hit():
    case = make_case({"relevant_doc_ids": ["doc1", "doc2"]})
    trace = make_trace(retrieved_chunks=[{"source": "doc1"}])
    result = RetrievalHitRateScorer().score(case, trace)
    assert result.score == 0.5
    assert result.passed is False
    assert result.details["missed"] == ["doc2"]


def test_retrieval_hit_rate_empty_requirement_auto_passes():
    case = make_case({"relevant_doc_ids": []})
    trace = make_trace(retrieved_chunks=[])
    result = RetrievalHitRateScorer().score(case, trace)
    assert result.passed is True
    assert result.score == 1.0


def test_llm_judge_happy_path():
    from harness.scorers.llm_judge import LLMJudgeScorer
    case = make_case({"reference_answer": "the sky is blue"}, input_={"question": "what color is the sky?"})
    trace = make_trace(output_text="The sky appears blue.")
    scorer = LLMJudgeScorer(FakeJudgeProvider(score=8))
    result = scorer.score(case, trace)
    assert result.score == 0.8
    assert result.passed is True


def test_llm_judge_handles_invalid_json_gracefully():
    from harness.scorers.llm_judge import LLMJudgeScorer
    case = make_case({"reference_answer": "x"}, input_={"question": "q"})
    trace = make_trace(output_text="some answer")
    scorer = LLMJudgeScorer(BrokenJudgeProvider())
    result = scorer.score(case, trace)
    assert result.passed is False
    assert result.score == 0.0
    assert "error" in result.details


def test_llm_judge_empty_candidate_fails_fast():
    from harness.scorers.llm_judge import LLMJudgeScorer
    case = make_case({"reference_answer": "x"}, input_={"question": "q"})
    trace = make_trace(output_text="")
    scorer = LLMJudgeScorer(FakeJudgeProvider())
    result = scorer.score(case, trace)
    assert result.passed is False
    assert result.score == 0.0

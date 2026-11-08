"""
Retrieval hit-rate: did the fused ranking actually surface the document(s)
we know are relevant to the query? This is checked independently of
whatever the model says in its final answer -- a good answer built on the
wrong retrieved documents is a retrieval bug wearing a generation costume,
and this scorer is what catches that distinction instead of letting a
lucky-sounding LLM answer paper over it.

A case with an empty `relevant_doc_ids` list (e.g. an out-of-corpus
adversarial query) has no retrieval requirement to check -- it auto-passes
here, and is graded on refusal-appropriateness by the LLM judge instead.
"""
from __future__ import annotations

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


class RetrievalHitRateScorer(Scorer):
    name = "retrieval_hit_rate"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        expected_ids = set(case.expected.get("relevant_doc_ids", []))
        if not expected_ids:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no retrieval requirement"},
                               applicable=False)

        retrieved_ids = {c["source"] for c in trace.retrieved_chunks}
        hits = expected_ids & retrieved_ids
        hit_rate = len(hits) / len(expected_ids)

        return ScoreResult(
            self.name,
            score=hit_rate,
            passed=(hit_rate == 1.0),
            details={
                "expected": sorted(expected_ids),
                "retrieved": sorted(retrieved_ids),
                "missed": sorted(expected_ids - retrieved_ids),
            } if hit_rate < 1.0 else {},
        )

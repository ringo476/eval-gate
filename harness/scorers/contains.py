"""Deterministic check: does the output contain all required substrings
(case-insensitive)? Cheapest, most reliable scorer -- prefer it whenever
the expected behavior can be phrased this way. An empty requirement list
always passes (nothing to check), which matters for adversarial/refusal
cases that only carry a reference_answer for the judge to grade."""
from __future__ import annotations

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


class ContainsScorer(Scorer):
    name = "contains"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        required = case.expected.get("answer_contains", [])
        if not required:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no requirement"})

        output_lower = trace.output_text.lower()
        missing = [s for s in required if s.lower() not in output_lower]
        score = 1.0 - (len(missing) / len(required))
        return ScoreResult(
            self.name,
            score=score,
            passed=(len(missing) == 0),
            details={"missing": missing} if missing else {},
        )

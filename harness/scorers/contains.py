"""Deterministic check: does the output contain all required text? Cheapest,
most reliable scorer -- prefer it whenever the expected behavior can be
phrased this way. An empty requirement list has nothing to check, which
matters for adversarial/refusal cases that only carry a reference_answer.

Each entry in `answer_contains` is either one string, or a list of acceptable
alternatives (`["carbon dioxide", "CO2"]`, any one will do).

Matching ignores differences that don't change the meaning, because a scorer
that fails a correct answer is a false alarm in the regression gate: case,
extra spaces, hyphens ("drought-resistant" / "drought resistant"), thousands
separators ("1,024" / "1024"), subscripts ("CO₂" / "CO2"), and the stray space
some models put inside decimals ("1. 5" / "1.5", found in a real run).
"""
from __future__ import annotations

import re
import unicodedata

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―-]", " ", text)
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ContainsScorer(Scorer):
    name = "contains"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        required = case.expected.get("answer_contains", [])
        if not required:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no requirement"}, applicable=False)

        output = normalize_text(trace.output_text)
        missing = []
        for entry in required:
            alternatives = entry if isinstance(entry, list) else [entry]
            if not any(normalize_text(alt) in output for alt in alternatives):
                missing.append(entry)
        score = 1.0 - (len(missing) / len(required))
        return ScoreResult(
            self.name,
            score=score,
            passed=(len(missing) == 0),
            details={"missing": missing} if missing else {},
        )

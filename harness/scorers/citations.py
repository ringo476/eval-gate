"""Deterministic hallucination check: if the answer cites a document, that
document must actually have been retrieved. A citation to a doc the model
never saw is a fabricated source, however good the sentence around it reads.

Free and needs no calibration. Answers that cite nothing have nothing to check.
"""
from __future__ import annotations

import re

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer

# Only the literal ids the model was shown. "Document 1" is left alone: models
# use it to mean "the first document", which would be a false alarm here.
_CITATION = re.compile(r"\bdoc[_-]?(\d+)\b", re.IGNORECASE)


def extract_citations(text: str) -> set[str]:
    return {f"doc{int(num)}" for num in _CITATION.findall(text)}


class CitationValidityScorer(Scorer):
    name = "citation_validity"

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        cited = extract_citations(trace.output_text)
        if not cited:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no citations"}, applicable=False)

        retrieved = {c["source"] for c in trace.retrieved_chunks}
        phantom = sorted(cited - retrieved)
        return ScoreResult(
            self.name,
            score=1.0 - len(phantom) / len(cited),
            passed=not phantom,
            details={"cited_but_not_retrieved": phantom, "retrieved": sorted(retrieved)} if phantom else {},
        )

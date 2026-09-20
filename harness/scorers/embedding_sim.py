"""Semantic similarity to a reference answer, using a local free embedding
model (no API cost). Use for open-ended answers where exact wording
doesn't matter but meaning does."""
from __future__ import annotations

from harness.core.embeddings import cosine_similarity, get_embedding_model
from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer


class EmbeddingSimilarityScorer(Scorer):
    name = "embedding_similarity"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        reference = case.expected.get("reference_answer")
        if not reference:
            return ScoreResult(self.name, score=1.0, passed=True, details={"note": "no reference"})
        if not trace.output_text.strip():
            return ScoreResult(self.name, score=0.0, passed=False, details={"note": "empty output"})

        model = get_embedding_model()
        emb = model.encode([reference, trace.output_text])
        cos_sim = cosine_similarity(emb[0], emb[1])
        return ScoreResult(
            self.name,
            score=max(0.0, cos_sim),
            passed=cos_sim >= self.threshold,
            details={"cosine_similarity": round(cos_sim, 4)},
        )

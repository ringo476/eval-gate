"""Semantic similarity to a reference answer, using a local free embedding
model (no API cost). Use for open-ended answers where exact wording
doesn't matter but meaning does. Model is loaded lazily and cached as a
module-level singleton so it's only paid for (in load time) once per
process, not once per case."""
from __future__ import annotations

from harness.core.models import EvalCase, RunTrace, ScoreResult
from harness.scorers.base import Scorer

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


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

        model = _get_model()
        import numpy as np
        emb = model.encode([reference, trace.output_text])
        a, b = emb[0], emb[1]
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return ScoreResult(
            self.name,
            score=max(0.0, cos_sim),
            passed=cos_sim >= self.threshold,
            details={"cosine_similarity": round(cos_sim, 4)},
        )

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.core.models import EvalCase, RunTrace, ScoreResult


class Scorer(ABC):
    name: str

    @abstractmethod
    def score(self, case: EvalCase, trace: RunTrace) -> ScoreResult:
        ...

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.core.models import EvalCase, RunTrace
from harness.core.providers import ModelProvider


class Adapter(ABC):
    task: str

    @abstractmethod
    def run(self, case: EvalCase, provider: ModelProvider) -> RunTrace:
        """Execute the target application for one case using the given
        model provider, and return a RunTrace scorers can grade. Must
        never raise for an ordinary bad-output case -- put failures in
        RunTrace.error instead, so one bad case doesn't kill the run.

        The one exception is BudgetExceededError, which must propagate:
        it is a run-level condition (every remaining case would fail the
        same way), and the Runner turns it into an aborted run."""
        ...

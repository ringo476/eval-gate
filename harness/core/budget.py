"""
Hard spend cap for a single harness run.

Without this, a bad prompt change (e.g. one that makes the judge loop
retrying, or a golden set that accidentally grew from 30 to 3000 cases)
turns into a surprise bill with no circuit breaker. BudgetGuard is checked
before every paid API call and raises before the call is made, not after.
"""
from __future__ import annotations

import threading


class BudgetExceededError(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def check_and_reserve(self, estimated_cost: float) -> None:
        """Call BEFORE making a paid call. Raises if this call would blow the budget."""
        with self._lock:
            if self._spent + estimated_cost > self.max_cost_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self._spent:.4f} spent, "
                    f"${estimated_cost:.4f} more requested, cap is ${self.max_cost_usd:.2f}. "
                    f"Raise HARNESS_MAX_COST_USD or run a smaller --smoke subset."
                )

    def record(self, actual_cost: float) -> None:
        with self._lock:
            self._spent += actual_cost

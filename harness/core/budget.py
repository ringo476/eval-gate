"""
Hard spend cap for a single harness run.

Without this, a bad prompt change (e.g. one that makes the judge loop
retrying, or a golden set that accidentally grew from 30 to 3000 cases)
turns into a surprise bill with no circuit breaker. BudgetGuard is checked
before every paid API call and raises before the call is made, not after.

The check is a real reservation. Calls run in parallel, so a check that only
looks at money already spent lets every in-flight worker pass at once and
overshoot the cap by up to (workers x cost of a call). Each call therefore
holds its worst-case cost until it finishes, then swaps it for the actual cost.
"""
from __future__ import annotations

import threading


class BudgetExceededError(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> float:
        with self._lock:
            return self._reserved

    def check_and_reserve(self, estimated_cost: float) -> None:
        """Call BEFORE making a paid call. Raises if this call could blow the
        budget; otherwise holds `estimated_cost` until record() or release()."""
        with self._lock:
            if self._spent + self._reserved + estimated_cost > self.max_cost_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self._spent:.4f} spent and ${self._reserved:.4f} in flight, "
                    f"${estimated_cost:.4f} more requested, cap is ${self.max_cost_usd:.2f}. "
                    f"Raise HARNESS_MAX_COST_USD or run a smaller --smoke subset."
                )
            self._reserved += estimated_cost

    def release(self, reservation: float) -> None:
        """The call failed and cost nothing: give the held amount back."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)

    def record(self, actual_cost: float, reservation: float = 0.0) -> None:
        """The call finished: swap its reservation for what it really cost."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation)
            self._spent += actual_cost

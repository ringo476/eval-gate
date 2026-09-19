import pytest

from harness.core.budget import BudgetExceededError, BudgetGuard


def test_allows_spend_under_cap():
    guard = BudgetGuard(max_cost_usd=1.0)
    guard.check_and_reserve(0.5)
    guard.record(0.5)
    assert guard.spent == 0.5


def test_raises_before_exceeding_cap():
    guard = BudgetGuard(max_cost_usd=1.0)
    guard.check_and_reserve(0.9)
    guard.record(0.9)
    with pytest.raises(BudgetExceededError):
        guard.check_and_reserve(0.2)
    # the rejected reservation must not have been recorded
    assert guard.spent == 0.9


def test_exact_boundary_is_allowed():
    guard = BudgetGuard(max_cost_usd=1.0)
    guard.check_and_reserve(1.0)  # spent + estimate == cap, should not raise
    guard.record(1.0)
    assert guard.spent == 1.0

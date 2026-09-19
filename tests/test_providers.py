import pytest

from harness.core.budget import BudgetExceededError, BudgetGuard
from harness.core.cache import ResponseCache
from harness.core.models import ProviderResponse
from harness.core.providers import BudgetEnforcingProvider, CachingProvider


class CountingProvider:
    """Records how many times generate() was actually called, so tests
    can assert a cache hit skipped the underlying call entirely."""
    name = "counting-model"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system=None, tools=None):
        self.calls += 1
        return ProviderResponse(text=f"response #{self.calls}", input_tokens=10, output_tokens=5)


def test_caching_provider_skips_second_identical_call(tmp_path):
    cache = ResponseCache(str(tmp_path / "cache.db"))
    inner = CountingProvider()
    provider = CachingProvider(inner, cache)

    r1 = provider.generate("same prompt", system="sys")
    r2 = provider.generate("same prompt", system="sys")

    assert inner.calls == 1
    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.text == r1.text
    cache.close()


def test_caching_provider_distinguishes_different_prompts(tmp_path):
    cache = ResponseCache(str(tmp_path / "cache.db"))
    inner = CountingProvider()
    provider = CachingProvider(inner, cache)

    provider.generate("prompt A")
    provider.generate("prompt B")

    assert inner.calls == 2
    cache.close()


def test_budget_enforcing_provider_blocks_before_calling_inner():
    inner = CountingProvider()
    guard = BudgetGuard(max_cost_usd=0.000001)  # effectively zero

    def pricey_estimate(model, input_tokens, output_tokens):
        return 1.0  # any call "costs" $1, guaranteed to blow the cap

    provider = BudgetEnforcingProvider(inner, guard, pricey_estimate)
    with pytest.raises(BudgetExceededError):
        provider.generate("anything")

    assert inner.calls == 0  # the real call must never have happened


def test_budget_enforcing_provider_records_actual_cost():
    inner = CountingProvider()
    guard = BudgetGuard(max_cost_usd=10.0)

    def flat_estimate(model, input_tokens, output_tokens):
        return 0.01

    provider = BudgetEnforcingProvider(inner, guard, flat_estimate)
    provider.generate("hello")
    assert guard.spent == 0.01
    assert inner.calls == 1

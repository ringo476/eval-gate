"""
Model provider abstraction.

Layering:
  AnthropicProvider   -- real API calls, retried with exponential backoff
  CachingProvider      -- wraps any provider, skips the call on a cache hit
  MockProvider         -- deterministic $0 fake, for developing/testing the
                          harness itself without an API key or spend

Always wire providers through get_provider(), which applies caching
automatically -- nothing else in the codebase should instantiate
AnthropicProvider directly, or you lose the cost protection.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from harness.core.cache import ResponseCache
from harness.core.models import ProviderResponse

logger = logging.getLogger("harness.providers")


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        ...


class AnthropicProvider(ModelProvider):
    def __init__(self, model: str, api_key: str):
        self.name = model
        import anthropic  # imported lazily so mock-only usage never needs the package
        self._client = anthropic.Anthropic(api_key=api_key)
        self._RateLimitError = anthropic.RateLimitError
        self._APIConnectionError = anthropic.APIConnectionError
        self._APIStatusError = anthropic.APIStatusError

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        return self._generate_with_retry(prompt, system, tools)

    def _generate_with_retry(self, prompt, system, tools) -> ProviderResponse:
        @retry(
            retry=retry_if_exception_type((self._RateLimitError, self._APIConnectionError)),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call() -> ProviderResponse:
            start = time.perf_counter()
            kwargs: dict[str, Any] = dict(
                model=self.name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            resp = self._client.messages.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            text_parts = []
            tool_calls = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({"name": block.name, "input": block.input, "id": block.id})

            return ProviderResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms,
                raw=None,  # never cached/serialized; kept out of the dataclass by convention
            )

        return _call()


class BudgetEnforcingProvider(ModelProvider):
    """
    Wraps a real (network-calling) provider with a hard spend cap. Checked
    BEFORE the call using a worst-case token estimate (prompt length +
    max_tokens ceiling), so a run stops before overshooting the budget
    rather than after. Must wrap the real provider directly (inside any
    CachingProvider), so cache hits never count against the budget.
    """

    MAX_OUTPUT_TOKENS = 1024  # kept in sync with AnthropicProvider's max_tokens

    def __init__(self, inner: ModelProvider, budget, estimate_cost_fn):
        self.name = inner.name
        self._inner = inner
        self._budget = budget
        self._estimate_cost_fn = estimate_cost_fn

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        worst_case_input_tokens = len(prompt.split()) + len((system or "").split())
        worst_case_cost = self._estimate_cost_fn(
            self.name, worst_case_input_tokens, self.MAX_OUTPUT_TOKENS
        )
        self._budget.check_and_reserve(worst_case_cost)

        response = self._inner.generate(prompt, system=system, tools=tools)
        actual_cost = self._estimate_cost_fn(self.name, response.input_tokens, response.output_tokens)
        self._budget.record(actual_cost)
        return response


class CachingProvider(ModelProvider):
    """Wraps a real provider with a response cache, so identical
    (model, system, prompt, tools) calls never hit the API twice."""

    def __init__(self, inner: ModelProvider, cache: ResponseCache):
        self.name = inner.name
        self._inner = inner
        self._cache = cache

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        cached = self._cache.get(self.name, system, prompt, tools)
        if cached is not None:
            logger.debug("cache hit for model=%s", self.name)
            cached.cache_hit = True
            return cached

        response = self._inner.generate(prompt, system=system, tools=tools)
        self._cache.put(self.name, system, prompt, tools, response)
        response.cache_hit = False
        return response


class MockProvider(ModelProvider):
    """
    Deterministic fake provider for $0 development/testing of the harness
    itself. Rule-based and NOT meant to score well -- just exercises every
    code path (including tool-call shapes) with zero API cost.
    """

    def __init__(self, model: str = "mock"):
        self.name = "mock"

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        text = f"[mock answer for: {prompt[:60]}]"
        latency_ms = (time.perf_counter() - start) * 1000
        return ProviderResponse(
            text=text,
            tool_calls=[],
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=latency_ms,
            raw=None,
        )


def get_provider(
    model: str,
    api_key: str | None,
    cache: ResponseCache | None,
    budget=None,
) -> ModelProvider:
    if model == "mock":
        return MockProvider()

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for model "
            f"'{model}'. Use --provider mock to run without one."
        )

    from harness.core.pricing import estimate_cost_usd

    provider: ModelProvider = AnthropicProvider(model, api_key)
    if budget is not None:
        provider = BudgetEnforcingProvider(provider, budget, estimate_cost_usd)
    if cache is not None:
        provider = CachingProvider(provider, cache)
    return provider

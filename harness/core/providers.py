"""
Model provider abstraction.

Layering:
  AnthropicProvider   -- real API calls, retried with exponential backoff
  BudgetEnforcingProvider -- hard spend cap, checked before each call
  CachingProvider      -- wraps any provider, skips the call on a cache hit
  MockProvider         -- deterministic $0 fake, for developing/testing the
                          harness itself without an API key or spend

Two entry points, both implemented at every layer:
  generate(prompt, system, tools)          -- single-turn (rag_fusion adapter)
  generate_messages(messages, system, tools) -- multi-turn conversation,
      for a tool-calling agent loop that needs to replay its own previous
      turn (including tool_use blocks) back to the model before it can
      see a tool_result. `messages` follows Anthropic's message format:
      [{"role": "user"|"assistant", "content": <str or content-block list>}]

Always wire providers through get_provider(), which applies caching and
budget enforcement automatically -- nothing else in the codebase should
instantiate AnthropicProvider directly, or you lose that protection.
"""
from __future__ import annotations

import json
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

    @abstractmethod
    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        """Like generate(), but for the next turn of an existing multi-turn
        conversation. response.raw carries the JSON-safe content-block list
        the caller must append back as this turn's assistant message before
        sending a tool_result for the next call."""
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
        return self._call([{"role": "user", "content": prompt}], system, tools)

    def generate_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        return self._call(messages, system, tools)

    def _call(self, messages: list[dict[str, Any]], system, tools) -> ProviderResponse:
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
                messages=messages,
            )
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            resp = self._client.messages.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            text_parts = []
            tool_calls = []
            raw_blocks = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    raw_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
                    raw_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

            return ProviderResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms,
                # JSON-safe content blocks, so a tool-calling loop can replay
                # this exact assistant turn back to the API, AND so this
                # response can still be cached/restored intact (see cache.py).
                raw=raw_blocks,
            )

        return _call()


class BudgetEnforcingProvider(ModelProvider):
    """
    Wraps a real (network-calling) provider with a hard spend cap. Checked
    BEFORE the call using a worst-case token estimate, so a run stops
    before overshooting the budget rather than after. Must wrap the real
    provider directly (inside any CachingProvider), so cache hits never
    count against the budget.
    """

    MAX_OUTPUT_TOKENS = 1024  # kept in sync with AnthropicProvider's max_tokens

    def __init__(self, inner: ModelProvider, budget, estimate_cost_fn):
        self.name = inner.name
        self._inner = inner
        self._budget = budget
        self._estimate_cost_fn = estimate_cost_fn

    def _enforce(self, worst_case_input_tokens: int, call_fn):
        worst_case_cost = self._estimate_cost_fn(self.name, worst_case_input_tokens, self.MAX_OUTPUT_TOKENS)
        self._budget.check_and_reserve(worst_case_cost)

        response = call_fn()
        actual_cost = self._estimate_cost_fn(self.name, response.input_tokens, response.output_tokens)
        self._budget.record(actual_cost)
        return response

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        worst_case = len(prompt.split()) + len((system or "").split())
        return self._enforce(worst_case, lambda: self._inner.generate(prompt, system=system, tools=tools))

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        worst_case = len(json.dumps(messages).split()) + len((system or "").split())
        return self._enforce(worst_case, lambda: self._inner.generate_messages(messages, system=system, tools=tools))


class CachingProvider(ModelProvider):
    """Wraps a real provider with a response cache, so identical
    (model, system, prompt-or-messages, tools) calls never hit the API
    twice. Multi-turn calls are keyed on the serialized message list."""

    def __init__(self, inner: ModelProvider, cache: ResponseCache):
        self.name = inner.name
        self._inner = inner
        self._cache = cache

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        return self._cached_call(prompt, system, tools, lambda: self._inner.generate(prompt, system=system, tools=tools))

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        key_repr = json.dumps(messages, sort_keys=True)
        return self._cached_call(
            key_repr, system, tools,
            lambda: self._inner.generate_messages(messages, system=system, tools=tools),
        )

    def _cached_call(self, key_repr: str, system, tools, call_fn) -> ProviderResponse:
        cached = self._cache.get(self.name, system, key_repr, tools)
        if cached is not None:
            logger.debug("cache hit for model=%s", self.name)
            cached.cache_hit = True
            return cached

        response = call_fn()
        self._cache.put(self.name, system, key_repr, tools, response)
        response.cache_hit = False
        return response


class MockProvider(ModelProvider):
    """
    Deterministic fake provider for $0 development/testing of the harness
    itself. Rule-based and NOT meant to score well -- just exercises every
    code path (including a multi-step tool-calling loop) with zero cost.
    """

    def __init__(self, model: str = "mock"):
        self.name = "mock"

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        start = time.perf_counter()
        text = f"[mock answer for: {prompt[:60]}]"
        latency_ms = (time.perf_counter() - start) * 1000
        return ProviderResponse(
            text=text, tool_calls=[], input_tokens=len(prompt.split()),
            output_tokens=len(text.split()), latency_ms=latency_ms, raw=None,
        )

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        """Simulates one agent turn: if the most recent message is a
        tool_result, give a final text answer; otherwise, if tools are
        available, call one heuristically based on the user's question."""
        start = time.perf_counter()
        last = messages[-1]
        already_used_tool = last["role"] == "user" and isinstance(last["content"], list) and any(
            block.get("type") == "tool_result" for block in last["content"]
        )

        if already_used_tool or not tools:
            text = "[mock final answer based on tool result]"
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderResponse(
                text=text, tool_calls=[], input_tokens=20, output_tokens=len(text.split()),
                latency_ms=latency_ms, raw=[{"type": "text", "text": text}],
            )

        first_user_text = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        lowered = first_user_text.lower()
        if any(ch.isdigit() for ch in lowered) and any(op in lowered for op in ["+", "-", "*", "/", "times", "plus", "minus"]):
            tool_name, tool_input = "calculator", {"expression": "1+1"}
        else:
            tool_name, tool_input = "kb_lookup", {"topic": first_user_text[:40]}

        latency_ms = (time.perf_counter() - start) * 1000
        return ProviderResponse(
            text="",
            tool_calls=[{"name": tool_name, "input": tool_input, "id": "mock-call-1"}],
            input_tokens=20, output_tokens=5, latency_ms=latency_ms,
            raw=[{"type": "tool_use", "id": "mock-call-1", "name": tool_name, "input": tool_input}],
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

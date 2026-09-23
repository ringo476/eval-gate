"""
Model provider abstraction.

Layering:
  AnthropicProvider / OpenAIProvider -- real API calls, retried with backoff
  BudgetEnforcingProvider -- hard spend cap, checked before each call
  CachingProvider      -- wraps any provider, skips the call on a cache hit
  MockProvider         -- deterministic $0 fake, for developing/testing the
                          harness itself without an API key or spend

Every provider implements the same four methods, which is what makes the
tool-calling agent adapter (harness/adapters/tool_agent.py) provider-
agnostic instead of hardcoded to one vendor's message format:

  generate(prompt, system, tools)            -- single-turn (rag_fusion adapter)
  generate_messages(messages, system, tools) -- multi-turn conversation turn
  build_assistant_message(response)          -- wraps a turn's response back
                                                 into whatever shape this
                                                 provider's API expects as an
                                                 "assistant" message, for
                                                 replaying it in the next call
  build_tool_result_messages(tool_results)   -- wraps executed-tool outputs
                                                 into whatever message shape
                                                 this provider expects a tool
                                                 result to look like

Anthropic and OpenAI use genuinely different wire formats for both of the
last two (Anthropic: one user message with tool_result content blocks;
OpenAI: one role="tool" message per tool call) -- that's exactly the kind
of provider-specific detail this abstraction exists to hide from the
adapter.

`messages` follows a plain [{"role": ..., "content": ...}, ...] shape;
`tools` follows the neutral schema in harness/adapters/tools.py
(name/description/parameters) -- each provider converts it to its own
wire format internally.

Always wire providers through get_provider(), which applies caching and
budget enforcement automatically -- nothing else in the codebase should
instantiate AnthropicProvider/OpenAIProvider directly, or you lose that
protection.
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
        ...

    @abstractmethod
    def build_assistant_message(self, response: ProviderResponse) -> dict[str, Any]:
        ...

    @abstractmethod
    def build_tool_result_messages(self, tool_results: list[tuple[dict, str]]) -> list[dict[str, Any]]:
        """tool_results: [(tool_call_dict, result_text), ...] where
        tool_call_dict is one of response.tool_calls (has 'id', 'name', 'input')."""
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(ModelProvider):
    def __init__(self, model: str, api_key: str):
        self.name = model
        import anthropic  # imported lazily so unused providers never need the package
        self._client = anthropic.Anthropic(api_key=api_key)
        self._RateLimitError = anthropic.RateLimitError
        self._APIConnectionError = anthropic.APIConnectionError

    @staticmethod
    def _convert_tools(tools: list[dict] | None):
        if not tools:
            return None
        return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        return self._call([{"role": "user", "content": prompt}], system, tools)

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        return self._call(messages, system, tools)

    def _call(self, messages, system, tools) -> ProviderResponse:
        @retry(
            retry=retry_if_exception_type((self._RateLimitError, self._APIConnectionError)),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call() -> ProviderResponse:
            start = time.perf_counter()
            kwargs: dict[str, Any] = dict(model=self.name, max_tokens=1024, messages=messages)
            if system:
                kwargs["system"] = system
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                kwargs["tools"] = converted_tools

            resp = self._client.messages.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            text_parts, tool_calls, raw_blocks = [], [], []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    raw_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
                    raw_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

            return ProviderResponse(
                text="".join(text_parts), tool_calls=tool_calls,
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms, raw=raw_blocks,
            )

        return _call()

    def build_assistant_message(self, response: ProviderResponse) -> dict:
        return {"role": "assistant", "content": response.raw}

    def build_tool_result_messages(self, tool_results: list[tuple[dict, str]]) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
            for tc, result in tool_results
        ]}]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(ModelProvider):
    def __init__(self, model: str, api_key: str):
        self.name = model
        import openai  # imported lazily so unused providers never need the package
        self._client = openai.OpenAI(api_key=api_key)
        self._RateLimitError = openai.RateLimitError
        self._APIConnectionError = openai.APIConnectionError

    @staticmethod
    def _convert_tools(tools: list[dict] | None):
        if not tools:
            return None
        return [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            }}
            for t in tools
        ]

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self._call(messages, tools)

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        full_messages = ([{"role": "system", "content": system}] if system else []) + list(messages)
        return self._call(full_messages, tools)

    def _call(self, messages, tools) -> ProviderResponse:
        @retry(
            retry=retry_if_exception_type((self._RateLimitError, self._APIConnectionError)),
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call() -> ProviderResponse:
            start = time.perf_counter()
            kwargs: dict[str, Any] = dict(model=self.name, max_tokens=1024, messages=messages)
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                kwargs["tools"] = converted_tools

            resp = self._client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            msg = resp.choices[0].message

            tool_calls = []
            raw_tool_calls = None
            if msg.tool_calls:
                raw_tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        parsed_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    tool_calls.append({"name": tc.function.name, "input": parsed_args, "id": tc.id})
                    raw_tool_calls.append({
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    })

            # raw carries exactly what's needed to reconstruct this turn's
            # assistant message later: text content plus, if present,
            # OpenAI's own tool_calls shape (arguments kept as the original
            # JSON string, since that's what the API expects back verbatim).
            raw = {"content": msg.content, "tool_calls": raw_tool_calls}

            return ProviderResponse(
                text=msg.content or "", tool_calls=tool_calls,
                input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens,
                latency_ms=latency_ms, raw=raw,
            )

        return _call()

    def build_assistant_message(self, response: ProviderResponse) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": response.raw.get("content")}
        if response.raw.get("tool_calls"):
            message["tool_calls"] = response.raw["tool_calls"]
        return message

    def build_tool_result_messages(self, tool_results: list[tuple[dict, str]]) -> list[dict]:
        return [
            {"role": "tool", "tool_call_id": tc["id"], "content": result}
            for tc, result in tool_results
        ]


# ---------------------------------------------------------------------------
# Wrappers (budget, cache) -- delegate every method to the wrapped provider
# ---------------------------------------------------------------------------

class BudgetEnforcingProvider(ModelProvider):
    """
    Wraps a real (network-calling) provider with a hard spend cap. Checked
    BEFORE the call using a worst-case token estimate, so a run stops
    before overshooting the budget rather than after. Must wrap the real
    provider directly (inside any CachingProvider), so cache hits never
    count against the budget.
    """

    MAX_OUTPUT_TOKENS = 1024  # kept in sync with each provider's max_tokens

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

    def build_assistant_message(self, response):
        return self._inner.build_assistant_message(response)

    def build_tool_result_messages(self, tool_results):
        return self._inner.build_tool_result_messages(tool_results)


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

    def build_assistant_message(self, response):
        return self._inner.build_assistant_message(response)

    def build_tool_result_messages(self, tool_results):
        return self._inner.build_tool_result_messages(tool_results)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

class MockProvider(ModelProvider):
    """
    Deterministic fake provider for $0 development/testing of the harness
    itself. Rule-based and NOT meant to score well -- just exercises every
    code path (including a multi-step tool-calling loop) with zero cost.
    Uses its own simple raw-message convention (mirroring Anthropic's
    shape); nothing sends it over a real API, so that's an implementation
    detail, not a compatibility requirement.
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

    def build_assistant_message(self, response: ProviderResponse) -> dict:
        return {"role": "assistant", "content": response.raw}

    def build_tool_result_messages(self, tool_results: list[tuple[dict, str]]) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
            for tc, result in tool_results
        ]}]


# ---------------------------------------------------------------------------

def get_provider(
    provider_name: str,
    model: str,
    api_key: str | None,
    cache: ResponseCache | None,
    budget=None,
) -> ModelProvider:
    if provider_name == "mock":
        return MockProvider()

    if not api_key:
        raise RuntimeError(
            f"An API key is required for provider '{provider_name}' (model '{model}'). "
            "Use --provider mock to run without one."
        )

    from harness.core.pricing import estimate_cost_usd

    if provider_name == "anthropic":
        provider: ModelProvider = AnthropicProvider(model, api_key)
    elif provider_name == "openai":
        provider = OpenAIProvider(model, api_key)
    else:
        raise ValueError(f"Unknown provider '{provider_name}'. Expected 'anthropic', 'openai', or 'mock'.")

    if budget is not None:
        provider = BudgetEnforcingProvider(provider, budget, estimate_cost_usd)
    if cache is not None:
        provider = CachingProvider(provider, cache)
    return provider

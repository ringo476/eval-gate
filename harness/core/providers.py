"""
Model provider abstraction.

Layering:
  AnthropicProvider / OpenAIProvider / OllamaProvider -- real model calls,
                          retried with backoff, with a client-side timeout
  BudgetEnforcingProvider -- hard spend cap, checked before each paid call
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
adapter. Ollama speaks the OpenAI wire format, so it reuses OpenAIProvider.

`messages` follows a plain [{"role": ..., "content": ...}, ...] shape;
`tools` follows the neutral schema in harness/adapters/tools.py
(name/description/parameters) -- each provider converts it to its own
wire format internally.

Always wire providers through get_provider(), which applies caching and
budget enforcement automatically -- nothing else in the codebase should
instantiate the concrete providers directly, or you lose that protection.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from harness.core.cache import ResponseCache
from harness.core.models import ProviderResponse

logger = logging.getLogger("harness.providers")

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
LOCAL_MIN_TIMEOUT_S = 600.0  # a cold-loaded 20B local model can take minutes to answer


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling/limit settings applied to every call a provider makes.
    They are part of the cache key: changing them must never serve a
    response generated under different settings."""
    temperature: float | None = 0.0
    max_output_tokens: int = 1024
    timeout_s: float = 120.0

    def fingerprint(self) -> str:
        return f"t={self.temperature};m={self.max_output_tokens}"


DEFAULT_SETTINGS = GenerationSettings()


class ModelProvider(ABC):
    name: str
    settings: GenerationSettings = DEFAULT_SETTINGS

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

    def scoped(self, sample_idx: int) -> ModelProvider:
        """A view of this provider for repeat-sample `sample_idx`. Only the
        caching layer cares (so each repeat is a fresh draw, not a replay
        of sample 0); everything else returns itself."""
        return self


# ---------------------------------------------------------------------------
# Shared plumbing for providers that make real network calls
# ---------------------------------------------------------------------------

class _NetworkProvider(ModelProvider):
    _retryable: tuple[type[BaseException], ...] = ()
    _retry_wait = staticmethod(wait_random_exponential(multiplier=1, max=30))
    _max_attempts = 5

    def _with_retry(self, fn):
        @retry(
            retry=retry_if_exception_type(self._retryable),
            wait=self._retry_wait,
            stop=stop_after_attempt(self._max_attempts),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def attempt():
            return fn()

        return attempt()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(_NetworkProvider):
    def __init__(self, model: str, api_key: str, settings: GenerationSettings = DEFAULT_SETTINGS, client=None):
        self.name = model
        self.settings = settings
        import anthropic  # imported lazily so unused providers never need the package
        # max_retries=0: retrying is ours (tenacity), so attempts don't compound silently.
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=settings.timeout_s, max_retries=0)
        self._retryable = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
        self._BadRequestError = anthropic.BadRequestError
        self._send_temperature = settings.temperature is not None

    @staticmethod
    def _convert_tools(tools: list[dict] | None):
        if not tools:
            return None
        return [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        return self._call([{"role": "user", "content": prompt}], system, tools)

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        return self._call(messages, system, tools)

    def _create(self, kwargs: dict[str, Any]):
        try:
            return self._client.messages.create(**kwargs)
        except self._BadRequestError as e:
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                logger.warning("model %s rejected temperature; retrying without it", self.name)
                self._send_temperature = False
                kwargs.pop("temperature")
                return self._client.messages.create(**kwargs)
            raise

    def _call(self, messages, system, tools) -> ProviderResponse:
        def attempt() -> ProviderResponse:
            start = time.perf_counter()
            kwargs: dict[str, Any] = dict(model=self.name, max_tokens=self.settings.max_output_tokens, messages=messages)
            if self._send_temperature:
                kwargs["temperature"] = self.settings.temperature
            if system:
                kwargs["system"] = system
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                kwargs["tools"] = converted_tools

            resp = self._create(kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            text_parts, tool_calls, raw_blocks = [], [], []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    raw_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
                    raw_blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

            finish = getattr(resp, "stop_reason", None)
            if finish == "max_tokens":
                logger.warning("model %s hit max_tokens=%d; output truncated", self.name, self.settings.max_output_tokens)
            return ProviderResponse(
                text="".join(text_parts), tool_calls=tool_calls,
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
                latency_ms=latency_ms, raw=raw_blocks, finish_reason=finish,
            )

        return self._with_retry(attempt)

    def build_assistant_message(self, response: ProviderResponse) -> dict:
        return {"role": "assistant", "content": response.raw}

    def build_tool_result_messages(self, tool_results: list[tuple[dict, str]]) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
            for tc, result in tool_results
        ]}]


# ---------------------------------------------------------------------------
# OpenAI (and anything that speaks its chat-completions wire format)
# ---------------------------------------------------------------------------

class OpenAIProvider(_NetworkProvider):
    # api.openai.com wants max_completion_tokens (max_tokens is rejected by
    # newer models). Subclasses/servers that only know max_tokens override it.
    token_param = "max_completion_tokens"

    def __init__(
        self, model: str, api_key: str | None, settings: GenerationSettings = DEFAULT_SETTINGS,
        base_url: str | None = None, client=None,
    ):
        self.name = model
        self.settings = settings
        import openai  # imported lazily so unused providers never need the package
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=settings.timeout_s, max_retries=0,
        )
        # APIConnectionError covers APITimeoutError; InternalServerError is any 5xx.
        self._retryable = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
        self._BadRequestError = openai.BadRequestError
        self._send_temperature = settings.temperature is not None

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

    def _adapt_kwargs(self, kwargs: dict[str, Any], error_text: str) -> bool:
        """Some models reject parameters others accept (temperature on
        reasoning models; max_tokens vs max_completion_tokens). Adapt once,
        remember the answer for later calls, and report whether we changed
        anything worth retrying."""
        msg = error_text.lower()
        if "temperature" in msg and "temperature" in kwargs:
            logger.warning("model %s rejected temperature; retrying without it", self.name)
            self._send_temperature = False
            kwargs.pop("temperature")
            return True
        for current, other in (("max_tokens", "max_completion_tokens"), ("max_completion_tokens", "max_tokens")):
            if current in msg and current in kwargs and other != current:
                logger.warning("model %s rejected %s; switching to %s", self.name, current, other)
                self.token_param = other
                kwargs[other] = kwargs.pop(current)
                return True
        return False

    def _create(self, kwargs: dict[str, Any]):
        for _ in range(3):
            try:
                return self._client.chat.completions.create(**kwargs)
            except self._BadRequestError as e:
                if not self._adapt_kwargs(kwargs, str(e)):
                    raise
        return self._client.chat.completions.create(**kwargs)

    def _call(self, messages, tools) -> ProviderResponse:
        def attempt() -> ProviderResponse:
            start = time.perf_counter()
            kwargs: dict[str, Any] = dict(model=self.name, messages=messages)
            kwargs[self.token_param] = self.settings.max_output_tokens
            if self._send_temperature:
                kwargs["temperature"] = self.settings.temperature
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                kwargs["tools"] = converted_tools

            resp = self._create(kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            choice = resp.choices[0]
            msg = choice.message

            tool_calls = []
            raw_tool_calls = None
            if msg.tool_calls:
                raw_tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        parsed_args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {}
                    if not isinstance(parsed_args, dict):
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

            text = msg.content or ""
            usage = getattr(resp, "usage", None)
            if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
                in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
            else:  # some OpenAI-compatible servers omit usage; estimate rather than crash
                in_tok = sum(len(json.dumps(m).split()) for m in messages)
                out_tok = len(text.split())

            finish = getattr(choice, "finish_reason", None)
            if finish == "length":
                logger.warning(
                    "model %s hit the output token limit (%d); output truncated -- raise "
                    "HARNESS_MAX_OUTPUT_TOKENS, especially for reasoning models",
                    self.name, self.settings.max_output_tokens,
                )
            return ProviderResponse(
                text=text, tool_calls=tool_calls, input_tokens=in_tok, output_tokens=out_tok,
                latency_ms=latency_ms, raw=raw, finish_reason=finish,
            )

        return self._with_retry(attempt)

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


class OllamaProvider(OpenAIProvider):
    """A local model served by Ollama through its OpenAI-compatible
    endpoint. Free, offline, and speaks the same wire format as OpenAI,
    so tool calling works unchanged for models that support it."""

    token_param = "max_tokens"

    def __init__(
        self, model: str, settings: GenerationSettings = DEFAULT_SETTINGS,
        base_url: str = DEFAULT_OLLAMA_URL, client=None,
    ):
        super().__init__(model, api_key="ollama", settings=settings, base_url=base_url, client=client)


def _ollama_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 3.0) -> list[str]:
    """Names of models the local Ollama server has pulled. Raises
    RuntimeError with an actionable message if it isn't reachable."""
    url = f"{_ollama_root(base_url)}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- fixed http(s) URL from config
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {_ollama_root(base_url)} ({e}). Start it (`ollama serve`, or open the "
            "Ollama app), or set HARNESS_OLLAMA_URL if it runs elsewhere."
        ) from e
    return [m.get("name", "") for m in data.get("models", [])]


def check_ollama(model: str, base_url: str = DEFAULT_OLLAMA_URL) -> None:
    """Fail fast, before any case runs, if the server is down or the model
    hasn't been pulled -- instead of every case erroring one by one."""
    names = list_ollama_models(base_url)
    if model in names or (":" not in model and f"{model}:latest" in names):
        return
    raise RuntimeError(
        f"Model '{model}' is not available in Ollama. Pull it with `ollama pull {model}`. "
        f"Available: {', '.join(names) or '(none)'}."
    )


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

    def __init__(self, inner: ModelProvider, budget, estimate_cost_fn):
        self.name = inner.name
        self.settings = getattr(inner, "settings", DEFAULT_SETTINGS)
        self._inner = inner
        self._budget = budget
        self._estimate_cost_fn = estimate_cost_fn

    @staticmethod
    def _worst_case_input_tokens(text: str) -> int:
        # ~4 chars/token is typical English; /3 deliberately over-estimates
        # (JSON, code and non-English text tokenize worse) so the
        # pre-call check errs toward stopping early, not overspending.
        return len(text) // 3 + 1

    def _enforce(self, worst_case_input_tokens: int, call_fn):
        worst_case_cost = self._estimate_cost_fn(
            self.name, worst_case_input_tokens, self.settings.max_output_tokens
        )
        self._budget.check_and_reserve(worst_case_cost)
        response = call_fn()
        actual_cost = self._estimate_cost_fn(self.name, response.input_tokens, response.output_tokens)
        self._budget.record(actual_cost)
        return response

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        worst_case = self._worst_case_input_tokens(prompt + (system or "") + json.dumps(tools or []))
        return self._enforce(worst_case, lambda: self._inner.generate(prompt, system=system, tools=tools))

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        worst_case = self._worst_case_input_tokens(json.dumps(messages) + (system or "") + json.dumps(tools or []))
        return self._enforce(worst_case, lambda: self._inner.generate_messages(messages, system=system, tools=tools))

    def build_assistant_message(self, response):
        return self._inner.build_assistant_message(response)

    def build_tool_result_messages(self, tool_results):
        return self._inner.build_tool_result_messages(tool_results)


class CachingProvider(ModelProvider):
    """Wraps a real provider with a response cache, so identical
    (model, settings, system, prompt-or-messages, tools) calls never hit
    the API twice. Multi-turn calls are keyed on the serialized message
    list. `salt` (the repeat-sample index) keeps repeat samples distinct."""

    def __init__(self, inner: ModelProvider, cache: ResponseCache, salt: int = 0):
        self.name = inner.name
        self.settings = getattr(inner, "settings", DEFAULT_SETTINGS)
        self._inner = inner
        self._cache = cache
        self._salt = salt

    def scoped(self, sample_idx: int) -> ModelProvider:
        if sample_idx == self._salt:
            return self
        return CachingProvider(self._inner, self._cache, salt=sample_idx)

    def _variant(self) -> str:
        base = self.settings.fingerprint() if getattr(self._inner, "settings", None) else ""
        return f"{base}|s={self._salt}" if self._salt else base

    def generate(self, prompt, system=None, tools=None) -> ProviderResponse:
        return self._cached_call(prompt, system, tools, lambda: self._inner.generate(prompt, system=system, tools=tools))

    def generate_messages(self, messages, system=None, tools=None) -> ProviderResponse:
        key_repr = json.dumps(messages, sort_keys=True)
        return self._cached_call(
            key_repr, system, tools,
            lambda: self._inner.generate_messages(messages, system=system, tools=tools),
        )

    def _cached_call(self, key_repr: str, system, tools, call_fn) -> ProviderResponse:
        variant = self._variant()
        cached = self._cache.get(self.name, system, key_repr, tools, variant)
        if cached is not None:
            logger.debug("cache hit for model=%s", self.name)
            cached.cache_hit = True
            return cached

        response = call_fn()
        self._cache.put(self.name, system, key_repr, tools, response, variant)
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

    def __init__(self, model: str = "mock", settings: GenerationSettings = DEFAULT_SETTINGS):
        self.name = "mock"
        self.settings = settings

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
    settings: GenerationSettings | None = None,
    base_url: str | None = None,
    allow_unpriced: bool = False,
) -> ModelProvider:
    settings = settings or DEFAULT_SETTINGS

    if provider_name == "mock":
        return MockProvider(settings=settings)

    from harness.core.pricing import estimate_cost_usd, is_priced

    provider: ModelProvider
    is_local = provider_name == "ollama"

    if is_local:
        if settings.timeout_s < LOCAL_MIN_TIMEOUT_S:
            settings = replace(settings, timeout_s=LOCAL_MIN_TIMEOUT_S)
        provider = OllamaProvider(model, settings=settings, base_url=base_url or DEFAULT_OLLAMA_URL)
    else:
        if not api_key:
            raise RuntimeError(
                f"An API key is required for provider '{provider_name}' (model '{model}'). "
                "Use --provider ollama or --provider mock to run without one."
            )
        if provider_name == "anthropic":
            provider = AnthropicProvider(model, api_key, settings=settings)
        elif provider_name == "openai":
            provider = OpenAIProvider(model, api_key, settings=settings)
        else:
            raise ValueError(f"Unknown provider '{provider_name}'. Expected 'anthropic', 'openai', 'ollama', or 'mock'.")

        if not is_priced(model) and not allow_unpriced:
            raise RuntimeError(
                f"No pricing entry for model '{model}' in harness/core/pricing.py, so the budget cap could not "
                "protect this run (unpriced calls would be counted as $0). Add its per-token price there, or pass "
                "--allow-unpriced to run anyway."
            )

    if budget is not None and not is_local:
        provider = BudgetEnforcingProvider(provider, budget, estimate_cost_usd)
    if cache is not None:
        provider = CachingProvider(provider, cache)
    return provider


__all__ = [
    "AnthropicProvider", "BudgetEnforcingProvider", "CachingProvider", "GenerationSettings", "MockProvider",
    "ModelProvider", "OllamaProvider", "OpenAIProvider", "check_ollama", "get_provider", "list_ollama_models",
]

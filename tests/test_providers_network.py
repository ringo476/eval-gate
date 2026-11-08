"""Provider-layer behavior that doesn't need a real API: cache scoping,
parameter adaptation, retries, Ollama preflight, pricing safety."""
import io
import json
import urllib.error
from types import SimpleNamespace

import httpx
import openai
import pytest
from tenacity import wait_none

from harness.core.budget import BudgetGuard
from harness.core.cache import ResponseCache
from harness.core.models import ProviderResponse
from harness.core.providers import (
    BudgetEnforcingProvider,
    CachingProvider,
    GenerationSettings,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    check_ollama,
    get_provider,
)

REQ = httpx.Request("POST", "http://test/v1/chat/completions")


def bad_request(message):
    return openai.BadRequestError(message, response=httpx.Response(400, request=REQ), body=None)


def completion(content="hi", tool_calls=None, usage=True, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7) if usage else None,
    )


class FakeClient:
    def __init__(self, *behaviors):
        self.behaviors = list(behaviors)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(dict(kwargs))
        outcome = self.behaviors.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def openai_provider(*behaviors, settings=GenerationSettings(temperature=0.3, max_output_tokens=64)):
    client = FakeClient(*behaviors)
    provider = OpenAIProvider("gpt-x", "key", settings=settings, client=client)
    provider._retry_wait = wait_none()
    return provider, client


# --- cache scoping ----------------------------------------------------------

class SettingsProvider:
    def __init__(self, settings):
        self.name = "m"
        self.settings = settings
        self.calls = 0

    def generate(self, prompt, system=None, tools=None):
        self.calls += 1
        return ProviderResponse(text=f"call {self.calls}")


def test_cache_does_not_serve_a_response_made_under_different_settings(tmp_path):
    cache = ResponseCache(str(tmp_path / "c.db"))
    cold, hot = SettingsProvider(GenerationSettings(temperature=0.0)), SettingsProvider(GenerationSettings(temperature=0.9))
    CachingProvider(cold, cache).generate("p")
    CachingProvider(hot, cache).generate("p")
    assert cold.calls == 1 and hot.calls == 1  # not shared
    CachingProvider(SettingsProvider(GenerationSettings(temperature=0.0)), cache).generate("p")  # same settings: hit
    cache.close()


def test_repeat_samples_are_fresh_draws_not_cache_replays(tmp_path):
    cache = ResponseCache(str(tmp_path / "c.db"))
    inner = SettingsProvider(GenerationSettings(temperature=0.8))
    provider = CachingProvider(inner, cache)

    first = provider.generate("p")
    second = provider.scoped(1).generate("p")
    third = provider.scoped(2).generate("p")
    assert inner.calls == 3 and len({first.text, second.text, third.text}) == 3

    # but each sample is itself cached, so re-running the same sample is free
    assert provider.scoped(1).generate("p").cache_hit is True
    assert provider.scoped(0).generate("p").cache_hit is True
    assert inner.calls == 3
    assert provider.scoped(0) is provider
    cache.close()


def test_cache_tolerates_unknown_fields_in_stored_rows(tmp_path):
    cache = ResponseCache(str(tmp_path / "c.db"))
    cache.put("m", None, "p", None, ProviderResponse(text="hi"))
    cache._conn.execute("UPDATE response_cache SET response_json = ?",
                        (json.dumps({"text": "hi", "field_from_the_future": 1}),))
    assert cache.get("m", None, "p", None).text == "hi"
    cache.close()


# --- budget estimate --------------------------------------------------------

def test_budget_worst_case_estimate_scales_with_characters_not_words():
    seen = []

    def estimate(model, input_tokens, output_tokens):
        seen.append((input_tokens, output_tokens))
        return 0.0

    inner = SettingsProvider(GenerationSettings(max_output_tokens=321))
    BudgetEnforcingProvider(inner, BudgetGuard(1.0), estimate).generate("x" * 3000)  # one 3000-char "word"
    worst_case_in, worst_case_out = seen[0]
    assert worst_case_in >= 1000  # a words-based estimate would have said 1
    assert worst_case_out == 321  # uses the provider's real max_output_tokens


# --- OpenAI-format provider -------------------------------------------------

def test_openai_uses_max_completion_tokens_and_sends_temperature():
    provider, client = openai_provider(completion("hello"))
    resp = provider.generate("q", system="s")
    kwargs = client.calls[0]
    assert kwargs["max_completion_tokens"] == 64 and "max_tokens" not in kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["messages"][0] == {"role": "system", "content": "s"}
    assert resp.text == "hello" and resp.input_tokens == 11 and resp.output_tokens == 7


def test_ollama_uses_max_tokens():
    client = FakeClient(completion("hello"))
    provider = OllamaProvider("llama3.1:8b", settings=GenerationSettings(max_output_tokens=50), client=client)
    provider.generate("q")
    assert client.calls[0]["max_tokens"] == 50 and "max_completion_tokens" not in client.calls[0]


def test_model_that_rejects_temperature_is_retried_without_it_and_remembered():
    provider, client = openai_provider(
        bad_request("Unsupported value: 'temperature' does not support 0.3 with this model."),
        completion("one"), completion("two"),
    )
    assert provider.generate("q").text == "one"
    assert "temperature" in client.calls[0] and "temperature" not in client.calls[1]
    provider.generate("q")
    assert "temperature" not in client.calls[2]  # learned: doesn't re-trip the error every call


def test_model_that_rejects_max_tokens_switches_param():
    provider, client = openai_provider(
        bad_request("Unsupported parameter: 'max_completion_tokens' is not supported, use 'max_tokens'."),
        completion("ok"),
    )
    assert provider.generate("q").text == "ok"
    assert "max_tokens" in client.calls[1] and "max_completion_tokens" not in client.calls[1]


def test_unrelated_bad_request_is_not_swallowed():
    provider, client = openai_provider(bad_request("This model's context length is exceeded."))
    with pytest.raises(openai.BadRequestError):
        provider.generate("q")
    assert len(client.calls) == 1


def test_transient_errors_are_retried_then_succeed():
    provider, client = openai_provider(
        openai.APIConnectionError(request=REQ), openai.APITimeoutError(request=REQ), completion("finally"),
    )
    assert provider.generate("q").text == "finally"
    assert len(client.calls) == 3


def test_retries_give_up_after_max_attempts():
    provider, client = openai_provider(*[openai.APIConnectionError(request=REQ)] * 5)
    with pytest.raises(openai.APIConnectionError):
        provider.generate("q")
    assert len(client.calls) == provider._max_attempts


def test_missing_usage_is_estimated_not_a_crash():
    provider, _ = openai_provider(completion("some words here", usage=False))
    resp = provider.generate("q")
    assert resp.input_tokens > 0 and resp.output_tokens == 3


def test_tool_call_parsing_and_message_shapes():
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name="calculator", arguments='{"expression": "1+1"}'))
    garbage = SimpleNamespace(id="call_2", function=SimpleNamespace(name="calculator", arguments="not json"))
    provider, _ = openai_provider(completion(None, tool_calls=[tc, garbage]))
    resp = provider.generate_messages([{"role": "user", "content": "q"}], tools=[
        {"name": "calculator", "description": "d", "parameters": {"type": "object"}}])

    assert resp.tool_calls[0] == {"name": "calculator", "input": {"expression": "1+1"}, "id": "call_1"}
    assert resp.tool_calls[1]["input"] == {}  # malformed arguments degrade to {}, not an exception
    assistant = provider.build_assistant_message(resp)
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["function"]["arguments"] == '{"expression": "1+1"}'
    results = provider.build_tool_result_messages([(resp.tool_calls[0], "2")])
    assert results == [{"role": "tool", "tool_call_id": "call_1", "content": "2"}]


def test_truncated_output_is_flagged(caplog):
    provider, _ = openai_provider(completion("cut off", finish="length"))
    with caplog.at_level("WARNING", logger="harness.providers"):
        resp = provider.generate("q")
    assert resp.finish_reason == "length" and "truncated" in caplog.text


# --- get_provider -----------------------------------------------------------

def test_unpriced_paid_model_is_refused_so_the_budget_cannot_be_silently_bypassed():
    with pytest.raises(RuntimeError, match="No pricing entry"):
        get_provider("openai", "gpt-some-new-model", "key", None, budget=BudgetGuard(1.0))
    provider = get_provider("openai", "gpt-some-new-model", "key", None, budget=BudgetGuard(1.0), allow_unpriced=True)
    assert provider.name == "gpt-some-new-model"


def test_priced_model_is_budget_wrapped():
    provider = get_provider("openai", "gpt-6-luna", "key", None, budget=BudgetGuard(1.0))
    assert isinstance(provider, BudgetEnforcingProvider)


def test_ollama_needs_no_key_and_is_not_budget_wrapped_but_is_cached(tmp_path):
    cache = ResponseCache(str(tmp_path / "c.db"))
    provider = get_provider("ollama", "llama3.1:8b", None, cache, budget=BudgetGuard(1.0))
    assert isinstance(provider, CachingProvider)
    assert isinstance(provider._inner, OllamaProvider)
    assert provider.settings.timeout_s >= 600  # cold-loading a big local model is slow
    cache.close()


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="API key is required"):
        get_provider("openai", "gpt-6-luna", None, None)


def test_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("nope", "m", "key", None)
    assert isinstance(get_provider("mock", "whatever", None, None), MockProvider)


# --- Ollama preflight -------------------------------------------------------

def fake_urlopen(payload=None, error=None):
    def _open(url, timeout=None):
        if error:
            raise error
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def test_check_ollama_accepts_a_pulled_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        fake_urlopen({"models": [{"name": "llama3.1:8b"}, {"name": "tiny:latest"}]}))
    check_ollama("llama3.1:8b")
    check_ollama("tiny")  # bare name resolves to :latest


def test_check_ollama_names_the_fix_for_a_missing_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen({"models": [{"name": "llama3.1:8b"}]}))
    with pytest.raises(RuntimeError, match="ollama pull qwen"):
        check_ollama("qwen")


def test_check_ollama_when_server_is_down(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen(error=urllib.error.URLError("refused")))
    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        check_ollama("anything")


def test_reasoning_model_that_refuses_tools_is_retried_with_reasoning_off_and_remembered():
    # Exact error from gpt-6-luna on the first real call.
    refusal = bad_request(
        "Function tools with reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. "
        "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
    )
    tools = [{"name": "calculator", "description": "d", "parameters": {"type": "object"}}]
    provider, client = openai_provider(refusal, completion("one"), completion("two"), completion("plain"))

    assert provider.generate_messages([{"role": "user", "content": "q"}], tools=tools).text == "one"
    assert "reasoning_effort" not in client.calls[0]
    assert client.calls[1]["reasoning_effort"] == "none"

    provider.generate_messages([{"role": "user", "content": "q"}], tools=tools)
    assert client.calls[2]["reasoning_effort"] == "none"  # learned: no repeat failure

    provider.generate("q")
    assert "reasoning_effort" not in client.calls[3]  # calls without tools keep the model's default

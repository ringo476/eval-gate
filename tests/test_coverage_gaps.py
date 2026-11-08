"""Paths that matter in production but aren't reached by the mock provider:
the embedding scorer, the Anthropic wire format, config validation, doctor."""
from types import SimpleNamespace

import anthropic
import httpx
import numpy as np
import pytest
from tenacity import wait_none

from harness import cli
from harness.config import load_config
from harness.core.models import EvalCase, RunTrace
from harness.core.providers import AnthropicProvider, GenerationSettings
from harness.scorers import embedding_sim
from harness.scorers.embedding_sim import EmbeddingSimilarityScorer

# --- embedding similarity ---------------------------------------------------


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[t] for t in texts])


def embed_score(monkeypatch, vectors, output="answer", reference="ref"):
    monkeypatch.setattr(embedding_sim, "get_embedding_model", lambda: FakeEmbedder(vectors))
    case = EvalCase(id="a", task="t", input={}, expected={"reference_answer": reference} if reference else {})
    return EmbeddingSimilarityScorer().score(case, RunTrace(case_id="a", output_text=output))


def test_similar_meaning_passes(monkeypatch):
    result = embed_score(monkeypatch, {"ref": [1.0, 0.0], "answer": [1.0, 0.1]})
    assert result.passed is True and result.score > 0.9


def test_different_meaning_fails_and_negative_similarity_floors_at_zero(monkeypatch):
    result = embed_score(monkeypatch, {"ref": [1.0, 0.0], "answer": [-1.0, 0.0]})
    assert result.passed is False and result.score == 0.0


def test_no_reference_is_not_applicable_and_empty_output_fails(monkeypatch):
    assert embed_score(monkeypatch, {}, reference=None).applicable is False
    assert embed_score(monkeypatch, {}, output="  ").passed is False


# --- Anthropic wire format --------------------------------------------------

REQ = httpx.Request("POST", "http://test/v1/messages")


def message(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=9, output_tokens=4))


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block():
    return SimpleNamespace(type="tool_use", id="t1", name="calculator", input={"expression": "1+1"})


def anthropic_provider(*behaviors):
    calls = []
    queue = list(behaviors)

    def create(**kwargs):
        calls.append(dict(kwargs))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicProvider("claude-x", "key", settings=GenerationSettings(temperature=0.2, max_output_tokens=99),
                                 client=client)
    provider._retry_wait = wait_none()
    return provider, calls


def test_anthropic_request_and_response_shape():
    provider, calls = anthropic_provider(message(text_block("hello "), text_block("world")))
    resp = provider.generate("q", system="s", tools=[{"name": "c", "description": "d", "parameters": {"type": "object"}}])
    assert resp.text == "hello world" and resp.input_tokens == 9
    assert calls[0]["system"] == "s" and calls[0]["max_tokens"] == 99 and calls[0]["temperature"] == 0.2
    assert calls[0]["tools"][0]["input_schema"] == {"type": "object"}  # neutral schema converted


def test_anthropic_tool_turn_replays_exactly():
    provider, _ = anthropic_provider(message(text_block("thinking"), tool_block()))
    resp = provider.generate_messages([{"role": "user", "content": "q"}])
    assert resp.tool_calls == [{"name": "calculator", "input": {"expression": "1+1"}, "id": "t1"}]
    assert provider.build_assistant_message(resp)["content"][1]["type"] == "tool_use"
    (result,) = provider.build_tool_result_messages([(resp.tool_calls[0], "2")])
    assert result["role"] == "user" and result["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "2"}


def test_anthropic_retries_transient_errors_and_adapts_to_temperature_rejection():
    rejected = anthropic.BadRequestError("temperature is not supported", response=httpx.Response(400, request=REQ), body=None)
    provider, calls = anthropic_provider(anthropic.APIConnectionError(request=REQ), rejected, message(text_block("ok")))
    assert provider.generate("q").text == "ok"
    assert "temperature" in calls[1] and "temperature" not in calls[2]


def test_anthropic_flags_truncation(caplog):
    provider, _ = anthropic_provider(message(text_block("cut"), stop_reason="max_tokens"))
    with caplog.at_level("WARNING", logger="harness.providers"):
        assert provider.generate("q").finish_reason == "max_tokens"
    assert "truncated" in caplog.text


# --- config validation ------------------------------------------------------


@pytest.mark.parametrize("name,value,message", [
    ("HARNESS_MAX_COST_USD", "lots", "must be a number"),
    ("HARNESS_MAX_COST_USD", "0", "must be > 0"),
    ("HARNESS_MAX_WORKERS", "two", "must be an integer"),
    ("HARNESS_MAX_WORKERS", "0", ">= 1"),
    ("HARNESS_TEMPERATURE", "9", "between 0 and 2"),
    ("HARNESS_MAX_OUTPUT_TOKENS", "1", ">= 16"),
])
def test_bad_config_fails_fast_naming_the_variable(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        load_config()


def test_empty_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")  # e.g. an unset GitHub secret
    config = load_config()
    assert config.openai_api_key is None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.require_api_key("openai")
    assert config.require_api_key("ollama") is None and config.require_api_key("mock") is None
    with pytest.raises(ValueError):
        config.require_api_key("nope")


# --- doctor -----------------------------------------------------------------


def test_doctor_never_prints_key_values(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    monkeypatch.setattr(cli, "list_ollama_models", lambda url: ["m:1"])
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "sk-super-secret-value" not in out
    assert "OPENAI_API_KEY is set" in out and "m:1" in out
    assert code in (0, 3)  # 3 only if the vendored target isn't present yet


def test_doctor_treats_a_stopped_ollama_as_optional(monkeypatch, capsys):
    def down(url):
        raise RuntimeError("Cannot reach Ollama at http://x (refused)")

    monkeypatch.setattr(cli, "list_ollama_models", down)
    cli.main(["doctor"])
    assert "only needed for --provider ollama" in capsys.readouterr().out

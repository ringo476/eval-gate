from harness.adapters.tool_agent import ToolAgentAdapter
from harness.core.cache import ResponseCache
from harness.core.models import EvalCase
from harness.core.providers import CachingProvider, MockProvider


def test_mock_provider_agent_turn_then_final_answer():
    provider = MockProvider()
    tools = [{"name": "calculator", "input_schema": {}}]

    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    first = provider.generate_messages(messages, system="sys", tools=tools)
    assert len(first.tool_calls) == 1
    assert first.raw[0]["type"] == "tool_use"

    messages.append({"role": "assistant", "content": first.raw})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": first.tool_calls[0]["id"], "content": "4"}
    ]})
    second = provider.generate_messages(messages, system="sys", tools=tools)
    assert second.tool_calls == []
    assert second.text


def test_multiturn_caching_replays_identical_conversation(tmp_path):
    cache = ResponseCache(str(tmp_path / "cache.db"))
    provider = CachingProvider(MockProvider(), cache)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    tools = [{"name": "calculator", "input_schema": {}}]

    r1 = provider.generate_messages(messages, system="sys", tools=tools)
    r2 = provider.generate_messages(messages, system="sys", tools=tools)

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    # raw content blocks must survive the cache round-trip -- the agent
    # loop needs them to replay the assistant turn correctly
    assert r2.raw == r1.raw
    cache.close()


def test_tool_agent_adapter_end_to_end_with_mock():
    adapter = ToolAgentAdapter()
    case = EvalCase(id="ta-test", task="tool_agent_qa", input={"query": "What is 5 + 5?"}, expected={})
    trace = adapter.run(case, MockProvider())

    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0]["name"] == "calculator"
    assert trace.output_text  # got a final answer after the tool call


def test_tool_agent_adapter_never_raises_on_bad_tool_input():
    """Even if a tool call has malformed input, the adapter must return an
    error trace, not propagate an exception (Adapter contract)."""

    class BadToolCallProvider:
        name = "bad-provider"

        def generate(self, prompt, system=None, tools=None):
            raise NotImplementedError

        def generate_messages(self, messages, system=None, tools=None):
            from harness.core.models import ProviderResponse
            # missing required "expression" key
            return ProviderResponse(
                text="", tool_calls=[{"name": "calculator", "input": {}, "id": "x"}],
                input_tokens=1, output_tokens=1,
                raw=[{"type": "tool_use", "id": "x", "name": "calculator", "input": {}}],
            )

        def build_assistant_message(self, response):
            return {"role": "assistant", "content": response.raw}

        def build_tool_result_messages(self, tool_results):
            return [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
                for tc, result in tool_results
            ]}]

    adapter = ToolAgentAdapter()
    case = EvalCase(id="ta-bad", task="tool_agent_qa", input={"query": "..."}, expected={})
    trace = adapter.run(case, BadToolCallProvider())

    # calculator("") should come back as a handled "error: ..." string,
    # not raise -- and even if something else did raise, the adapter's
    # own try/except must have caught it rather than crash the run.
    assert trace.case_id == "ta-bad"

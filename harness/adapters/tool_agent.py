"""
Tool-calling agent adapter -- genuinely agentic, unlike rag_fusion.py's
fixed pipeline: the model decides, turn by turn, whether to call a tool,
which one, with what arguments, and when it has enough to give a final
answer. The adapter drives the loop and executes tools locally; the model
controls the control flow.
"""
from __future__ import annotations

import time

from harness.adapters.base import Adapter
from harness.adapters.tools import TOOL_SCHEMAS, execute_tool
from harness.core.models import EvalCase, RunTrace
from harness.core.pricing import estimate_cost_usd
from harness.core.providers import ModelProvider

AGENT_SYSTEM = (
    "You are an assistant with access to two tools: `calculator` for arithmetic "
    "and `kb_lookup` for factual questions about climate change. Always use a tool "
    "rather than computing or recalling facts yourself. You may call more than one "
    "tool, in sequence, if the question requires it. Once you have what you need, "
    "give a final answer in plain text without calling any more tools."
)

MAX_STEPS = 4  # hard ceiling on tool-call turns, so a confused loop can't run away


def _cost_of(model: str, response) -> float:
    if response.cache_hit:
        return 0.0
    return estimate_cost_usd(model, response.input_tokens, response.output_tokens)


class ToolAgentAdapter(Adapter):
    task = "tool_agent_qa"

    def run(self, case: EvalCase, provider: ModelProvider) -> RunTrace:
        query = case.input["query"]
        messages: list[dict] = [{"role": "user", "content": query}]
        tool_calls_made: list[dict] = []
        total_input_tokens = total_output_tokens = 0
        total_cost = 0.0
        all_cached = True

        try:
            start = time.perf_counter()

            for _step in range(MAX_STEPS):
                response = provider.generate_messages(messages, system=AGENT_SYSTEM, tools=TOOL_SCHEMAS)
                total_input_tokens += response.input_tokens
                total_output_tokens += response.output_tokens
                total_cost += _cost_of(provider.name, response)
                all_cached = all_cached and response.cache_hit

                if not response.tool_calls:
                    latency_ms = (time.perf_counter() - start) * 1000
                    return RunTrace(
                        case_id=case.id,
                        output_text=response.text,
                        tool_calls=tool_calls_made,
                        model=provider.name,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        latency_ms=latency_ms,
                        cost_usd=total_cost,
                        cache_hit=all_cached,
                    )

                # Replay this exact assistant turn before sending tool
                # results back -- every provider's API needs its own prior
                # tool-call turn echoed back before it will accept a
                # matching tool result, but the *shape* of that message
                # differs by provider (Anthropic: content blocks;
                # OpenAI: a `tool_calls` field). That's exactly what
                # build_assistant_message() hides from this adapter.
                messages.append(provider.build_assistant_message(response))

                executed = []
                for tc in response.tool_calls:
                    result_text = execute_tool(tc["name"], tc["input"])
                    tool_calls_made.append(tc)
                    executed.append((tc, result_text))
                messages.extend(provider.build_tool_result_messages(executed))

            # Hit MAX_STEPS without a final answer -- report it as a real
            # failure (empty output), not a silent success with no answer.
            latency_ms = (time.perf_counter() - start) * 1000
            return RunTrace(
                case_id=case.id,
                output_text="",
                tool_calls=tool_calls_made,
                model=provider.name,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                cache_hit=all_cached,
                error=f"exceeded MAX_STEPS={MAX_STEPS} without a final answer",
            )

        except Exception as e:  # noqa: BLE001 -- adapters must never raise, see base.Adapter
            return RunTrace(
                case_id=case.id,
                output_text="",
                tool_calls=tool_calls_made,
                model=provider.name,
                cost_usd=total_cost,
                error=f"{type(e).__name__}: {e}",
            )

"""
Approximate per-token pricing, USD per 1M tokens.

Estimates for cost-tracking purposes only -- verify against
https://platform.openai.com/docs/pricing (OpenAI) or
https://anthropic.com/pricing (Anthropic, kept for optional use) before
relying on them for real budgeting. Kept in one place so a pricing
update is a one-line change; if estimate_cost_usd() sees a model name
that isn't listed here, it returns $0 rather than raising, so an unknown
future model name doesn't crash a run over a cost-tracking miss -- check
the logs if a real run reports suspiciously low cost.
"""

PRICING_PER_MTOK = {
    # OpenAI -- primary provider for this project
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},

    # Anthropic -- supported but not the default; see harness/config.py
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},

    "mock": {"input": 0.0, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

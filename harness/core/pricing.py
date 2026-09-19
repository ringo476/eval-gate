"""
Approximate per-token pricing, USD per 1M tokens.

Estimates for cost-tracking purposes only -- verify against
https://anthropic.com/pricing before relying on them for real budgeting.
Kept in one place so a pricing update is a one-line change.
"""

PRICING_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "mock": {"input": 0.0, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

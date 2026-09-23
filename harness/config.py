"""
Central config loading and validation.

Fails fast and loudly at startup with an actionable message rather than
letting a missing env var surface 40 API calls into a run as a cryptic
auth error. This is the single place that reads os.environ -- nothing
else in the codebase should call os.environ.get directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    openai_api_key: str | None
    anthropic_api_key: str | None
    max_cost_usd: float
    max_workers: int
    judge_model: str | None
    cache_db_path: str
    runs_db_path: str

    def require_api_key(self, provider_name: str) -> str:
        if provider_name == "openai":
            if not self.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
                    "(get a key at https://platform.openai.com/api-keys), or pass "
                    "--provider mock to develop/test without an API key or spending money."
                )
            return self.openai_api_key
        if provider_name == "anthropic":
            if not self.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                    "or pass --provider mock to develop/test without an API key or spending money."
                )
            return self.anthropic_api_key
        raise ValueError(f"Unknown provider '{provider_name}'. Expected 'openai', 'anthropic', or 'mock'.")


def load_config() -> Config:
    max_cost_raw = os.environ.get("HARNESS_MAX_COST_USD", "5.00")
    max_workers_raw = os.environ.get("HARNESS_MAX_WORKERS", "4")

    try:
        max_cost_usd = float(max_cost_raw)
    except ValueError as e:
        raise RuntimeError(f"HARNESS_MAX_COST_USD must be a number, got {max_cost_raw!r}") from e

    try:
        max_workers = int(max_workers_raw)
    except ValueError as e:
        raise RuntimeError(f"HARNESS_MAX_WORKERS must be an integer, got {max_workers_raw!r}") from e

    if max_workers < 1:
        raise RuntimeError("HARNESS_MAX_WORKERS must be >= 1")
    if max_cost_usd <= 0:
        raise RuntimeError("HARNESS_MAX_COST_USD must be > 0")

    return Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_cost_usd=max_cost_usd,
        max_workers=max_workers,
        # None (not a hardcoded default) when unset: the CLI picks a
        # sensible default *for whichever --provider was chosen*, since a
        # judge model name only makes sense paired with a matching
        # provider (see DEFAULT_JUDGE_MODEL_BY_PROVIDER in cli.py).
        judge_model=os.environ.get("HARNESS_JUDGE_MODEL"),
        cache_db_path=os.environ.get("HARNESS_CACHE_DB", ".harness_cache.db"),
        runs_db_path=os.environ.get("HARNESS_RUNS_DB", "runs.db"),
    )

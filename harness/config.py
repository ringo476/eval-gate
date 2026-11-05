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

PROVIDERS = ("openai", "anthropic", "ollama", "mock")
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Config:
    openai_api_key: str | None
    anthropic_api_key: str | None
    max_cost_usd: float
    max_workers: int
    judge_model: str | None
    cache_db_path: str
    runs_db_path: str
    baselines_dir: str = "baselines"
    reports_dir: str = "reports"
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    temperature: float = 0.0
    max_output_tokens: int = 1024

    def require_api_key(self, provider_name: str) -> str | None:
        """Returns the key for a hosted provider, raising an actionable
        error if it's missing. Providers that need no key (mock, ollama)
        return None."""
        if provider_name in ("mock", "ollama"):
            return None
        if provider_name == "openai":
            if not self.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
                    "(get a key at https://platform.openai.com/api-keys), or use "
                    "--provider ollama (local model) / --provider mock to run without a key."
                )
            return self.openai_api_key
        if provider_name == "anthropic":
            if not self.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in, "
                    "or use --provider ollama / --provider mock to run without a key."
                )
            return self.anthropic_api_key
        raise ValueError(f"Unknown provider '{provider_name}'. Expected one of: {', '.join(PROVIDERS)}.")


def _env_number(name: str, default: str, cast, kind: str):
    raw = os.environ.get(name, default)
    try:
        return cast(raw)
    except ValueError as e:
        raise RuntimeError(f"{name} must be {kind}, got {raw!r}") from e


def load_config() -> Config:
    max_cost_usd = _env_number("HARNESS_MAX_COST_USD", "5.00", float, "a number")
    max_workers = _env_number("HARNESS_MAX_WORKERS", "4", int, "an integer")
    temperature = _env_number("HARNESS_TEMPERATURE", "0.0", float, "a number")
    max_output_tokens = _env_number("HARNESS_MAX_OUTPUT_TOKENS", "1024", int, "an integer")

    if max_workers < 1:
        raise RuntimeError("HARNESS_MAX_WORKERS must be >= 1")
    if max_cost_usd <= 0:
        raise RuntimeError("HARNESS_MAX_COST_USD must be > 0")
    if not 0.0 <= temperature <= 2.0:
        raise RuntimeError("HARNESS_TEMPERATURE must be between 0 and 2")
    if max_output_tokens < 16:
        raise RuntimeError("HARNESS_MAX_OUTPUT_TOKENS must be >= 16")

    return Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        max_cost_usd=max_cost_usd,
        max_workers=max_workers,
        # None (not a hardcoded default) when unset: the CLI picks a
        # sensible default *for whichever --provider was chosen*, since a
        # judge model name only makes sense paired with a matching
        # provider (see DEFAULT_JUDGE_MODEL_BY_PROVIDER in cli.py).
        judge_model=os.environ.get("HARNESS_JUDGE_MODEL") or None,
        cache_db_path=os.environ.get("HARNESS_CACHE_DB", ".harness_cache.db"),
        runs_db_path=os.environ.get("HARNESS_RUNS_DB", "runs.db"),
        baselines_dir=os.environ.get("HARNESS_BASELINES_DIR", "baselines"),
        reports_dir=os.environ.get("HARNESS_REPORTS_DIR", "reports"),
        ollama_base_url=os.environ.get("HARNESS_OLLAMA_URL", DEFAULT_OLLAMA_URL),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

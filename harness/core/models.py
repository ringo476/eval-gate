"""Core data structures shared across the harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """One golden test case, loaded from a JSONL dataset."""
    id: str
    task: str
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    human_label: float | None = None  # 0..1, only present in calibration sets

    @property
    def is_pinned(self) -> bool:
        """Pinned cases are known-past-failures / must-never-regress cases."""
        return "pinned" in self.tags


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = None
    cache_hit: bool = False


@dataclass
class RunTrace:
    """Everything an adapter produced for one case, for scorers to inspect."""
    case_id: str
    output_text: str
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: str | None = None


@dataclass
class ScoreResult:
    scorer_name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case: EvalCase
    trace: RunTrace
    scores: list[ScoreResult]

    @property
    def aggregate_score(self) -> float:
        if self.trace.error:
            return 0.0
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def passed(self) -> bool:
        if self.trace.error:
            return False
        return all(s.passed for s in self.scores)


@dataclass
class RunSummary:
    run_id: str
    task: str
    model: str
    timestamp: str
    case_results: list[CaseResult]

    @property
    def total_cost_usd(self) -> float:
        return sum(c.trace.cost_usd for c in self.case_results)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.trace.latency_ms for c in self.case_results)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.case_results if c.trace.error)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(1 for c in self.case_results if c.passed) / len(self.case_results)

    @property
    def mean_score(self) -> float:
        if not self.case_results:
            return 0.0
        return sum(c.aggregate_score for c in self.case_results) / len(self.case_results)

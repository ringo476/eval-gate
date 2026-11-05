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
    finish_reason: str | None = None  # "length" means the output was truncated


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
    """One *sample* of one case. With --repeats 1 (the default) there is
    exactly one CaseResult per case; with --repeats N there are N."""
    case: EvalCase
    trace: RunTrace
    scores: list[ScoreResult]
    sample_idx: int = 0

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
class CaseAggregate:
    """All samples of one case, collapsed. This is the unit regression
    detection reasons about: one sample flipping is noise, a case whose
    pass rate collapses across samples is a regression."""
    case: EvalCase
    samples: list[CaseResult]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_passed(self) -> int:
        return sum(1 for s in self.samples if s.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_samples if self.samples else 0.0

    @property
    def passed(self) -> bool:
        """Strict majority of samples passed (so n=1 behaves exactly like
        the old single-sample semantics)."""
        return self.n_passed * 2 > self.n_samples

    @property
    def status(self) -> str:
        if self.n_passed == self.n_samples:
            return "pass"
        if self.n_passed == 0:
            return "fail"
        return "flaky"

    @property
    def mean_score(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.aggregate_score for s in self.samples) / len(self.samples)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.samples if s.trace.error)

    @property
    def representative(self) -> CaseResult:
        """The sample worth showing in a report: the first failing one, else the first."""
        for s in self.samples:
            if not s.passed:
                return s
        return self.samples[0]

    @property
    def total_cost_usd(self) -> float:
        return sum(s.trace.cost_usd for s in self.samples)

    @property
    def mean_latency_ms(self) -> float:
        return sum(s.trace.latency_ms for s in self.samples) / len(self.samples) if self.samples else 0.0


@dataclass
class RunSummary:
    run_id: str
    task: str
    model: str
    timestamp: str
    case_results: list[CaseResult]  # one entry per sample, ordered by (case, sample_idx)
    meta: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None

    @property
    def cases(self) -> list[CaseAggregate]:
        grouped: dict[str, list[CaseResult]] = {}
        for cr in self.case_results:
            grouped.setdefault(cr.case.id, []).append(cr)
        return [CaseAggregate(case=samples[0].case, samples=samples) for samples in grouped.values()]

    @property
    def is_partial(self) -> bool:
        return bool(self.meta.get("partial", False))

    @property
    def repeats(self) -> int:
        return max((c.n_samples for c in self.cases), default=1)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.trace.cost_usd for c in self.case_results)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.trace.latency_ms for c in self.case_results)

    @property
    def error_count(self) -> int:
        """Number of samples whose adapter run failed outright."""
        return sum(1 for c in self.case_results if c.trace.error)

    @property
    def pass_rate(self) -> float:
        cases = self.cases
        if not cases:
            return 0.0
        return sum(1 for c in cases if c.passed) / len(cases)

    @property
    def mean_score(self) -> float:
        cases = self.cases
        if not cases:
            return 0.0
        return sum(c.mean_score for c in cases) / len(cases)

    @property
    def flaky_cases(self) -> list[str]:
        return [c.case.id for c in self.cases if c.status == "flaky"]

"""
Task registry: one place mapping a task name to its golden dataset, its
adapter, and its scorer set. Adding a third adapter later means adding
one entry here, not touching the CLI's control flow.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness.adapters.base import Adapter
from harness.core.providers import ModelProvider
from harness.scorers.base import Scorer


@dataclass(frozen=True)
class TaskSpec:
    dataset_path: str
    adapter_factory: Callable[[], Adapter]
    scorers_factory: Callable[[ModelProvider], list[Scorer]]


def _rag_fusion_adapter():
    from harness.adapters.rag_fusion import RagFusionAdapter
    from harness.corpus import CORPUS
    return RagFusionAdapter(CORPUS)


def _rag_fusion_scorers(judge_provider):
    from harness.scorers.citations import CitationValidityScorer
    from harness.scorers.contains import ContainsScorer
    from harness.scorers.embedding_sim import EmbeddingSimilarityScorer
    from harness.scorers.llm_judge import GroundednessScorer, LLMJudgeScorer
    from harness.scorers.retrieval import RetrievalHitRateScorer

    return [
        ContainsScorer(),
        RetrievalHitRateScorer(),
        CitationValidityScorer(),
        EmbeddingSimilarityScorer(),
        LLMJudgeScorer(judge_provider),
        GroundednessScorer(judge_provider),
    ]


def _tool_agent_adapter():
    from harness.adapters.tool_agent import ToolAgentAdapter
    return ToolAgentAdapter()


def _tool_agent_scorers(judge_provider):
    from harness.scorers.contains import ContainsScorer
    from harness.scorers.embedding_sim import EmbeddingSimilarityScorer
    from harness.scorers.llm_judge import LLMJudgeScorer
    from harness.scorers.tool_usage import ToolUsageScorer

    return [
        ToolUsageScorer(),
        ContainsScorer(),
        EmbeddingSimilarityScorer(),
        LLMJudgeScorer(judge_provider),
    ]


TASKS: dict[str, TaskSpec] = {
    "rag_fusion_qa": TaskSpec(
        dataset_path="golden/rag_fusion_qa.jsonl",
        adapter_factory=_rag_fusion_adapter,
        scorers_factory=_rag_fusion_scorers,
    ),
    "tool_agent_qa": TaskSpec(
        dataset_path="golden/tool_agent_qa.jsonl",
        adapter_factory=_tool_agent_adapter,
        scorers_factory=_tool_agent_scorers,
    ),
}


def get_task(task_name: str) -> TaskSpec:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Known tasks: {', '.join(TASKS)}")
    return TASKS[task_name]

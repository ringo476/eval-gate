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
    # The `task` value the dataset's rows carry, when it differs from the
    # task name: a variant that reuses another task's golden set.
    dataset_task: str | None = None


def _rag_adapter(task_name: str, **variant):
    def factory():
        from harness.adapters.rag_fusion import RagFusionAdapter
        from harness.corpus import CORPUS
        return RagFusionAdapter(CORPUS, task_name=task_name, **variant)
    return factory


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


RAG_GOLDEN = "golden/rag_fusion_qa.jsonl"
RAG_HARD_GOLDEN = "golden/rag_hard_qa.jsonl"

TASKS: dict[str, TaskSpec] = {
    # The RAG variants below share one golden set, so `compare --tasks` can
    # pair them question by question. Neighbours differ in one thing each.
    "rag_fusion_qa": TaskSpec(  # rag-fusion as published
        dataset_path=RAG_GOLDEN,
        adapter_factory=_rag_adapter("rag_fusion_qa"),
        scorers_factory=_rag_fusion_scorers,
    ),
    "rag_single_qa": TaskSpec(  # control: one search with the original query
        dataset_path=RAG_GOLDEN,
        adapter_factory=_rag_adapter("rag_single_qa", retrieval="single"),
        scorers_factory=_rag_fusion_scorers,
        dataset_task="rag_fusion_qa",
    ),
    "rag_fusion_plain_qa": TaskSpec(  # fusion search, expanded queries kept out of the answer prompt
        dataset_path=RAG_GOLDEN,
        adapter_factory=_rag_adapter("rag_fusion_plain_qa", show_queries=False),
        scorers_factory=_rag_fusion_scorers,
        dataset_task="rag_fusion_qa",
    ),
    "rag_fusion_guarded_qa": TaskSpec(  # ...plus a prompt-injection guard
        dataset_path=RAG_GOLDEN,
        adapter_factory=_rag_adapter("rag_fusion_guarded_qa", show_queries=False, guard=True),
        scorers_factory=_rag_fusion_scorers,
        dataset_task="rag_fusion_qa",
    ),
    # Questions worded unlike the documents they need, the case query
    # expansion is meant for. Separate golden set, so these runs don't mix
    # into the main RAG task's history, baselines or gate.
    "rag_fusion_hard_qa": TaskSpec(
        dataset_path=RAG_HARD_GOLDEN,
        adapter_factory=_rag_adapter("rag_fusion_hard_qa"),
        scorers_factory=_rag_fusion_scorers,
    ),
    "rag_single_hard_qa": TaskSpec(
        dataset_path=RAG_HARD_GOLDEN,
        adapter_factory=_rag_adapter("rag_single_hard_qa", retrieval="single"),
        scorers_factory=_rag_fusion_scorers,
        dataset_task="rag_fusion_hard_qa",
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


def dataset_task_of(task_name: str) -> str:
    """The `task` value rows in this task's dataset must carry."""
    return get_task(task_name).dataset_task or task_name

"""The rag_single_qa control: rag-fusion's pipeline with expansion and fusion removed."""
import pytest

from harness.adapters import rag_fusion
from harness.cli import main
from harness.core.dataset import load_jsonl
from harness.core.models import EvalCase
from harness.core.providers import MockProvider, ProviderResponse
from harness.tasks import TASKS, dataset_task_of, get_task


def _run(monkeypatch, retrieval):
    searched, prompts = [], []

    class Target:
        def vector_search(self, query, collection, n_results=5):
            searched.append(query)
            return {"doc2": 0.9, "doc1": 0.8, "doc3": 0.7, "doc4": 0.6}

        def reciprocal_rank_fusion(self, results, verbose=False):
            return {"doc9": 0.5, "doc1": 0.4, "doc2": 0.3}

    class Collection:
        def get(self, ids):
            return {"ids": ids, "documents": [f"text of {i}" for i in ids]}

    class Provider(MockProvider):
        def generate(self, prompt, system=None, tools=None):
            prompts.append(system)
            return ProviderResponse(text="q1\nq2" if system == rag_fusion.QUERY_EXPANSION_SYSTEM else "answer",
                                    input_tokens=10, output_tokens=5)

    monkeypatch.setattr(rag_fusion, "get_target_module", lambda: Target())
    adapter = rag_fusion.RagFusionAdapter({"doc1": "x"}, top_k=3, retrieval=retrieval)
    monkeypatch.setattr(adapter, "_get_collection", lambda: Collection())
    trace = adapter.run(EvalCase(id="a", task="rag_fusion_qa", input={"query": "q"}, expected={}), Provider())
    return adapter, trace, searched, prompts


def test_single_mode_searches_once_with_the_original_query_and_makes_one_model_call(monkeypatch):
    adapter, trace, searched, prompts = _run(monkeypatch, "single")
    assert trace.error is None
    assert adapter.task == "rag_single_qa"
    assert searched == ["q"]
    assert prompts == [rag_fusion.SYNTHESIS_SYSTEM]  # no expansion call
    assert trace.retrieval_ranking == ["doc2", "doc1", "doc3", "doc4"]  # the plain search order, not fused
    assert [c["source"] for c in trace.retrieved_chunks] == ["doc2", "doc1", "doc3"]
    assert (trace.input_tokens, trace.output_tokens) == (10, 5)


def test_fusion_mode_is_unchanged(monkeypatch):
    adapter, trace, searched, prompts = _run(monkeypatch, "fusion")
    assert adapter.task == "rag_fusion_qa"
    assert searched == ["q", "q1", "q2"]
    assert prompts == [rag_fusion.QUERY_EXPANSION_SYSTEM, rag_fusion.SYNTHESIS_SYSTEM]
    assert trace.retrieval_ranking == ["doc9", "doc1", "doc2"]
    assert (trace.input_tokens, trace.output_tokens) == (20, 10)


def test_unknown_retrieval_mode_is_refused():
    with pytest.raises(ValueError):
        rag_fusion.RagFusionAdapter({}, retrieval="hybrid")


def test_single_task_reuses_the_rag_golden_set():
    spec = get_task("rag_single_qa")
    assert spec.dataset_path == get_task("rag_fusion_qa").dataset_path
    assert dataset_task_of("rag_single_qa") == "rag_fusion_qa"
    assert dataset_task_of("tool_agent_qa") == "tool_agent_qa"
    assert len(load_jsonl(spec.dataset_path, expected_task=dataset_task_of("rag_single_qa"))) >= 10
    assert spec.adapter_factory().task == "rag_single_qa"
    assert set(TASKS) >= {"rag_fusion_qa", "rag_single_qa", "tool_agent_qa"}


def test_compare_refuses_tasks_with_different_golden_sets(capsys):
    assert main(["compare", "--tasks", "rag_fusion_qa", "tool_agent_qa", "--models", "mock"]) == 3
    assert "share one golden set" in capsys.readouterr().err

"""
Adapter for the vendored rag-fusion target app (Raudaschl/rag-fusion,
pinned commit -- see scripts/setup_target.sh).

Design choice: we import and reuse rag-fusion's retrieval/fusion logic
UNMODIFIED (vector_search, reciprocal_rank_fusion) straight from the
vendored copy, because that logic is already model-agnostic. The two
steps the original hardcodes to OpenAI -- query expansion and answer
synthesis -- are reimplemented here against our own ModelProvider
abstraction, with equivalent prompts, so the harness can run different
models and providers (OpenAI, Anthropic, a local Ollama model) through an
otherwise-identical retrieval pipeline.

The vendored copy under targets/rag-fusion is never modified.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

from harness.adapters.base import Adapter
from harness.core.budget import BudgetExceededError
from harness.core.models import EvalCase, RunTrace
from harness.core.pricing import estimate_cost_usd
from harness.core.providers import ModelProvider

TARGET_ROOT = Path(__file__).resolve().parent.parent.parent / "targets" / "rag-fusion"

_target_module = None


def get_target_module():
    """Import the vendored main.py under its own module name (not on
    sys.path, no name collisions with the rest of the process)."""
    global _target_module
    if _target_module is not None:
        return _target_module

    main_path = TARGET_ROOT / "main.py"
    if not main_path.exists():
        raise RuntimeError(
            f"Target app not found at {main_path}. Run scripts/setup_target.sh "
            "(scripts/setup_target.ps1 on Windows) to vendor it first."
        )
    spec = importlib.util.spec_from_file_location("rag_fusion_target", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_fusion_target"] = module
    spec.loader.exec_module(module)
    _target_module = module
    return module


QUERY_EXPANSION_SYSTEM = (
    "You are a search expert. Generate multiple search queries based on a "
    "single input query, to improve retrieval recall. Output exactly 4 "
    "queries, one per line, with no numbering, bullets, or extra commentary."
)

SYNTHESIS_SYSTEM = (
    "You are a helpful assistant that synthesizes search results into a "
    "comprehensive, well-structured answer. Give more weight to the "
    "original query than to the expanded queries. Answer only from the "
    "provided documents -- if they don't contain the answer, say so."
)


def _leg_cost(model: str, response) -> float:
    """Actual USD spent for one provider call: 0 if it was a cache hit,
    since no money changed hands, else the estimated API cost."""
    if response.cache_hit:
        return 0.0
    return estimate_cost_usd(model, response.input_tokens, response.output_tokens)


def _clean_query_lines(raw_text: str) -> list[str]:
    """Defensive parsing: the model is told not to number its output, but
    strip common numbering/bullet artifacts anyway rather than trust it."""
    lines = []
    for line in raw_text.strip().split("\n"):
        cleaned = line.strip().lstrip("-*").strip()
        cleaned = cleaned.lstrip("0123456789").lstrip(".").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


class RagFusionAdapter(Adapter):
    """Runs the rag-fusion pipeline end to end against our own corpus,
    with `provider` supplying both the query-expansion and
    answer-synthesis steps."""

    task = "rag_fusion_qa"

    def __init__(self, corpus_docs: dict[str, str], top_k: int = 3):
        """corpus_docs: {doc_id: document_text}"""
        self.corpus_docs = corpus_docs
        self.top_k = top_k
        self._collection = None
        self._collection_lock = threading.Lock()

    def _get_collection(self):
        # Double-checked locking: the runner calls this from multiple
        # worker threads concurrently. Without the lock, several threads
        # race to construct ChromaDB's client at once, which produced
        # real (nondeterministic) errors in testing -- "Could not connect
        # to tenant default_tenant", "RustBindingsAPI has no attribute
        # 'bindings'" -- because the underlying client isn't safe to
        # initialize from multiple threads simultaneously.
        if self._collection is None:
            with self._collection_lock:
                if self._collection is None:
                    import chromadb
                    client = chromadb.Client()  # fresh in-memory client per process
                    collection = client.get_or_create_collection(name="harness_corpus")
                    collection.add(
                        documents=list(self.corpus_docs.values()),
                        ids=list(self.corpus_docs.keys()),
                    )
                    self._collection = collection
        return self._collection

    def run(self, case: EvalCase, provider: ModelProvider) -> RunTrace:
        original_query = case.input["query"]

        try:
            target = get_target_module()
            collection = self._get_collection()

            start = time.perf_counter()

            expansion_resp = provider.generate(
                f"Generate 4 search queries related to: {original_query}",
                system=QUERY_EXPANSION_SYSTEM,
            )
            expanded_queries = _clean_query_lines(expansion_resp.text)
            all_queries = [original_query] + expanded_queries

            # All ChromaDB access is serialized behind one lock -- the
            # client isn't proven thread-safe for concurrent queries, and
            # this corpus is tiny (24 docs), so the lock costs nothing
            # measurable while removing an entire class of flaky failures.
            all_results = {}
            with self._collection_lock:
                for query in all_queries:
                    all_results[query] = target.vector_search(query, collection, n_results=5)

                fused = target.reciprocal_rank_fusion(all_results, verbose=False)
                top_doc_ids = list(fused.keys())[: self.top_k]

                fetched = collection.get(ids=top_doc_ids)
                top_docs = list(zip(fetched["ids"], fetched["documents"]))

            doc_block = "\n".join(f"[{doc_id}] {text}" for doc_id, text in top_docs)
            query_block = "\n".join(f"- {q}" for q in all_queries)
            synthesis_prompt = (
                f"Original query: \"{original_query}\"\n\n"
                f"Expanded queries:\n{query_block}\n\n"
                f"Retrieved documents:\n{doc_block}\n\n"
                f"Provide a well-structured answer to the original query, "
                f"using only the documents above."
            )
            synthesis_resp = provider.generate(synthesis_prompt, system=SYNTHESIS_SYSTEM)

            latency_ms = (time.perf_counter() - start) * 1000

            total_cost = _leg_cost(provider.name, expansion_resp) + _leg_cost(provider.name, synthesis_resp)
            fully_cached = expansion_resp.cache_hit and synthesis_resp.cache_hit

            return RunTrace(
                case_id=case.id,
                output_text=synthesis_resp.text,
                retrieved_chunks=[{"source": doc_id, "text": text} for doc_id, text in top_docs],
                model=provider.name,
                input_tokens=expansion_resp.input_tokens + synthesis_resp.input_tokens,
                output_tokens=expansion_resp.output_tokens + synthesis_resp.output_tokens,
                latency_ms=latency_ms,
                cost_usd=total_cost,
                cache_hit=fully_cached,
            )

        except BudgetExceededError:
            raise  # run-level condition: the Runner aborts the whole run, see runner.py
        except Exception as e:  # noqa: BLE001 -- adapters must never raise, see base.Adapter
            return RunTrace(
                case_id=case.id,
                output_text="",
                model=provider.name,
                error=f"{type(e).__name__}: {e}",
            )

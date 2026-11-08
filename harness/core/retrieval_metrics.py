"""Standard retrieval metrics, for judging the search step separately from the
writing step. `ranking` is every retrieved doc id, best first; `relevant` is
the set a human marked as the right answer for that query.

  recall@k     of the relevant docs, how many are in the top k?
  precision@k  of the top k, how many are relevant?
  MRR          1 / (rank of the first relevant doc); rewards finding one early
  nDCG@k       like recall, but a relevant doc counts for less the lower it ranks
"""
from __future__ import annotations

import math
from collections.abc import Iterable

DEFAULT_KS = (1, 3, 5)


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Divides by k even when fewer than k docs came back: a search that
    returned too little shouldn't score higher for it."""
    if k <= 0:
        return 0.0
    return len(set(ranking[:k]) & relevant) / k


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranking, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(ranking[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def retrieval_metrics(ranking: list[str], relevant: set[str], ks: Iterable[int] = DEFAULT_KS) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranking, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(ranking, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranking, relevant, k)
    metrics["mrr"] = reciprocal_rank(ranking, relevant)
    return metrics


def aggregate_retrieval(case_results: Iterable) -> dict[str, float] | None:
    """Mean of each metric over samples that have both a ranking and at least
    one relevant doc. Out-of-corpus cases are excluded: with nothing relevant
    to find, they say nothing about search quality. None if no case qualifies
    (e.g. the agent task)."""
    totals: dict[str, float] = {}
    n = 0
    for cr in case_results:
        relevant = set(cr.case.expected.get("relevant_doc_ids", []))
        if cr.trace.error or not relevant or not cr.trace.retrieval_ranking:
            continue
        n += 1
        for name, value in retrieval_metrics(cr.trace.retrieval_ranking, relevant).items():
            totals[name] = totals.get(name, 0.0) + value
    if not n:
        return None
    return {**{name: round(value / n, 4) for name, value in totals.items()}, "n": n}

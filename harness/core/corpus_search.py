"""Thin semantic-search wrapper around the shared document corpus, used
by both the RAG-fusion adapter's retrieval step's sanity checks and the
tool-calling agent's kb_lookup tool -- one corpus, one search primitive,
two adapters."""
from __future__ import annotations

from harness.core.embeddings import most_similar
from harness.corpus import CORPUS


def search_corpus(query: str, top_k: int = 1) -> list[tuple[str, str, float]]:
    return most_similar(query, CORPUS, top_k=top_k)

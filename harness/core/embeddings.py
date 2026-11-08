"""
Shared local embedding model, lazy-loaded once per process. Both the
embedding-similarity scorer and the tool-calling agent's kb_lookup tool
need semantic search -- this is the one place that loads the model, so
it's paid for (load time) once, not per caller.
"""
from __future__ import annotations

import threading

import numpy as np

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    global _model
    if _model is None:
        with _model_lock:  # workers start at once; without this each would load its own copy
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def most_similar(query: str, candidates: dict[str, str], top_k: int = 1) -> list[tuple[str, str, float]]:
    """Returns [(key, text, similarity)] for the top_k most similar
    candidates to query, sorted descending by similarity."""
    model = get_embedding_model()
    keys = list(candidates.keys())
    texts = list(candidates.values())
    embeddings = model.encode([query] + texts)
    query_emb, doc_embs = embeddings[0], embeddings[1:]

    scored = [(keys[i], texts[i], cosine_similarity(query_emb, doc_embs[i])) for i in range(len(keys))]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

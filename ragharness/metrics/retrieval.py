from __future__ import annotations

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult


def precision_at_k(item: EvalItem, result: RAGResult, *, k: int = 5) -> float:
    """Fraction of top-k retrieved docs that appear in expected_docs."""
    if not item.expected_docs:
        return 0.0
    top_k = result.retrieved_docs[:k]
    if not top_k:
        return 0.0
    expected_set = set(item.expected_docs)
    hits = sum(1 for doc in top_k if doc in expected_set)
    return hits / len(top_k)

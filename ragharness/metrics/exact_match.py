from __future__ import annotations

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult


def exact_match(item: EvalItem, result: RAGResult) -> float:
    """1.0 if normalized answer matches expected, 0.0 otherwise."""
    return (
        1.0
        if result.answer.strip().lower() == item.expected_answer.strip().lower()
        else 0.0
    )

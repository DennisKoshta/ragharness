from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class RAGResult:
    """Result returned by a RAG system for a single query."""

    answer: str
    retrieved_docs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class RAGSystem(Protocol):
    """Protocol that any RAG system must satisfy.

    Implement a single ``query`` method that accepts a question string
    and returns a ``RAGResult`` containing the answer, retrieved
    documents, and metadata (latency_ms, prompt_tokens,
    completion_tokens, model, top_k).
    """

    def query(self, question: str) -> RAGResult: ...

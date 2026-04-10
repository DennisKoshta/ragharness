from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


class LangChainRAGSystem:
    """Wraps a LangChain RetrievalQA chain. Not yet implemented."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import langchain  # noqa: F401
        except ImportError:
            raise ImportError(
                "langchain package required. Install with: pip install ragharness[langchain]"
            ) from None
        raise NotImplementedError(
            "LangChain adapter is not yet implemented. "
            "Use the 'raw' adapter or implement the RAGSystem protocol directly."
        )

    def query(self, question: str) -> RAGResult:
        raise NotImplementedError

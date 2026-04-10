from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


class LlamaIndexRAGSystem:
    """Wraps a LlamaIndex query engine. Not yet implemented."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import llama_index  # noqa: F401
        except ImportError:
            raise ImportError(
                "llama-index package required. Install with: pip install ragharness[llamaindex]"
            ) from None
        raise NotImplementedError(
            "LlamaIndex adapter is not yet implemented. "
            "Use the 'raw' adapter or implement the RAGSystem protocol directly."
        )

    def query(self, question: str) -> RAGResult:
        raise NotImplementedError

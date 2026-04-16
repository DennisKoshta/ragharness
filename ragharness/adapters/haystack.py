from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


class HaystackRAGSystem:
    """Wraps a Haystack Pipeline. Not yet implemented."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import haystack  # noqa: F401
        except ImportError:
            raise ImportError(
                "haystack-ai package required. "
                "Install with: pip install ragharness[haystack]"
            ) from None
        raise NotImplementedError(
            "Haystack adapter is not yet implemented. "
            "Use the 'raw' adapter or implement the RAGSystem protocol directly."
        )

    def query(self, question: str) -> RAGResult:
        raise NotImplementedError

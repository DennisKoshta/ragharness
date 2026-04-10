from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


class R2RRAGSystem:
    """Wraps an R2R client. Not yet implemented."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import r2r  # noqa: F401
        except ImportError:
            raise ImportError(
                "r2r package required. Install with: pip install ragharness[r2r]"
            ) from None
        raise NotImplementedError(
            "R2R adapter is not yet implemented. "
            "Use the 'raw' adapter or implement the RAGSystem protocol directly."
        )

    def query(self, question: str) -> RAGResult:
        raise NotImplementedError

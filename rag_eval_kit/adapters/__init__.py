from __future__ import annotations

from typing import Any

from rag_eval_kit.protocol import RAGSystem


def create_adapter(
    adapter_type: str,
    adapter_config: dict[str, Any],
    sweep_overrides: dict[str, Any] | None = None,
) -> RAGSystem:
    """Instantiate a RAGSystem from an adapter type, its config, and sweep overrides.

    Sweep overrides are merged on top of the base adapter_config so that
    swept parameters (top_k, temperature, …) take effect.
    """
    merged = {**adapter_config, **(sweep_overrides or {})}

    if adapter_type == "raw":
        from rag_eval_kit.adapters.raw import RawRAGSystem

        return RawRAGSystem(**merged)
    elif adapter_type == "langchain":
        from rag_eval_kit.adapters.langchain import LangChainRAGSystem

        return LangChainRAGSystem(**merged)
    elif adapter_type == "llamaindex":
        from rag_eval_kit.adapters.llamaindex import LlamaIndexRAGSystem

        return LlamaIndexRAGSystem(**merged)
    elif adapter_type == "r2r":
        from rag_eval_kit.adapters.r2r import R2RRAGSystem

        return R2RRAGSystem(**merged)
    elif adapter_type == "haystack":
        from rag_eval_kit.adapters.haystack import HaystackRAGSystem

        return HaystackRAGSystem(**merged)
    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type!r}. "
            f"Supported: raw, langchain, llamaindex, r2r, haystack"
        )

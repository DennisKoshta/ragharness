from __future__ import annotations

from typing import Any

from ragharness.protocol import RAGResult


def token_cost(results: list[RAGResult], *, pricing: dict[str, Any]) -> float:
    """Total estimated cost in USD based on token counts and a pricing table.

    ``pricing`` maps model names to dicts with ``input_per_1k`` and
    ``output_per_1k`` keys (cost per 1 000 tokens).  If a result's model
    is not found in the table the cost for that result is zero.
    """
    total = 0.0
    for r in results:
        model = r.metadata.get("model", "")
        model_pricing = pricing.get(model, {})
        input_rate = model_pricing.get("input_per_1k", 0.0)
        output_rate = model_pricing.get("output_per_1k", 0.0)
        prompt_tokens = r.metadata.get("prompt_tokens", 0)
        completion_tokens = r.metadata.get("completion_tokens", 0)
        total += (prompt_tokens / 1000) * input_rate
        total += (completion_tokens / 1000) * output_rate
    return total

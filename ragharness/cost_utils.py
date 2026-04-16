"""Token counting and pre-run cost estimation.

When ``tiktoken`` is installed (``pip install ragharness[cost]``) the
prompt-token count is derived from the actual dataset text; otherwise
we fall back to ``len(text) // 4``. Completion tokens stay heuristic —
the true completion length is unknown before the run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragharness.dataset import EvalDataset

logger = logging.getLogger(__name__)

# Fallback pricing (USD per 1k tokens) for known models. Users can override
# by supplying a ``token_cost`` metric in the config with their own pricing
# table; ``estimate_sweep_cost`` reads that when available.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
    "claude-sonnet-4-20250514": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-opus-4-20250514": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "claude-haiku-4-5": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
}

_FALLBACK_CHARS_PER_TOKEN = 4
_encoding_cache: dict[str, Any] = {}
_tiktoken_warned = False


def _get_encoding(model: str) -> Any | None:
    """Return a tiktoken encoding for *model*, or ``None`` if unavailable."""
    global _tiktoken_warned
    try:
        import tiktoken
    except ImportError:
        if not _tiktoken_warned:
            logger.debug(
                "tiktoken not installed; using char/4 fallback. "
                "Install with: pip install ragharness[cost]"
            )
            _tiktoken_warned = True
        return None

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    name = enc.name
    if name not in _encoding_cache:
        _encoding_cache[name] = enc
    return _encoding_cache[name]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in *text* for *model*.

    Uses ``tiktoken`` when available, otherwise ``max(1, len(text) // 4)``.
    Returns 0 for empty input.
    """
    if not text:
        return 0
    enc = _get_encoding(model)
    if enc is None:
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text))


def estimate_sweep_cost(
    dataset: EvalDataset,
    sweep_configs: list[dict[str, Any]],
    *,
    model: str = "gpt-4o",
    input_per_1k: float | None = None,
    output_per_1k: float | None = None,
    avg_completion_tokens: int = 200,
    template_overhead_tokens: int = 50,
) -> float:
    """Estimate total sweep cost in USD, using real prompt-token counts.

    Each dataset item's question is tokenised once; the total is
    multiplied by the number of sweep configurations (all configs run
    every question). A flat ``template_overhead_tokens`` is added per
    call to account for the RAG prompt template + retrieved context the
    real adapter will prepend — unavoidable because we cannot run
    retrieval pre-run.

    Pricing resolution order: explicit kwargs > ``DEFAULT_PRICING[model]``
    > ``(0, 0)`` (estimate shows $0 for unknown models, which surfaces
    as "no confirmation prompt" but the sweep still runs).
    """
    if not sweep_configs or len(dataset) == 0:
        return 0.0

    if input_per_1k is None or output_per_1k is None:
        pricing = DEFAULT_PRICING.get(model, {})
        if input_per_1k is None:
            input_per_1k = pricing.get("input_per_1k", 0.0)
        if output_per_1k is None:
            output_per_1k = pricing.get("output_per_1k", 0.0)

    per_question_prompt = 0
    for item in dataset:
        per_question_prompt += count_tokens(item.question, model) + template_overhead_tokens

    n_configs = len(sweep_configs)
    total_prompt_tokens = per_question_prompt * n_configs
    total_completion_tokens = len(dataset) * n_configs * avg_completion_tokens

    prompt_cost = (total_prompt_tokens / 1000) * input_per_1k
    completion_cost = (total_completion_tokens / 1000) * output_per_1k
    return prompt_cost + completion_cost


def resolve_model_from_config(adapter_config: dict[str, Any], default: str = "gpt-4o") -> str:
    """Pick a reasonable model id out of an adapter_config dict."""
    model = adapter_config.get("llm_model") or adapter_config.get("model")
    return str(model) if model else default


def resolve_pricing_from_metrics(
    metrics: list[str | dict[str, Any]], model: str
) -> tuple[float | None, float | None]:
    """Pull ``input_per_1k`` / ``output_per_1k`` for *model* from a metrics list.

    Looks for a ``token_cost`` entry with a ``pricing`` dict keyed by
    model id. Returns ``(None, None)`` when not present so
    :func:`estimate_sweep_cost` falls through to defaults.
    """
    for entry in metrics:
        if not isinstance(entry, dict):
            continue
        params = entry.get("token_cost")
        if not isinstance(params, dict):
            continue
        pricing = params.get("pricing")
        if not isinstance(pricing, dict):
            continue
        model_pricing = pricing.get(model)
        if not isinstance(model_pricing, dict):
            continue
        return (
            model_pricing.get("input_per_1k"),
            model_pricing.get("output_per_1k"),
        )
    return None, None

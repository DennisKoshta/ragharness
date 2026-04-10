from __future__ import annotations

from typing import Any, Callable

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult
from ragharness.metrics.cost import token_cost
from ragharness.metrics.exact_match import exact_match
from ragharness.metrics.latency import latency_p50, latency_p95
from ragharness.metrics.retrieval import precision_at_k

PerQuestionMetric = Callable[[EvalItem, RAGResult], float]
AggregateMetric = Callable[..., float]

PER_QUESTION_REGISTRY: dict[str, PerQuestionMetric] = {
    "exact_match": exact_match,
    "precision_at_k": precision_at_k,
    # llm_judge is registered lazily — it needs config to instantiate
}

AGGREGATE_REGISTRY: dict[str, AggregateMetric] = {
    "latency_p50": latency_p50,
    "latency_p95": latency_p95,
    "token_cost": token_cost,
}


def get_per_question_metric(name: str, **kwargs: Any) -> PerQuestionMetric:
    """Look up a per-question metric by name.

    For ``llm_judge``, *kwargs* are forwarded to ``LLMJudge.__init__``.
    """
    if name == "llm_judge":
        from ragharness.metrics.llm_judge import LLMJudge

        return LLMJudge(**kwargs)

    if name not in PER_QUESTION_REGISTRY:
        raise ValueError(f"Unknown per-question metric: {name!r}")
    return PER_QUESTION_REGISTRY[name]


def get_aggregate_metric(name: str) -> AggregateMetric:
    """Look up an aggregate metric by name."""
    if name not in AGGREGATE_REGISTRY:
        raise ValueError(f"Unknown aggregate metric: {name!r}")
    return AGGREGATE_REGISTRY[name]

from __future__ import annotations

import numpy as np

from ragharness.protocol import RAGResult


def latency_p50(results: list[RAGResult]) -> float:
    """Median latency in milliseconds across all results."""
    if not results:
        return 0.0
    latencies = [r.metadata.get("latency_ms", 0.0) for r in results]
    return float(np.percentile(latencies, 50))


def latency_p95(results: list[RAGResult]) -> float:
    """95th percentile latency in milliseconds across all results."""
    if not results:
        return 0.0
    latencies = [r.metadata.get("latency_ms", 0.0) for r in results]
    return float(np.percentile(latencies, 95))

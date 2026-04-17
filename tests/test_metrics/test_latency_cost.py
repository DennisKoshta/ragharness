from rag_eval_kit.metrics.cost import token_cost
from rag_eval_kit.metrics.latency import latency_p50, latency_p95
from rag_eval_kit.protocol import RAGResult


def _results_with_latencies(latencies: list[float]) -> list[RAGResult]:
    return [RAGResult(answer="a", metadata={"latency_ms": lat}) for lat in latencies]


# ── latency ──────────────────────────────────────────────


def test_latency_p50():
    results = _results_with_latencies([100, 200, 300, 400, 500])
    assert latency_p50(results) == 300.0


def test_latency_p95():
    results = _results_with_latencies([100, 200, 300, 400, 500])
    assert latency_p95(results) == 480.0


def test_latency_empty():
    assert latency_p50([]) == 0.0
    assert latency_p95([]) == 0.0


# ── token cost ───────────────────────────────────────────


def test_token_cost_basic():
    pricing = {
        "gpt-4o": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    }
    results = [
        RAGResult(
            answer="a",
            metadata={"model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 200},
        ),
        RAGResult(
            answer="b",
            metadata={"model": "gpt-4o", "prompt_tokens": 500, "completion_tokens": 100},
        ),
    ]
    cost = token_cost(results, pricing=pricing)
    # (1000/1000)*0.003 + (200/1000)*0.015 + (500/1000)*0.003 + (100/1000)*0.015
    # = 0.003 + 0.003 + 0.0015 + 0.0015 = 0.009
    assert abs(cost - 0.009) < 1e-9


def test_token_cost_unknown_model():
    pricing = {"gpt-4o": {"input_per_1k": 0.003, "output_per_1k": 0.015}}
    results = [
        RAGResult(answer="a", metadata={"model": "unknown-model", "prompt_tokens": 1000}),
    ]
    assert token_cost(results, pricing=pricing) == 0.0


def test_token_cost_empty():
    assert token_cost([], pricing={}) == 0.0

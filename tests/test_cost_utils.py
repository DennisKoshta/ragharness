"""Token-counting + sweep cost estimator tests."""

from __future__ import annotations

import builtins

import pytest

from ragharness import cost_utils
from ragharness.cost_utils import (
    count_tokens,
    estimate_sweep_cost,
    resolve_model_from_config,
    resolve_pricing_from_metrics,
)
from ragharness.dataset import EvalDataset, EvalItem


def _reset_caches() -> None:
    cost_utils._encoding_cache.clear()
    cost_utils._tiktoken_warned = False


# ── count_tokens ────────────────────────────────────────


def test_count_tokens_empty_returns_zero():
    _reset_caches()
    assert count_tokens("") == 0


def test_count_tokens_fallback_when_tiktoken_missing(monkeypatch):
    """Simulate tiktoken not installed → uses len(text) // 4."""
    _reset_caches()
    real_import = builtins.__import__

    def _no_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)

    # "hello world!" is 12 chars → 12 // 4 = 3
    assert count_tokens("hello world!") == 3


def test_count_tokens_fallback_min_one(monkeypatch):
    """Short text still produces at least 1 token in the fallback path."""
    _reset_caches()
    real_import = builtins.__import__

    def _no_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated missing tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)
    assert count_tokens("a") >= 1


def test_count_tokens_with_tiktoken_if_available():
    """When tiktoken is installed, counts should be > 0 for real text."""
    pytest.importorskip("tiktoken")
    _reset_caches()
    n = count_tokens("The quick brown fox jumps over the lazy dog", model="gpt-4o")
    assert n > 0
    assert n < 50  # sanity: sentence shouldn't balloon


def test_count_tokens_unknown_model_falls_back_to_cl100k():
    """Unknown model name should not raise — tiktoken path uses cl100k_base."""
    pytest.importorskip("tiktoken")
    _reset_caches()
    n = count_tokens("hello", model="some-future-model")
    assert n > 0


# ── estimate_sweep_cost ─────────────────────────────────


def _mk_dataset(n: int) -> EvalDataset:
    return EvalDataset(
        [EvalItem(question=f"Question number {i} here", expected_answer=str(i)) for i in range(n)]
    )


def test_estimate_sweep_cost_empty_dataset_returns_zero():
    cost = estimate_sweep_cost(_mk_dataset(0), [{}], input_per_1k=0.001, output_per_1k=0.002)
    assert cost == 0.0


def test_estimate_sweep_cost_no_configs_returns_zero():
    cost = estimate_sweep_cost(_mk_dataset(5), [], input_per_1k=0.001, output_per_1k=0.002)
    assert cost == 0.0


def test_estimate_sweep_cost_scales_with_configs_and_questions():
    """Doubling either dimension should ~double the cost."""
    ds = _mk_dataset(5)
    c1 = estimate_sweep_cost(ds, [{}], input_per_1k=0.001, output_per_1k=0.002)
    c2 = estimate_sweep_cost(ds, [{}, {}], input_per_1k=0.001, output_per_1k=0.002)
    assert c2 == pytest.approx(2 * c1, rel=1e-6)

    c3 = estimate_sweep_cost(_mk_dataset(10), [{}], input_per_1k=0.001, output_per_1k=0.002)
    assert c3 == pytest.approx(2 * c1, rel=1e-6)


def test_estimate_sweep_cost_uses_default_pricing_for_known_model():
    """Without explicit rates, known models should produce nonzero cost."""
    cost = estimate_sweep_cost(_mk_dataset(3), [{}], model="gpt-4o")
    assert cost > 0


def test_estimate_sweep_cost_unknown_model_returns_zero_default_pricing():
    """Unknown model + no explicit rates → default 0 pricing → 0 cost."""
    cost = estimate_sweep_cost(_mk_dataset(3), [{}], model="completely-unknown-xyz")
    assert cost == 0.0


# ── resolve_model_from_config ───────────────────────────


def test_resolve_model_prefers_llm_model():
    assert resolve_model_from_config({"llm_model": "a", "model": "b"}) == "a"


def test_resolve_model_falls_back_to_model_key():
    assert resolve_model_from_config({"model": "b"}) == "b"


def test_resolve_model_default_when_missing():
    assert resolve_model_from_config({}, default="gpt-4o") == "gpt-4o"


# ── resolve_pricing_from_metrics ────────────────────────


def test_resolve_pricing_extracts_model_rates():
    metrics = [
        "exact_match",
        {
            "token_cost": {
                "pricing": {
                    "gpt-4o": {"input_per_1k": 0.01, "output_per_1k": 0.03},
                }
            }
        },
    ]
    ip, op = resolve_pricing_from_metrics(metrics, "gpt-4o")
    assert ip == 0.01
    assert op == 0.03


def test_resolve_pricing_returns_none_when_missing():
    ip, op = resolve_pricing_from_metrics(["exact_match"], "gpt-4o")
    assert ip is None
    assert op is None


def test_resolve_pricing_returns_none_for_unknown_model():
    metrics = [
        {"token_cost": {"pricing": {"gpt-4o": {"input_per_1k": 0.01, "output_per_1k": 0.03}}}},
    ]
    ip, op = resolve_pricing_from_metrics(metrics, "claude-something-else")
    assert ip is None
    assert op is None

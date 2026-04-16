"""Parallel execution + checkpoint integration for _run_single_config."""

from __future__ import annotations

import json
import threading
import time

import pytest

from ragharness.checkpoint import CheckpointWriter, load_checkpoint
from ragharness.config import RagHarnessConfig, SystemConfig
from ragharness.dataset import EvalDataset, EvalItem
from ragharness.metrics import get_per_question_metric
from ragharness.orchestrator import _run_single_config, run_sweep
from ragharness.protocol import RAGResult


class _BlockingAdapter:
    """Adapter that blocks every query until *n_concurrent* are in-flight.

    Used to prove real parallelism: a serial executor cannot reach
    ``n_concurrent`` in-flight calls and would deadlock, so the test
    asserts within a short timeout.
    """

    def __init__(self, n_concurrent: int, answer: str = "42") -> None:
        self.n_concurrent = n_concurrent
        self.answer = answer
        self._lock = threading.Lock()
        self._in_flight = 0
        self._ready = threading.Event()

    def query(self, question: str) -> RAGResult:
        with self._lock:
            self._in_flight += 1
            if self._in_flight >= self.n_concurrent:
                self._ready.set()
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError(f"Only {self._in_flight} in flight; expected {self.n_concurrent}")
        with self._lock:
            self._in_flight -= 1
        return RAGResult(
            answer=self.answer,
            retrieved_docs=[],
            metadata={"model": "dummy", "prompt_tokens": 1, "completion_tokens": 1},
        )


class _OrderedAdapter:
    """Adapter that records question order and returns a question-derived answer."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def query(self, question: str) -> RAGResult:
        time.sleep(0.01)
        with self._lock:
            self.seen.append(question)
        return RAGResult(
            answer=f"ans-{question}",
            retrieved_docs=[],
            metadata={"model": "dummy"},
        )


def _mk_dataset(n: int) -> EvalDataset:
    return EvalDataset([EvalItem(question=f"Q{i}", expected_answer=f"ans-Q{i}") for i in range(n)])


def _pq_metrics() -> dict:
    return {"exact_match": get_per_question_metric("exact_match")}


# ── Parallelism proof ───────────────────────────────────


def test_concurrency_actually_runs_in_parallel(monkeypatch):
    """With concurrency=4, four calls must be in-flight simultaneously."""
    adapter = _BlockingAdapter(n_concurrent=4)
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    run = _run_single_config(
        cfg_idx=0,
        n_configs=1,
        sweep_params={},
        dataset=_mk_dataset(8),
        system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
        pq_metrics=_pq_metrics(),
        agg_metrics={},
        verbose=False,
        concurrency=4,
    )
    assert len(run.raw_results) == 8


# ── Order preservation ─────────────────────────────────


def test_result_order_preserved_under_concurrency(monkeypatch):
    """Even with parallel execution, results must be in dataset order."""
    adapter = _OrderedAdapter()
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    run = _run_single_config(
        cfg_idx=0,
        n_configs=1,
        sweep_params={},
        dataset=_mk_dataset(10),
        system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
        pq_metrics=_pq_metrics(),
        agg_metrics={},
        verbose=False,
        concurrency=4,
    )
    answers = [r.answer for r in run.raw_results]
    assert answers == [f"ans-Q{i}" for i in range(10)]


# ── Serial path unchanged ──────────────────────────────


def test_concurrency_one_runs_serially(monkeypatch):
    """concurrency=1 must not spawn a ThreadPoolExecutor."""
    from tests.conftest import DummyRAGSystem

    adapter = DummyRAGSystem(answer="x")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    run = _run_single_config(
        cfg_idx=0,
        n_configs=1,
        sweep_params={},
        dataset=_mk_dataset(3),
        system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
        pq_metrics=_pq_metrics(),
        agg_metrics={},
        verbose=False,
        concurrency=1,
    )
    assert len(run.raw_results) == 3
    assert adapter.call_count == 3


# ── Checkpoint: writes on success ───────────────────────


def test_checkpoint_written_during_run(monkeypatch, tmp_path):
    from tests.conftest import DummyRAGSystem

    adapter = DummyRAGSystem(answer="42")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    ck_path = tmp_path / "ck.jsonl"
    with CheckpointWriter(ck_path) as writer:
        _run_single_config(
            cfg_idx=0,
            n_configs=1,
            sweep_params={"top_k": 3},
            dataset=_mk_dataset(5),
            system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
            pq_metrics=_pq_metrics(),
            agg_metrics={},
            verbose=False,
            concurrency=1,
            checkpoint_writer=writer,
            checkpoint_rows={},
        )

    rows = load_checkpoint(ck_path)
    assert len(rows) == 5
    for item_idx in range(5):
        assert (0, item_idx) in rows
        assert rows[(0, item_idx)]["config_params"] == {"top_k": 3}


# ── Checkpoint: resume skips done items ────────────────


def test_checkpoint_resume_skips_completed_items(monkeypatch, tmp_path):
    from tests.conftest import DummyRAGSystem

    adapter = DummyRAGSystem(answer="live")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    # Pretend items 0 and 2 are already done
    preloaded = {
        (0, 0): {
            "config_idx": 0,
            "item_idx": 0,
            "config_params": {},
            "answer": "cached-0",
            "retrieved_docs": [],
            "metadata": {"model": "cached"},
            "scores": {"exact_match": 1.0},
        },
        (0, 2): {
            "config_idx": 0,
            "item_idx": 2,
            "config_params": {},
            "answer": "cached-2",
            "retrieved_docs": [],
            "metadata": {"model": "cached"},
            "scores": {"exact_match": 0.0},
        },
    }

    run = _run_single_config(
        cfg_idx=0,
        n_configs=1,
        sweep_params={},
        dataset=_mk_dataset(4),
        system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
        pq_metrics=_pq_metrics(),
        agg_metrics={},
        verbose=False,
        concurrency=1,
        checkpoint_writer=None,
        checkpoint_rows=preloaded,
    )
    # Adapter should have been called for indices 1 and 3 only
    assert adapter.call_count == 2
    assert run.raw_results[0].answer == "cached-0"
    assert run.raw_results[1].answer == "live"
    assert run.raw_results[2].answer == "cached-2"
    assert run.raw_results[3].answer == "live"


def test_checkpoint_mismatched_config_params_triggers_rerun(monkeypatch, tmp_path, caplog):
    from tests.conftest import DummyRAGSystem

    adapter = DummyRAGSystem(answer="fresh")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    preloaded = {
        (0, 0): {
            "config_idx": 0,
            "item_idx": 0,
            "config_params": {"top_k": 3},
            "answer": "stale",
            "retrieved_docs": [],
            "metadata": {},
            "scores": {"exact_match": 1.0},
        }
    }
    import logging

    with caplog.at_level(logging.WARNING, logger="ragharness.orchestrator"):
        run = _run_single_config(
            cfg_idx=0,
            n_configs=1,
            sweep_params={"top_k": 5},  # differs from the cached row
            dataset=_mk_dataset(1),
            system_cfg=SystemConfig(adapter="raw", adapter_config={"llm_provider": "openai"}),
            pq_metrics=_pq_metrics(),
            agg_metrics={},
            verbose=False,
            concurrency=1,
            checkpoint_writer=None,
            checkpoint_rows=preloaded,
        )
    assert run.raw_results[0].answer == "fresh"
    assert any("mismatched" in rec.message for rec in caplog.records)


# ── Full run_sweep with checkpoint ──────────────────────


def test_run_sweep_with_checkpoint(monkeypatch, tmp_path):
    from tests.conftest import DummyRAGSystem

    adapter = DummyRAGSystem(answer="42")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter,
    )

    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(
        "\n".join(json.dumps({"question": f"Q{i}", "expected_answer": "42"}) for i in range(3))
    )
    ck_path = tmp_path / "ck.jsonl"

    cfg = RagHarnessConfig(
        dataset={"source": "jsonl", "path": str(ds_path)},
        system={"adapter": "raw", "adapter_config": {"llm_provider": "openai"}},
        metrics=["exact_match"],
        output={"checkpoint": str(ck_path)},
    )

    # First run: writes checkpoint
    run_sweep(cfg, no_confirm=True)
    assert adapter.call_count == 3
    rows_after_first = load_checkpoint(ck_path)
    assert len(rows_after_first) == 3

    # Second run: should skip all items
    adapter2 = DummyRAGSystem(answer="should-not-be-used")
    monkeypatch.setattr(
        "ragharness.orchestrator.create_adapter",
        lambda *a, **kw: adapter2,
    )
    result = run_sweep(cfg, no_confirm=True)
    assert adapter2.call_count == 0
    assert all(r.answer == "42" for r in result.runs[0].raw_results)


# ── Config validation ───────────────────────────────────


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="concurrency must be"):
        RagHarnessConfig(
            dataset={"source": "jsonl", "path": "x.jsonl"},
            system={"adapter": "raw"},
            concurrency=0,
        )

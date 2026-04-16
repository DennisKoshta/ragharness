"""Checkpoint writer / loader round-trip and concurrency tests."""

from __future__ import annotations

import json
import threading

from ragharness.checkpoint import (
    CheckpointWriter,
    load_checkpoint,
    row_to_result,
)
from ragharness.protocol import RAGResult


def _make_result(answer: str = "42", model: str = "gpt-4o") -> RAGResult:
    return RAGResult(
        answer=answer,
        retrieved_docs=["doc_a", "doc_b"],
        metadata={
            "latency_ms": 123.4,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "model": model,
        },
    )


# ── Round-trip ──────────────────────────────────────────


def test_write_and_load_single_row(tmp_path):
    path = tmp_path / "ck.jsonl"
    with CheckpointWriter(path) as w:
        w.write(
            config_idx=0,
            item_idx=3,
            config_params={"top_k": 5},
            result=_make_result(),
            scores={"exact_match": 1.0},
        )

    rows = load_checkpoint(path)
    assert (0, 3) in rows
    row = rows[(0, 3)]
    assert row["answer"] == "42"
    assert row["config_params"] == {"top_k": 5}
    assert row["scores"] == {"exact_match": 1.0}
    assert row["metadata"]["model"] == "gpt-4o"


def test_load_checkpoint_missing_file_returns_empty(tmp_path):
    assert load_checkpoint(tmp_path / "does-not-exist.jsonl") == {}


def test_load_checkpoint_skips_malformed_lines(tmp_path):
    path = tmp_path / "ck.jsonl"
    path.write_text(
        '{"config_idx": 0, "item_idx": 0, "answer": "ok"}\n'
        "not-valid-json\n"
        '{"config_idx": 0, "item_idx": 1, "answer": "also-ok"}\n'
    )
    rows = load_checkpoint(path)
    assert len(rows) == 2
    assert rows[(0, 0)]["answer"] == "ok"
    assert rows[(0, 1)]["answer"] == "also-ok"


def test_load_checkpoint_skips_rows_missing_indices(tmp_path):
    path = tmp_path / "ck.jsonl"
    path.write_text(
        '{"item_idx": 0, "answer": "no config_idx"}\n'
        '{"config_idx": 0, "item_idx": 0, "answer": "good"}\n'
    )
    rows = load_checkpoint(path)
    assert rows == {(0, 0): {"config_idx": 0, "item_idx": 0, "answer": "good"}}


# ── row_to_result ───────────────────────────────────────


def test_row_to_result_reconstruction():
    row = {
        "answer": "hello",
        "retrieved_docs": ["d1", "d2"],
        "metadata": {"latency_ms": 5.0},
    }
    result = row_to_result(row)
    assert result.answer == "hello"
    assert result.retrieved_docs == ["d1", "d2"]
    assert result.metadata == {"latency_ms": 5.0}


def test_row_to_result_handles_missing_fields():
    result = row_to_result({})
    assert result.answer == ""
    assert result.retrieved_docs == []
    assert result.metadata == {}


# ── Thread safety ───────────────────────────────────────


def test_concurrent_writes_produce_n_distinct_rows(tmp_path):
    """100 threads each writing one row → 100 parseable lines, no interleave."""
    path = tmp_path / "ck.jsonl"
    n_writes = 100

    with CheckpointWriter(path) as w:
        barrier = threading.Barrier(n_writes)

        def _writer(i: int) -> None:
            barrier.wait()
            w.write(
                config_idx=0,
                item_idx=i,
                config_params={"run": i},
                result=_make_result(answer=f"a{i}"),
                scores={"exact_match": float(i % 2)},
            )

        threads = [threading.Thread(target=_writer, args=(i,)) for i in range(n_writes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    with path.open() as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == n_writes
    parsed = [json.loads(line) for line in lines]
    assert {p["item_idx"] for p in parsed} == set(range(n_writes))


# ── Append semantics ────────────────────────────────────


def test_writer_appends_to_existing_file(tmp_path):
    path = tmp_path / "ck.jsonl"
    with CheckpointWriter(path) as w:
        w.write(
            config_idx=0,
            item_idx=0,
            config_params={},
            result=_make_result(),
            scores={},
        )

    with CheckpointWriter(path) as w:
        w.write(
            config_idx=0,
            item_idx=1,
            config_params={},
            result=_make_result(answer="second"),
            scores={},
        )

    rows = load_checkpoint(path)
    assert len(rows) == 2
    assert rows[(0, 0)]["answer"] == "42"
    assert rows[(0, 1)]["answer"] == "second"

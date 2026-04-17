from __future__ import annotations

import csv

from ragbench.dataset import EvalItem
from ragbench.orchestrator import RunResult, SweepResult
from ragbench.protocol import RAGResult
from ragbench.reporters.chart_reporter import write_charts
from ragbench.reporters.csv_reporter import write_csv


def _make_sweep_result() -> SweepResult:
    """Build a small SweepResult with 2 configs x 2 questions."""
    items = [
        EvalItem(question="Q1", expected_answer="A1"),
        EvalItem(question="Q2", expected_answer="A2"),
    ]

    def _run(params: dict[str, object], answers: list[str]) -> RunResult:
        results = [
            RAGResult(
                answer=ans,
                retrieved_docs=["doc1"],
                metadata={
                    "latency_ms": 100.0 + i * 50,
                    "model": "m",
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                },
            )
            for i, ans in enumerate(answers)
        ]
        per_q = [
            {"exact_match": 1.0 if ans == exp.expected_answer else 0.0}
            for ans, exp in zip(answers, items)
        ]
        agg = {
            "latency_p50": 100.0,
            "latency_p95": 150.0,
            "mean_exact_match": sum(s["exact_match"] for s in per_q) / len(per_q),
        }
        return RunResult(
            config_params=params,
            per_question_scores=per_q,
            aggregate_scores=agg,
            raw_results=results,
            items=items,
        )

    return SweepResult(
        runs=[
            _run({"top_k": 3}, ["A1", "A2"]),
            _run({"top_k": 5}, ["A1", "wrong"]),
        ]
    )


# ── CSV reporter ─────────────────────────────────────────


def test_write_csv_creates_files(tmp_path):
    result = _make_sweep_result()
    out = write_csv(result, tmp_path / "out")

    detail = out / "results_detail.csv"
    summary = out / "results_summary.csv"
    assert detail.exists()
    assert summary.exists()


def test_write_csv_detail_row_count(tmp_path):
    result = _make_sweep_result()
    out = write_csv(result, tmp_path / "out")

    with (out / "results_detail.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    # 2 configs * 2 questions = 4 rows
    assert len(rows) == 4


def test_write_csv_detail_columns(tmp_path):
    result = _make_sweep_result()
    out = write_csv(result, tmp_path / "out")

    with (out / "results_detail.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
    assert "top_k" in fields
    assert "question" in fields
    assert "exact_match" in fields
    assert "latency_ms" in fields


def test_write_csv_summary_row_count(tmp_path):
    result = _make_sweep_result()
    out = write_csv(result, tmp_path / "out")

    with (out / "results_summary.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    # 2 configs = 2 rows
    assert len(rows) == 2


def test_write_csv_empty_sweep(tmp_path):
    out = write_csv(SweepResult(), tmp_path / "out")
    assert out.exists()
    assert not (out / "results_detail.csv").exists()


# ── Chart reporter ───────────────────────────────────────


def test_write_charts_creates_pngs(tmp_path):
    result = _make_sweep_result()
    out = write_charts(result, tmp_path / "charts")

    pngs = list(out.glob("*.png"))
    assert len(pngs) >= 2  # at least accuracy + latency distribution


def test_write_charts_accuracy(tmp_path):
    result = _make_sweep_result()
    out = write_charts(result, tmp_path / "charts")

    assert (out / "chart_accuracy.png").exists()


def test_write_charts_latency(tmp_path):
    result = _make_sweep_result()
    out = write_charts(result, tmp_path / "charts")

    assert (out / "chart_latency_distribution.png").exists()


def test_write_charts_empty_sweep(tmp_path):
    out = write_charts(SweepResult(), tmp_path / "charts")
    assert out.exists()
    pngs = list(out.glob("*.png"))
    assert len(pngs) == 0

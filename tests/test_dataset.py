import json

import pytest

from ragharness.dataset import EvalDataset, EvalItem


def test_eval_item_defaults():
    item = EvalItem(question="What?", expected_answer="Yes")
    assert item.question == "What?"
    assert item.expected_answer == "Yes"
    assert item.expected_docs is None
    assert item.tags is None


def test_eval_item_with_all_fields():
    item = EvalItem(
        question="Capital of France?",
        expected_answer="Paris",
        expected_docs=["doc1"],
        tags={"category": "geography"},
    )
    assert item.expected_docs == ["doc1"]
    assert item.tags["category"] == "geography"


# ── JSONL loading ────────────────────────────────────────


def test_from_jsonl(tmp_path):
    data = [
        {"question": "Q1", "expected_answer": "A1"},
        {"question": "Q2", "expected_answer": "A2", "expected_docs": ["d1", "d2"]},
        {"question": "Q3", "expected_answer": "A3", "tags": {"topic": "science"}},
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in data))

    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 3
    assert ds[0].question == "Q1"
    assert ds[0].expected_docs is None
    assert ds[1].expected_docs == ["d1", "d2"]
    assert ds[2].tags == {"topic": "science"}


def test_from_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_answer": "A1"}\n'
        "\n"
        '{"question": "Q2", "expected_answer": "A2"}\n'
    )
    ds = EvalDataset.from_jsonl(path)
    assert len(ds) == 2


def test_from_jsonl_malformed_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "Q1", "expected_answer": "A1"}\nnot json\n')

    with pytest.raises(ValueError, match="line 2"):
        EvalDataset.from_jsonl(path)


# ── CSV loading ──────────────────────────────────────────


def test_from_csv(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text(
        "question,expected_answer,expected_docs,tags\n"
        'What is 1+1?,2,["doc1"],{"topic": "math"}\n'
        "Capital of France?,Paris,,\n"
    )
    ds = EvalDataset.from_csv(path)
    assert len(ds) == 2
    assert ds[0].question == "What is 1+1?"
    assert ds[0].expected_docs == ["doc1"]
    assert ds[0].tags == {"topic": "math"}
    assert ds[1].expected_docs is None
    assert ds[1].tags is None


# ── Iteration ────────────────────────────────────────────


def test_iteration(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question": "Q1", "expected_answer": "A1"}\n'
        '{"question": "Q2", "expected_answer": "A2"}\n'
    )
    ds = EvalDataset.from_jsonl(path)
    questions = [item.question for item in ds]
    assert questions == ["Q1", "Q2"]


# ── HuggingFace stub ────────────────────────────────────


def test_huggingface_not_implemented():
    with pytest.raises(NotImplementedError):
        EvalDataset.from_huggingface("hotpot_qa")

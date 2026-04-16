"""Shared fixtures used across the test suite."""

from __future__ import annotations

import json

import pytest

from ragharness.dataset import EvalItem
from ragharness.protocol import RAGResult

# ── Dummy RAG system ─────────────────────────────────────


class DummyRAGSystem:
    """A minimal RAGSystem implementation that returns fixed responses."""

    def __init__(self, answer: str = "42", docs: list[str] | None = None) -> None:
        self.answer = answer
        self.docs = docs or []
        self.call_count = 0

    def query(self, question: str) -> RAGResult:
        self.call_count += 1
        return RAGResult(
            answer=self.answer,
            retrieved_docs=self.docs,
            metadata={
                "latency_ms": 100.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "model": "dummy",
                "top_k": len(self.docs),
            },
        )


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    """Provide dummy API keys so check_api_key doesn't short-circuit tests.

    Real API calls are mocked out via DummyRAGSystem, so the values here
    never hit the wire — but the auth check runs unconditionally.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def dummy_rag_system() -> DummyRAGSystem:
    return DummyRAGSystem()


@pytest.fixture
def sample_eval_item() -> EvalItem:
    return EvalItem(
        question="What is the answer to life?",
        expected_answer="42",
        expected_docs=["doc_a", "doc_b"],
        tags={"topic": "philosophy"},
    )


@pytest.fixture
def sample_rag_result() -> RAGResult:
    return RAGResult(
        answer="42",
        retrieved_docs=["doc_a", "doc_c"],
        metadata={
            "latency_ms": 150.0,
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "model": "gpt-4o",
            "top_k": 5,
        },
    )


@pytest.fixture
def sample_dataset_path(tmp_path):
    """Creates a temporary 3-item JSONL dataset file and returns its path."""
    items = [
        {"question": "Q1", "expected_answer": "A1"},
        {"question": "Q2", "expected_answer": "A2", "expected_docs": ["d1"]},
        {"question": "Q3", "expected_answer": "A3"},
    ]
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in items))
    return path

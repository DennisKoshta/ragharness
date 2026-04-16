"""Implement the RAGSystem protocol and evaluate it with the metric registry.

Any object with a ``query(question) -> RAGResult`` method satisfies the protocol —
no inheritance required. The orchestrator's YAML path only wires the built-in
adapters (``raw``, ``langchain``, ...), so to evaluate a custom object we drive
the eval loop directly: iterate the dataset, call ``query``, score with the
metric registry.

Usage:
    python examples/custom_rag_system.py
"""

from __future__ import annotations

import time
from pathlib import Path

from ragharness import EvalDataset, RAGResult, RAGSystem
from ragharness.metrics import get_aggregate_metric, get_per_question_metric


class KeywordRAGSystem:
    """Toy RAG system: returns the first doc whose content overlaps the question."""

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus

    def query(self, question: str) -> RAGResult:
        start = time.perf_counter()
        q_words = {w.lower().strip("?.,") for w in question.split()}
        best_doc, best_overlap = "", 0
        for doc in self.corpus.values():
            overlap = len(q_words & {w.lower() for w in doc.split()})
            if overlap > best_overlap:
                best_doc, best_overlap = doc, overlap

        answer = best_doc.split(".")[0] if best_doc else "I don't know."
        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResult(
            answer=answer,
            retrieved_docs=[best_doc] if best_doc else [],
            metadata={
                "latency_ms": elapsed_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model": "keyword-baseline",
                "top_k": 1,
            },
        )


def main() -> None:
    system: RAGSystem = KeywordRAGSystem(
        corpus={
            "paris": "Paris is the capital of France.",
            "rome": "Rome is the capital of Italy.",
            "shakespeare": "William Shakespeare wrote Romeo and Juliet.",
        }
    )

    dataset = EvalDataset.from_jsonl(Path(__file__).parent / "sample_dataset.jsonl")

    exact_match = get_per_question_metric("exact_match")
    latency_p50 = get_aggregate_metric("latency_p50")

    results, scores = [], []
    for item in dataset:
        result = system.query(item.question)
        results.append(result)
        scores.append(exact_match(item, result))
        print(f"  Q: {item.question}")
        print(f"  A: {result.answer}  (expected: {item.expected_answer})  em={scores[-1]:.2f}")

    mean_em = sum(scores) / len(scores) if scores else 0.0
    print(f"\nmean_exact_match = {mean_em:.2f}")
    print(f"latency_p50_ms   = {latency_p50(results):.2f}")


if __name__ == "__main__":
    main()

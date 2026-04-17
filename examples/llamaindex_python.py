"""LlamaIndex adapter: pre-built index and pre-built query_engine paths.

LlamaIndex is Python-first, so the most natural use is to build an index
(or a full query engine) yourself and hand it to the adapter. This lets
you wire in custom embedders, retrievers, postprocessors, and response
synthesisers — anything LlamaIndex can express — then benchmark it.

Requires OPENAI_API_KEY (used by LlamaIndex's default embedder and LLM).

Usage:
    python examples/llamaindex_python.py
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from rag_eval_kit import EvalDataset
from rag_eval_kit.adapters.llamaindex import LlamaIndexRAGSystem
from rag_eval_kit.metrics import get_per_question_metric

SAMPLE_CORPUS = [
    "Paris is the capital of France.",
    "William Shakespeare wrote Romeo and Juliet.",
    "The speed of light in a vacuum is 299,792,458 meters per second.",
    "World War II ended in 1945.",
    "The chemical symbol for gold is Au.",
    "Leonardo da Vinci painted the Mona Lisa.",
    "Jupiter is the largest planet in our solar system.",
    "Python was created by Guido van Rossum.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Albert Einstein developed general relativity.",
]


def index_path() -> None:
    """Pass a pre-built index; adapter handles query-engine construction per sweep."""
    documents = [Document(text=t) for t in SAMPLE_CORPUS]
    index = VectorStoreIndex.from_documents(documents)

    system = LlamaIndexRAGSystem(
        index=index,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        top_k=3,
    )
    _evaluate(system, label="pre-built index")


def query_engine_path() -> None:
    """Pass a fully-configured query engine; adapter just drives it."""
    documents = [Document(text=t) for t in SAMPLE_CORPUS]
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        llm=OpenAI(model="gpt-4o-mini", temperature=0.0),
    )

    system = LlamaIndexRAGSystem(query_engine=query_engine)
    _evaluate(system, label="pre-built query_engine")


def _evaluate(system: LlamaIndexRAGSystem, *, label: str) -> None:
    dataset = EvalDataset.from_jsonl(Path(__file__).parent / "sample_dataset.jsonl")
    exact_match = get_per_question_metric("exact_match")

    print(f"\n--- {label} ---")
    for item in list(dataset)[:3]:
        result = system.query(item.question)
        em = exact_match(item, result)
        print(f"  Q: {item.question}")
        print(f"  A: {result.answer!r}  (expected: {item.expected_answer!r})  em={em:.2f}")


def main() -> None:
    index_path()
    query_engine_path()


if __name__ == "__main__":
    main()

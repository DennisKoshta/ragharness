"""Build an LCEL chain in Python and hand it to LangChainRAGSystem.

The YAML adapter path constructs the chain internally from ``llm_provider`` /
``llm_model`` / ``retriever``. When you need full control — custom prompts,
multi-step chains, structured output, whatever LCEL can express — build the
``Runnable`` yourself and pass it via ``chain=``. The adapter will invoke it
and normalise the output into a ``RAGResult``.

Requires an API key: ``ANTHROPIC_API_KEY`` (or swap ChatAnthropic for ChatOpenAI).

Usage:
    python examples/langchain_runnable.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ragbench import EvalDataset
from ragbench.adapters.langchain import LangChainRAGSystem
from ragbench.metrics import get_per_question_metric


def build_chain() -> object:
    """An LCEL chain: prompt → LLM → string output."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer in one short phrase. No full sentences, no punctuation."),
            ("human", "{question}"),
        ]
    )
    llm = ChatAnthropic(
        model_name="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens_to_sample=256,
        timeout=None,
        stop=None,
    )
    return prompt | llm | StrOutputParser()


def main() -> None:
    chain = build_chain()
    system = LangChainRAGSystem(chain=chain)

    dataset = EvalDataset.from_jsonl(Path(__file__).parent / "sample_dataset.jsonl")
    exact_match = get_per_question_metric("exact_match")

    for item in dataset:
        result = system.query(item.question)
        em = exact_match(item, result)
        print(f"  Q: {item.question}")
        print(f"  A: {result.answer!r}  (expected: {item.expected_answer!r})  em={em:.2f}")


if __name__ == "__main__":
    main()

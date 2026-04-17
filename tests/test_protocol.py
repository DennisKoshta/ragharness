from ragbench.protocol import RAGResult, RAGSystem


class _ValidSystem:
    def query(self, question: str) -> RAGResult:
        return RAGResult(answer="test")


class _InvalidSystem:
    def not_query(self, question: str) -> str:
        return "nope"


def test_rag_result_defaults():
    result = RAGResult(answer="hello")
    assert result.answer == "hello"
    assert result.retrieved_docs == []
    assert result.metadata == {}


def test_rag_result_with_all_fields():
    result = RAGResult(
        answer="Paris",
        retrieved_docs=["doc1", "doc2"],
        metadata={"latency_ms": 150.0, "model": "gpt-4o"},
    )
    assert result.answer == "Paris"
    assert len(result.retrieved_docs) == 2
    assert result.metadata["latency_ms"] == 150.0


def test_protocol_conformance():
    system = _ValidSystem()
    assert isinstance(system, RAGSystem)


def test_protocol_non_conformance():
    obj = _InvalidSystem()
    assert not isinstance(obj, RAGSystem)

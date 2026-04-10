from ragharness.dataset import EvalItem
from ragharness.metrics.retrieval import precision_at_k
from ragharness.protocol import RAGResult


def _item(expected_docs: list[str] | None) -> EvalItem:
    return EvalItem(question="q", expected_answer="a", expected_docs=expected_docs)


def _result(docs: list[str]) -> RAGResult:
    return RAGResult(answer="a", retrieved_docs=docs)


def test_perfect_retrieval():
    assert precision_at_k(_item(["a", "b"]), _result(["a", "b"]), k=2) == 1.0


def test_partial_overlap():
    assert precision_at_k(_item(["a", "b"]), _result(["a", "c"]), k=2) == 0.5


def test_no_overlap():
    assert precision_at_k(_item(["a"]), _result(["b", "c"]), k=2) == 0.0


def test_empty_retrieved_docs():
    assert precision_at_k(_item(["a"]), _result([]), k=5) == 0.0


def test_no_expected_docs():
    assert precision_at_k(_item(None), _result(["a", "b"]), k=5) == 0.0


def test_k_limits_docs_considered():
    # 3 retrieved, but k=2 means only first 2 are checked
    item = _item(["a", "b", "c"])
    result = _result(["a", "x", "c"])
    assert precision_at_k(item, result, k=2) == 0.5  # 1 hit out of 2

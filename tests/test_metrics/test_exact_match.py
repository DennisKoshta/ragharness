from rag_eval_kit.dataset import EvalItem
from rag_eval_kit.metrics.exact_match import exact_match
from rag_eval_kit.protocol import RAGResult


def _item(expected: str) -> EvalItem:
    return EvalItem(question="q", expected_answer=expected)


def _result(answer: str) -> RAGResult:
    return RAGResult(answer=answer)


def test_exact_match():
    assert exact_match(_item("Paris"), _result("Paris")) == 1.0


def test_case_insensitive():
    assert exact_match(_item("paris"), _result("PARIS")) == 1.0


def test_whitespace_stripped():
    assert exact_match(_item("  Paris "), _result("Paris  ")) == 1.0


def test_non_match():
    assert exact_match(_item("Paris"), _result("London")) == 0.0


def test_empty_answer():
    assert exact_match(_item("Paris"), _result("")) == 0.0

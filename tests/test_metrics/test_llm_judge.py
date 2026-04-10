import math
from unittest.mock import MagicMock

import pytest

from ragharness.dataset import EvalItem
from ragharness.metrics.llm_judge import LLMJudge
from ragharness.protocol import RAGResult


def _item() -> EvalItem:
    return EvalItem(question="Capital of France?", expected_answer="Paris")


def _result() -> RAGResult:
    return RAGResult(answer="Paris")


def test_parses_valid_json_score(monkeypatch):
    judge = LLMJudge(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": 0.85, "reasoning": "good"}')

    score = judge(_item(), _result())
    assert score == 0.85


def test_clamps_score_above_one(monkeypatch):
    judge = LLMJudge(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": 1.5, "reasoning": "x"}')

    assert judge(_item(), _result()) == 1.0


def test_clamps_score_below_zero(monkeypatch):
    judge = LLMJudge(provider="openai")
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: '{"score": -0.3, "reasoning": "x"}')

    assert judge(_item(), _result()) == 0.0


def test_retries_on_bad_json(monkeypatch):
    call_count = 0

    def _fake_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return "not json"
        return '{"score": 0.7, "reasoning": "ok"}'

    judge = LLMJudge(provider="openai", max_retries=2)
    monkeypatch.setattr(judge, "_call_llm", _fake_llm)

    score = judge(_item(), _result())
    assert score == 0.7
    assert call_count == 3  # 1 initial + 2 retries


def test_returns_nan_after_exhausted_retries(monkeypatch):
    judge = LLMJudge(provider="openai", max_retries=2)
    monkeypatch.setattr(judge, "_call_llm", lambda prompt: "garbage")

    score = judge(_item(), _result())
    assert math.isnan(score)


def test_unknown_provider_raises():
    judge = LLMJudge(provider="fake_provider")
    with pytest.raises(ValueError, match="Unknown LLM judge provider"):
        judge._get_client()

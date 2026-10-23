import json
import time

import pytest

from llm_pipeline import chain
from llm_pipeline.errors import LLMAPIError
from llm_pipeline.prompts import MEANING_PROMPT
from llm_pipeline.schemas import Meaning
from tests.test_chain import RESPONSES, install_fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def raising_fake(monkeypatch, retryable):
    calls = []

    def fake(system, user):
        calls.append(system)
        raise LLMAPIError("сбой API", retryable=retryable)

    monkeypatch.setattr(chain, "call_llm", fake)
    return calls


def test_api_down_retried_then_chain_stops(monkeypatch):
    calls = raising_fake(monkeypatch, retryable=True)
    state = chain.run_chain("текст")
    assert state.failed_step == "extract_meaning"
    assert len(calls) == 3


def test_fatal_api_error_not_retried(monkeypatch):
    calls = raising_fake(monkeypatch, retryable=False)
    state = chain.run_chain("текст")
    assert state.failed_step == "extract_meaning"
    assert len(calls) == 1


def test_text_instead_of_json_stops_after_repair(monkeypatch):
    calls = install_fake(monkeypatch, {**RESPONSES, MEANING_PROMPT: "Конечно! Вот ответ."})
    state = chain.run_chain("текст")
    assert state.failed_step == "extract_meaning"
    assert len(calls) == 2


def test_missing_keys_repaired_on_second_try(monkeypatch):
    good = json.dumps({"summary": "s", "key_points": ["a", "b", "c"], "language": "Russian"})
    replies = iter(['{"summary": "s"}', good])
    monkeypatch.setattr(chain, "call_llm", lambda system, user: next(replies))
    result = chain.run_step("extract_meaning", "SYSTEM", "user", Meaning)
    assert result.language == "Russian"


def test_empty_answer_gives_fallback(monkeypatch):
    monkeypatch.setattr(chain, "call_llm", lambda system, user: RESPONSES.get(system, ""))
    state = chain.run_chain("текст")
    assert state.failed_step is None
    assert "пустой" in state.fallback_reason
    assert state.passed is False


def test_too_long_answer_gives_fallback(monkeypatch):
    long_answer = "а" * (chain.MAX_ANSWER_CHARS + 1)
    monkeypatch.setattr(chain, "call_llm", lambda system, user: RESPONSES.get(system, long_answer))
    state = chain.run_chain("текст")
    assert "слишком длинный" in state.fallback_reason


def test_empty_input_makes_no_model_calls(monkeypatch):
    calls = install_fake(monkeypatch, RESPONSES)
    state = chain.run_chain("   ")
    assert state.failed_step == "check_input"
    assert calls == []

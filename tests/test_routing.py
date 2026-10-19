import json
from typing import get_args

import pytest

from llm_pipeline import pipeline
from llm_pipeline.prompts import CATEGORY_DESCRIPTIONS
from llm_pipeline.routes import ROUTES, get_route_prompt
from llm_pipeline.schemas import Category

CLASSIFICATION = {
    "summary": "s",
    "category": "complaint",
    "intent": "i",
    "sentiment": "negative",
    "key_points": ["a", "b", "c"],
}


def test_every_category_has_route():
    assert set(get_args(Category)) == set(ROUTES)


def test_every_category_has_description():
    assert set(get_args(Category)) == set(CATEGORY_DESCRIPTIONS)


def test_unknown_category_falls_back_to_other():
    assert get_route_prompt("нет_такой") == get_route_prompt("other")


def test_empty_answer_raises(monkeypatch):
    replies = iter([json.dumps(CLASSIFICATION), "   "])
    monkeypatch.setattr(pipeline, "call_llm", lambda system, user: next(replies))
    with pytest.raises(pipeline.LLMOutputError, match="пустой ответ"):
        pipeline.process_text("любой текст")


def test_broken_classification_skips_answer(monkeypatch):
    calls = []

    def fake_llm(system, user):
        calls.append(system)
        return "не json"

    monkeypatch.setattr(pipeline, "call_llm", fake_llm)
    with pytest.raises(pipeline.LLMOutputError):
        pipeline.process_text("любой текст")
    assert len(calls) == 1

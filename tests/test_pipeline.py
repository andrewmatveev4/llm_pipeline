import json

import pytest

from llm_pipeline import pipeline
from llm_pipeline.routes import get_route_prompt


def test_process_text_routes_by_category(monkeypatch):
    classification = {
        "summary": "s",
        "category": "complaint",
        "intent": "i",
        "sentiment": "negative",
        "key_points": ["a", "b", "c"],
    }
    calls = []

    def fake_llm(system, user):
        calls.append(system)
        return json.dumps(classification) if len(calls) == 1 else "Извините за неудобства."

    monkeypatch.setattr(pipeline, "call_llm", fake_llm)
    result = pipeline.process_text("любой текст")

    assert result.route == "complaint"
    assert result.final_answer == "Извините за неудобства."
    assert calls[1] == get_route_prompt("complaint")


def test_process_text_raises_clear_error_on_markdown(monkeypatch):
    raw = "```json\n{}\n```"
    monkeypatch.setattr(pipeline, "call_llm", lambda system, user: raw)
    with pytest.raises(pipeline.LLMOutputError) as exc:
        pipeline.process_text("любой текст")
    assert "невалидный JSON" in str(exc.value)
    assert exc.value.raw == raw

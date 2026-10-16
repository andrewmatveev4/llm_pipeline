import json

import pytest

from llm_pipeline import pipeline


def test_process_text_parses_json(monkeypatch):
    fake = {
        "summary": "s",
        "category": "question",
        "sentiment": "neutral",
        "key_points": ["a", "b", "c"],
        "final_answer": "r",
    }
    monkeypatch.setattr(pipeline, "call_llm", lambda system, user: json.dumps(fake))
    assert pipeline.process_text("любой текст") == fake


def test_process_text_fails_on_markdown_json(monkeypatch):
    monkeypatch.setattr(pipeline, "call_llm", lambda system, user: "```json\n{}\n```")
    with pytest.raises(json.JSONDecodeError):
        pipeline.process_text("любой текст")

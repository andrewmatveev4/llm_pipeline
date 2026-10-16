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
    assert pipeline.process_text("любой текст").model_dump() == fake


def test_process_text_raises_clear_error_on_markdown(monkeypatch):
    raw = "```json\n{}\n```"
    monkeypatch.setattr(pipeline, "call_llm", lambda system, user: raw)
    with pytest.raises(pipeline.LLMOutputError) as exc:
        pipeline.process_text("любой текст")
    assert "невалидный JSON" in str(exc.value)
    assert exc.value.raw == raw

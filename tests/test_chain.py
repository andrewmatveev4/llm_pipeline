import json

from llm_pipeline import chain
from llm_pipeline.chain import REPAIR_HINT
from llm_pipeline.prompts import (
    DETAILS_PROMPT, FINAL_RULES, MEANING_PROMPT, REQUEST_PROMPT, SELF_CHECK_PROMPT,
)
from llm_pipeline.routes import get_route_prompt

RESPONSES = {
    MEANING_PROMPT: json.dumps({
        "summary": "s", "key_points": ["a", "b", "c"], "language": "Russian",
    }),
    REQUEST_PROMPT: json.dumps({
        "category": "complaint", "intent": "i", "sentiment": "negative",
    }),
    DETAILS_PROMPT: json.dumps({
        "product": "доставка", "problem": "p", "requested_action": "r",
        "urgency": "high", "facts": ["f"], "missing_info": [],
    }),
    SELF_CHECK_PROMPT: json.dumps({"contradictions": [], "lost_details": []}),
}


def install_fake(monkeypatch, responses):
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        return responses.get(system, "Готовый ответ.")

    monkeypatch.setattr(chain, "call_llm", fake_llm)
    return calls


def test_chain_runs_all_steps(monkeypatch):
    calls = install_fake(monkeypatch, RESPONSES)
    state = chain.run_chain("текст")
    assert state.failed_step is None
    assert state.answer == "Готовый ответ."
    assert state.passed is True
    assert len(calls) == 5


def test_answer_uses_route_and_previous_fields(monkeypatch):
    calls = install_fake(monkeypatch, RESPONSES)
    chain.run_chain("текст")
    system, user = calls[3]
    assert system == get_route_prompt("complaint") + FINAL_RULES
    assert "Reply in Russian" in user
    assert "Product: доставка" in user


def test_findings_make_verdict_false(monkeypatch):
    responses = {**RESPONSES, SELF_CHECK_PROMPT: json.dumps(
        {"contradictions": [], "lost_details": ["пропущен факт"]})}
    install_fake(monkeypatch, responses)
    state = chain.run_chain("текст")
    assert state.passed is False
    assert state.check.lost_details == ["пропущен факт"]


def test_broken_step_stops_chain(monkeypatch):
    responses = {**RESPONSES, REQUEST_PROMPT: "не json"}
    calls = install_fake(monkeypatch, responses)
    state = chain.run_chain("текст")
    assert state.failed_step == "classify_request"
    assert state.meaning is not None
    assert state.request is None
    assert len(calls) == 3  # шаг 1 + шаг 2 дважды: основной вызов и повтор с подсказкой
    assert REPAIR_HINT in calls[2][0]

import logging

from pydantic import BaseModel

from llm_pipeline.llm_client import call_llm
from llm_pipeline.pipeline import LLMOutputError, parse_llm_output
from llm_pipeline.prompts import (
    DETAILS_PROMPT,
    FINAL_RULES,
    MEANING_PROMPT,
    REQUEST_PROMPT,
    SELF_CHECK_PROMPT,
    build_check_prompt,
    build_details_prompt,
    build_final_prompt,
    build_request_prompt,
    build_user_prompt,
)
from llm_pipeline.routes import get_route_prompt
from llm_pipeline.schemas import ChainState, Details, Meaning, RequestType, SelfCheck

logger = logging.getLogger(__name__)
MAX_INPUT_CHARS = 4000


def run_step(name: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    logger.info("[%s] старт", name)
    result = parse_llm_output(call_llm(system, user), schema)
    logger.info("[%s] результат: %s", name, result.model_dump_json())
    return result


def extract_meaning(state: ChainState) -> None:
    user = build_user_prompt(state.text)
    state.meaning = run_step("extract_meaning", MEANING_PROMPT, user, Meaning)


def classify_request(state: ChainState) -> None:
    user = build_request_prompt(state.text, state.meaning.summary, state.meaning.language)
    state.request = run_step("classify_request", REQUEST_PROMPT, user, RequestType)


def build_fields(state: ChainState) -> None:
    user = build_details_prompt(state.text, state.meaning, state.request)
    state.details = run_step("build_fields", DETAILS_PROMPT, user, Details)


def write_answer(state: ChainState) -> None:
    route = state.request.category
    logger.info("[write_answer] старт, маршрут: %s", route)
    system = get_route_prompt(route) + FINAL_RULES
    user = build_final_prompt(state.text, state.meaning, state.request, state.details)
    answer = call_llm(system, user).strip()
    if not answer:
        raise LLMOutputError("Модель вернула пустой ответ на шаге write_answer", answer)
    state.answer = answer
    logger.info("[write_answer] результат: %s", answer)


def self_check(state: ChainState) -> None:
    user = build_check_prompt(state.text, state.details, state.answer)
    state.check = run_step("self_check", SELF_CHECK_PROMPT, user,   SelfCheck)
    state.passed = not state.check.contradictions and not state.check.lost_details
    logger.info("[self_check] решение: %s", "OK" if state.passed else "НУЖНА ПРОВЕРКА")


STEPS = [extract_meaning, classify_request, build_fields, write_answer, self_check]


def check_input(text: str) -> str | None:
    if not text.strip():
        return "Пустой текст: нечего обрабатывать"
    if len(text) > MAX_INPUT_CHARS:
        return f"Текст слишком длинный: {len(text)} символов, максимум {MAX_INPUT_CHARS}"
    return None


def run_chain(text: str) -> ChainState:
    state = ChainState(text=text)
    problem = check_input(text)
    if problem:
        state.failed_step = "check_input"
        state.error = problem
        logger.warning("[check_input] %s", problem)
        return state
    for step in STEPS:
        try:
            step(state)
        except LLMOutputError as e:
            state.failed_step = step.__name__
            state.error = str(e)
            logger.error("[%s] ошибка, цепочка остановлена: %s", step.__name__, e)
            break
    return state

import logging

from pydantic import BaseModel

from llm_pipeline.llm_client import call_llm
from llm_pipeline.pipeline import parse_llm_output

from llm_pipeline.prompts import (
    MEANING_PROMPT, REQUEST_PROMPT, build_request_prompt, build_user_prompt,
)
from llm_pipeline.schemas import ChainState, Meaning, RequestType

logger = logging.getLogger(__name__)


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

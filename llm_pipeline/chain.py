import logging

from pydantic import BaseModel

from llm_pipeline.llm_client import call_llm
from llm_pipeline.pipeline import parse_llm_output

logger = logging.getLogger(__name__)


def run_step(name: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    logger.info("[%s] старт", name)
    result = parse_llm_output(call_llm(system, user), schema)
    logger.info("[%s] результат: %s", name, result.model_dump_json())
    return result

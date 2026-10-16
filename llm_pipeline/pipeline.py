import json


from pydantic import ValidationError
from llm_pipeline.schemas import LLMOutput
from llm_pipeline.llm_client import call_llm
from llm_pipeline.prompts import SYSTEM_PROMPT, build_user_prompt


def process_text(text: str, system_prompt: str = SYSTEM_PROMPT) -> LLMOutput:
    raw = call_llm(system_prompt, build_user_prompt(text))
    return parse_llm_output(raw)


class LLMOutputError(Exception):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def describe_errors(e: ValidationError) -> str:
    errors = e.errors()
    if errors[0]["type"] == "json_invalid":
        return f"Модель вернула невалидный JSON: {errors[0]['msg']}"
    parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in errors]
    return "Ответ модели не соответствует схеме: " + "; ".join(parts)


def parse_llm_output(raw: str) -> LLMOutput:
    try:
        return LLMOutput.model_validate_json(raw)
    except ValidationError as e:
        raise LLMOutputError(describe_errors(e), raw) from e

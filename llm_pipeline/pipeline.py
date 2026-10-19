from pydantic import ValidationError

from llm_pipeline.llm_client import call_llm
from llm_pipeline.prompts import SYSTEM_PROMPT, build_answer_prompt, build_user_prompt
from llm_pipeline.routes import get_route_prompt
from llm_pipeline.schemas import Classification, PipelineResult



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


def parse_llm_output(raw: str) -> Classification:
    try:
        return Classification.model_validate_json(raw)
    except ValidationError as e:
        raise LLMOutputError(describe_errors(e), raw) from e


def classify(text: str, system_prompt: str = SYSTEM_PROMPT) -> Classification:
    raw = call_llm(system_prompt, build_user_prompt(text))
    return parse_llm_output(raw)


def generate_answer(text: str, classification: Classification) -> str:
    route_prompt = get_route_prompt(classification.category)
    answer = call_llm(route_prompt, build_answer_prompt(text, classification.intent)).strip()
    if not answer:
        raise LLMOutputError("Модель вернула пустой ответ на шаге генерации", answer)
    return answer


def process_text(text: str, system_prompt: str = SYSTEM_PROMPT) -> PipelineResult:
    classification = classify(text, system_prompt)
    answer = generate_answer(text, classification)
    return PipelineResult(
        **classification.model_dump(),
        route=classification.category,
        final_answer=answer,
    )

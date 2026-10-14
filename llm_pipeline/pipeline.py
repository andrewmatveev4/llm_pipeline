import json

from llm_pipeline.llm_client import call_llm
from llm_pipeline.prompts import SYSTEM_PROMPT, build_user_prompt


def process_text(text: str) -> dict:
    raw = call_llm(SYSTEM_PROMPT, build_user_prompt(text))
    return json.loads(raw)

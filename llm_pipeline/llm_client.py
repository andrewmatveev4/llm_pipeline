import logging
import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI
from llm_pipeline.errors import ConfigError, LLMAPIError, LLMOutputError

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("OPENAI_API_KEY не задан: заполни .env по образцу .env.example")
    return OpenAI(api_key=api_key, max_retries=0, timeout=30)


def call_llm(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1000) -> str:
    client = get_client()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise ConfigError("LLM_MODEL не задан: заполни .env по образцу .env.example")
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except openai.RateLimitError as e:
        if e.code == "insufficient_quota":
            raise LLMAPIError("Закончился баланс API (insufficient_quota)", retryable=False) from e
        raise LLMAPIError(f"Слишком много запросов: {e}", retryable=True) from e
    except (openai.APIConnectionError, openai.InternalServerError) as e:
        raise LLMAPIError(f"Временная ошибка API: {e}", retryable=True) from e
    except openai.APIError as e:
        raise LLMAPIError(f"Ошибка API: {e}", retryable=False) from e
    elapsed = time.perf_counter() - start
    logger.debug(
        "LLM call: %s, %.1f s, tokens in=%s out=%s",
        model, elapsed, response.usage.prompt_tokens, response.usage.completion_tokens,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise LLMOutputError(f"Ответ обрезан: модель упёрлась в лимит {max_tokens} токенов",
                             choice.message.content or "")
    return choice.message.content or ""

import os

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан: заполни .env по образцу .env.example")
    return OpenAI(api_key=api_key)


def call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    client = get_client()
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL не задан: заполни .env по образцу .env.example")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""
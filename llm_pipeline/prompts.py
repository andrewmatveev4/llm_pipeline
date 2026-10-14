SYSTEM_PROMPT = """Ты помощник, который обрабатывает текст пользователя.
Верни ТОЛЬКО JSON-объект, без пояснений и без markdown, строго с такими ключами:
{
  "summary": "краткое содержание, 1-2 предложения",
  "key_points": ["мысль 1", "мысль 2", "мысль 3"],
  "response": "короткий полезный ответ пользователю, 2-4 предложения"
}
В key_points ровно 3 элемента.
Отвечай на языке исходного текста."""

USER_TEMPLATE = "Обработай текст между тегами <text>.\n<text>\n{text}\n</text>"


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text)
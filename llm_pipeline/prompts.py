SYSTEM_PROMPT = """You process a user's text.
Return ONLY a JSON object, no explanations and no markdown, with exactly these keys:
{
  "summary": "<short summary, 1-2 sentences>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"],
  "response": "<short helpful reply to the user, 2-4 sentences>"
}
key_points must contain exactly 3 items.
Language: detect the language of the text inside the <text> tags and write ALL values
in that language."""

USER_TEMPLATE = "Process the text inside the <text> tags.\n<text>\n{text}\n</text>"


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text)
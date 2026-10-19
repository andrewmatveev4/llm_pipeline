from typing import get_args

from llm_pipeline.schemas import Category, Sentiment

ALLOWED_VALUES = (
    "\ncategory must be one of: " + ", ".join(get_args(Category)) + "."
    + "\nsentiment must be one of: " + ", ".join(get_args(Sentiment)) + "."
)

BASE_PROMPT = """You process a user's text.
Return ONLY a JSON object, no explanations and no markdown, with exactly these keys:
{
  "summary": "<short summary>",
  "category": "<one category>",
  "intent": "<what exactly the user wants, 3-8 words>",
  "sentiment": "<one sentiment>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"]
}
key_points must contain exactly 3 items.""" + ALLOWED_VALUES + """
Language: detect the language of the text inside the <text> tags and write summary,
intent and key_points in that language. category and sentiment are always in English."""

LIMITS = """
Length limits:
- summary: one sentence, at most 25 words.
- each key point: at most 12 words."""

EXAMPLE = """
Example.
Input: <text>The office printer jams daily and the IT ticket has been open for a week.</text>
Output:
{
  "summary": "The office printer jams daily and the IT ticket is still unresolved.",
  "category": "complaint",
  "intent": "get the printer fixed",
  "sentiment": "negative",
  "key_points": [
    "The printer jams every morning.",
    "An IT ticket was opened a week ago.",
    "The issue is still not fixed."
  ]
}"""

PROMPT_VARIANTS = {
    "v1_baseline": BASE_PROMPT,
    "v2_limits": BASE_PROMPT + LIMITS,
    "v3_example": BASE_PROMPT + LIMITS + EXAMPLE,
}


ANSWER_TEMPLATE = (
    "Reply to the user's message inside the <text> tags.\n"
    "User intent: {intent}\n<text>\n{text}\n</text>"
)


SYSTEM_PROMPT = PROMPT_VARIANTS["v2_limits"]

USER_TEMPLATE = "Process the text inside the <text> tags.\n<text>\n{text}\n</text>"


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text)


def build_answer_prompt(text: str, intent: str) -> str:
    return ANSWER_TEMPLATE.format(text=text, intent=intent)

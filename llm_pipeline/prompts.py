BASE_PROMPT = """You process a user's text.
Return ONLY a JSON object, no explanations and no markdown, with exactly these keys:
{
  "summary": "<short summary>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"],
  "response": "<short helpful reply to the user>"
}
key_points must contain exactly 3 items.
Language: detect the language of the text inside the <text> tags and write ALL values
in that language."""

LIMITS = """
Length limits:
- summary: one sentence, at most 25 words.
- each key point: at most 12 words.
- response: at most 3 sentences."""

EXAMPLE = """
Example.
Input: <text>The office printer jams daily and the IT ticket has been open for a week.</text>
Output:
{
  "summary": "The office printer jams daily and the IT ticket is still unresolved.",
  "key_points": [
    "The printer jams every morning.",
    "An IT ticket was opened a week ago.",
    "The issue is still not fixed."
  ],
  "response": "Escalate the ticket to your IT lead and mention the daily impact."
}"""

PROMPT_VARIANTS = {
    "v1_baseline": BASE_PROMPT,
    "v2_limits": BASE_PROMPT + LIMITS,
    "v3_example": BASE_PROMPT + LIMITS + EXAMPLE,
}

SYSTEM_PROMPT = PROMPT_VARIANTS["v2_limits"]

USER_TEMPLATE = "Process the text inside the <text> tags.\n<text>\n{text}\n</text>"


def build_user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(text=text)

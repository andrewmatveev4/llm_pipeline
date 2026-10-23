import re
from llm_pipeline.prompts import POINT_MAX_WORDS, SUMMARY_MAX_WORDS


def count_words(s: str) -> int:
    return len(s.split())


def count_sentences(s: str) -> int:
    parts = re.split(r"[.!?]+", s)
    return len([p for p in parts if p.strip()])


def has_cyrillic(s: str) -> bool:
    return re.search(r"[а-яёА-ЯЁ]", s) is not None


RESPONSE_MAX_SENTENCES = 3
REQUIRED_KEYS = ("summary", "key_points", "final_answer")


def check_output(output: dict, text: str) -> dict:
    if not all(k in output for k in REQUIRED_KEYS):
        return {"has_keys": False, "passed": False}

    summary = output["summary"]
    points = output["key_points"]
    response = output["final_answer"]
    input_cyr = has_cyrillic(text)

    checks = {
        "has_keys": True,
        "summary_ok": count_words(summary) <= SUMMARY_MAX_WORDS,
        "points_count_ok": len(points) == 3,
        "points_short_ok": all(count_words(p) <= POINT_MAX_WORDS for p in points),
        "response_ok": count_sentences(response) <= RESPONSE_MAX_SENTENCES,
        "language_ok": has_cyrillic(summary) == input_cyr and has_cyrillic(response) == input_cyr,
    }
    checks["passed"] = all(checks.values())
    checks["summary_words"] = count_words(summary)
    return checks

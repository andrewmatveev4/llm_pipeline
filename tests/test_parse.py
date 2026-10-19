import pytest

from llm_pipeline.pipeline import LLMOutputError, parse_llm_output

OK = ('{"summary": "s", "category": "general_question", "intent": "i", "sentiment": "neutral", '
      '"key_points": ["a", "b", "c"], "final_answer": "f"}')


def test_valid_output_parses():
    assert parse_llm_output(OK).category == "general_question"


@pytest.mark.parametrize("raw, expected", [
    ("", "невалидный JSON"),
    ("```json\n" + OK + "\n```", "невалидный JSON"),
    (OK.replace('"sentiment": "neutral", ', ""), "sentiment: Field required"),
    (OK.replace('["a", "b", "c"]', '"a, b, c"'), "key_points"),
    (OK.replace('"general_question"', '"техподдержка"'), "category"),
])
def test_bad_output_raises_clear_error(raw, expected):
    with pytest.raises(LLMOutputError) as exc:
        parse_llm_output(raw)
    assert expected in str(exc.value)
    assert exc.value.raw == raw

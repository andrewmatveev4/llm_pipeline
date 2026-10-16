from llm_pipeline.checks import check_output, count_sentences, has_cyrillic

GOOD = {
    "summary": "Ноутбук тормозит.",
    "key_points": ["а", "б", "в"],
    "final_answer": "Проверь процессы.",
}


def test_count_sentences_ignores_trailing_empty():
    assert count_sentences("Раз. Два! Три?") == 3


def test_has_cyrillic_handles_yo():
    assert has_cyrillic("ё")


def test_good_output_passes():
    assert check_output(GOOD, "Мой ноутбук")["passed"]


def test_two_points_fail_only_count_check():
    result = check_output({**GOOD, "key_points": ["а", "б"]}, "Мой ноутбук")
    assert result["points_count_ok"] is False
    assert result["points_short_ok"] is True


def test_language_mismatch_fails():
    assert check_output(GOOD, "My laptop")["language_ok"] is False


def test_missing_key_fails_without_crash():
    assert check_output({"summary": "x"}, "text") == {"has_keys": False, "passed": False}

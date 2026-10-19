from main import build_report


def test_report_counts_zeros_and_failures():
    results = [
        {"name": "a", "ok": True, "output": {"category": "complaint", "sentiment": "negative"}},
        {"name": "b", "ok": False, "error": "x", "raw": ""},
    ]
    report = build_report(results)
    assert report["failed"] == ["b"]
    assert report["by_category"]["general_question"] == 0
    assert report["needs_attention"] == ["a"]

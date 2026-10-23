from collections import Counter
from typing import get_args

from llm_pipeline.schemas import Category, Sentiment


def count_all(allowed: tuple, values) -> dict:
    counts = Counter(values)
    return {v: counts.get(v, 0) for v in allowed}


def build_report(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    return {
        "total": len(results),
        "ok": len(ok),
        "failed": [r["name"] for r in results if not r["ok"]],
        "by_category": count_all(get_args(Category), (r["output"]["category"] for r in ok)),
        "by_sentiment": count_all(get_args(Sentiment), (r["output"]["sentiment"] for r in ok)),
        "needs_attention": [r["name"] for r in ok if r["output"]["sentiment"] == "negative"],
    }

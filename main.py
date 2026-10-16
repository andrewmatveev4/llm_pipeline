import json
from pathlib import Path
from typing import get_args
from collections import Counter

from llm_pipeline.schemas import Category, Sentiment
from llm_pipeline.pipeline import LLMOutputError, process_text

INPUT_DIR = "sample_inputs"
OUTPUT_PATH = "outputs/day03_results.json"
REPORT_PATH = "outputs/day03_report.json"


def load_inputs(folder: str) -> list[tuple[str, str]]:
    paths = sorted(Path(folder).glob("*.txt"))
    return [(p.stem, p.read_text(encoding="utf-8").strip()) for p in paths]


def save_results(results: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


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


def main() -> None:
    results = []
    for name, text in load_inputs(INPUT_DIR):
        print(f"Обрабатываю {name}...")
        try:
            output = process_text(text)
        except LLMOutputError as e:
            print(f"  ОШИБКА: {e}")
            results.append({"name": name, "input": text, "ok": False,
                            "error": str(e), "raw": e.raw})
            continue
        results.append({"name": name, "input": text, "ok": True,
                        "output": output.model_dump()})
        print(f"  [{output.category} / {output.sentiment}] {output.summary}")
    save_results(results, OUTPUT_PATH)
    ok_count = sum(r["ok"] for r in results)
    print(f"Успешно: {ok_count} из {len(results)}. Сохранено в {OUTPUT_PATH}")
    report = build_report(results)
    save_results(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

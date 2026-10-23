import json

from llm_pipeline.files import load_inputs, save_json
from llm_pipeline.pipeline import LLMOutputError, process_text
from llm_pipeline.report import build_report

INPUT_DIR = "sample_inputs"
OUTPUT_PATH = "outputs/day04_results.json"
REPORT_PATH = "outputs/day04_report.json"


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
    save_json(results, OUTPUT_PATH)
    ok_count = sum(r["ok"] for r in results)
    print(f"Успешно: {ok_count} из {len(results)}. Сохранено в {OUTPUT_PATH}")
    report = build_report(results)
    save_json(report, REPORT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

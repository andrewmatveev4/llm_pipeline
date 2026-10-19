import json
from pathlib import Path

from llm_pipeline.pipeline import LLMOutputError, process_text
from main import load_inputs

LABELS_PATH = "sample_inputs/labels.json"
RESULTS_PATH = "outputs/day04_routing.json"


def run_classification(inputs: list, labels: dict) -> list[dict]:
    rows = []
    for name, text in inputs:
        print(f"Обрабатываю {name}...")
        try:
            result = process_text(text)
        except LLMOutputError as e:
            rows.append({"name": name, "expected": labels[name], "actual": None,
                         "correct": False, "error": str(e)})
            continue
        rows.append({"name": name, "expected": labels[name], "actual": result.category,
                     "correct": result.category == labels[name],
                     "result": result.model_dump()})
    return rows


def accuracy_report(rows: list[dict]) -> dict:
    correct = sum(r["correct"] for r in rows)
    return {
        "accuracy": f"{correct}/{len(rows)}",
        "mismatches": [
            {"name": r["name"], "expected": r["expected"], "actual": r["actual"]}
            for r in rows if not r["correct"]
        ],
    }


def main() -> None:
    path = Path(RESULTS_PATH)
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"Загружено из кэша. Для нового прогона удали {RESULTS_PATH}")
    else:
        labels = json.loads(Path(LABELS_PATH).read_text(encoding="utf-8"))
        rows = run_classification(load_inputs("sample_inputs"), labels)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(accuracy_report(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
import json
from pathlib import Path

from llm_pipeline.checks import count_sentences
from llm_pipeline.pipeline import LLMOutputError, classify, generate_answer, process_text
from llm_pipeline.routes import ROUTES
from main import load_inputs

LABELS_PATH = "sample_inputs/labels.json"
RESULTS_PATH = "outputs/day04_routing.json"
FORCED_PATH = "outputs/day04_forced_routes.json"
FORCED_TEXT = "01_laptop"


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


def run_forced_routes(text: str) -> list[dict]:
    classification = classify(text)
    rows = []
    for route in ROUTES:
        forced = classification.model_copy(update={"category": route})
        answer = generate_answer(text, forced)
        rows.append({"route": route, "sentences": count_sentences(answer), "answer": answer})
    return rows


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


def show_forced_routes() -> None:
    path = Path(FORCED_PATH)
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        text = dict(load_inputs("sample_inputs"))[FORCED_TEXT]
        rows = run_forced_routes(text)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(f"\n=== {r['route']} ({r['sentences']} предл.)\n{r['answer']}")


if __name__ == "__main__":
    main()
    show_forced_routes()

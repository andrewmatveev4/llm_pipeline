import json
from pathlib import Path

from llm_pipeline.chain import run_chain
from llm_pipeline.log import setup_logging
from main import load_inputs

RESULTS_PATH = "outputs/day05_chain.json"


def summarize(states: list[dict]) -> dict:
    done = [s for s in states if s["failed_step"] is None]
    return {
        "total": len(states),
        "completed": len(done),
        "passed_self_check": sum(s["passed"] for s in done),
        "failed": {s["name"]: s["failed_step"] for s in states if s["failed_step"]},
        "needs_review": [s["name"] for s in done if not s["passed"]],
    }


def main() -> None:
    setup_logging()
    path = Path(RESULTS_PATH)
    if path.exists():
        states = json.loads(path.read_text(encoding="utf-8"))
        print(f"Загружено из кэша. Для нового прогона удали {RESULTS_PATH}")
    else:
        states = []
        for name, text in load_inputs("sample_inputs"):
            print(f"\n######## {name}")
            state = run_chain(text)
            states.append({"name": name, **state.model_dump()})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(states), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

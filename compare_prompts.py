from collections import defaultdict
import json
from pathlib import Path

from llm_pipeline.checks import check_output
from llm_pipeline.pipeline import LLMOutputError, process_text
from llm_pipeline.prompts import PROMPT_VARIANTS
from main import load_inputs

RESULTS_PATH = "outputs/day02_comparison.json"
REPEATS = 3


def run_experiment(inputs: list, variants: dict, repeats: int) -> list[dict]:
    rows = []
    for variant, prompt in variants.items():
        for name, text in inputs:
            for i in range(repeats):
                print(f"{variant} | {name} | повтор {i + 1}")
                try:
                    output = process_text(text, prompt).model_dump()
                    checks = check_output(output, text)
                except LLMOutputError:
                    output = None
                    checks = {"passed": False}
                rows.append({"variant": variant, "input": name, "repeat": i,
                             "parsed": output is not None, "output": output, **checks})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    table = []
    for variant, vrows in by_variant.items():
        words = defaultdict(list)
        for r in vrows:
            if r.get("summary_words") is not None:
                words[r["input"]].append(r["summary_words"])
        spreads = [max(w) - min(w) for w in words.values()]
        table.append({
            "variant": variant,
            "passed": sum(r["passed"] for r in vrows),
            "total": len(vrows),
            "parse_fail": sum(not r["parsed"] for r in vrows),
            "avg_spread": round(sum(spreads) / len(spreads), 1) if spreads else None,
        })
    return table


def main() -> None:
    path = Path(RESULTS_PATH)
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"Загружено из кэша: {len(rows)} строк. Для нового прогона удали {RESULTS_PATH}")
    else:
        rows = run_experiment(load_inputs("sample_inputs"), PROMPT_VARIANTS, REPEATS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in summarize(rows):
        print(line)


if __name__ == "__main__":
    main()

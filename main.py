import json
from pathlib import Path

from llm_pipeline.pipeline import process_text

INPUT_DIR = "sample_inputs"
OUTPUT_PATH = "outputs/day01_results.json"


def load_inputs(folder: str) -> list[tuple[str, str]]:
    paths = sorted(Path(folder).glob("*.txt"))
    return [(p.stem, p.read_text(encoding="utf-8").strip()) for p in paths]


def save_results(results: list[dict], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    results = []
    for name, text in load_inputs(INPUT_DIR):
        print(f"Обрабатываю {name}...")
        output = process_text(text)
        results.append({"name": name, "input": text, "output": output})
        print(f"  summary: {output['summary']}")
    save_results(results, OUTPUT_PATH)
    print(f"Сохранено {len(results)} результатов в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
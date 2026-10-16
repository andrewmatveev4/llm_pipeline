import json
from pathlib import Path

from llm_pipeline.checks import check_output
from llm_pipeline.pipeline import process_text
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
                    output = process_text(text, prompt)
                    checks = check_output(output, text)
                except json.JSONDecodeError:
                    output = None
                    checks = {"passed": False}
                rows.append({"variant": variant, "input": name, "repeat": i,
                             "parsed": output is not None, "output": output, **checks})
    return rows
import json
from pathlib import Path

for path in sorted(Path("demo/outputs").glob("*.json")):
    s = json.loads(path.read_text(encoding="utf-8"))
    category = s["request"]["category"] if s["request"] else "-"
    answer = (s["answer"] or "-")[:70].replace("\n", " ")
    print(f"{path.stem:12} | {category:9} | упал: {s['failed_step']} | "
          f"вердикт: {s['passed']} | {answer}")
import json
from pathlib import Path


def load_inputs(folder: str) -> list[tuple[str, str]]:
    paths = sorted(Path(folder).glob("*.txt"))
    return [(p.stem, p.read_text(encoding="utf-8").strip()) for p in paths]


def save_json(data, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

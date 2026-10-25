import argparse
import logging
import sys
from pathlib import Path

from llm_pipeline.errors import ConfigError
from llm_pipeline.chain import run_chain
from llm_pipeline.log import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-pipeline",
        description="Обработка текста LLM-цепочкой: смысл, категория, поля, ответ, самопроверка",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("text", nargs="?", help="текст для обработки")
    source.add_argument("--file", help="путь к файлу с текстом")
    parser.add_argument("--json", action="store_true", help="вывести полный результат в JSON")
    parser.add_argument("--verbose", action="store_true", help="подробные логи каждого шага")
    return parser


def read_text(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return args.text


def format_human(state) -> str:
    if state.failed_step:
        return f"Не удалось обработать: шаг {state.failed_step}\nПричина: {state.error}"
    lines = [
        f"Категория: {state.request.category} | intent: {state.request.intent}",
        f"Тональность: {state.request.sentiment} | срочность: {state.details.urgency}",
        "",
        state.answer,
        "",
    ]
    if state.fallback_reason:
        lines.append(f"Внимание: отдан запасной ответ ({state.fallback_reason})")
    lines.append("Самопроверка: " + ("OK" if state.passed else "нужна проверка человеком"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    try:
        text = read_text(args)
    except OSError as e:
        print(f"Не удалось прочитать файл: {e}", file=sys.stderr)
        return 2
    try:
        state = run_chain(text)
    except ConfigError as e:
        print(f"Ошибка настройки: {e}", file=sys.stderr)
        return 2
    print(state.model_dump_json(indent=2) if args.json else format_human(state))
    return 0 if state.failed_step is None else 1


if __name__ == "__main__":
    sys.exit(main())

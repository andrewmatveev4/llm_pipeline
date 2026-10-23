import pytest

import json

from llm_pipeline.cli import build_parser, main


def test_text_argument():
    args = build_parser().parse_args(["привет"])
    assert args.text == "привет"
    assert args.file is None
    assert args.json is False


def test_file_and_flags():
    args = build_parser().parse_args(["--file", "a.txt", "--json", "--verbose"])
    assert args.file == "a.txt"
    assert args.json is True
    assert args.verbose is True


def test_text_or_file_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_text_and_file_together_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["привет", "--file", "a.txt"])


def test_main_empty_text_returns_1(capsys):
    assert main(["   "]) == 1
    assert "check_input" in capsys.readouterr().out


def test_main_json_output(capsys):
    main(["   ", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["failed_step"] == "check_input"

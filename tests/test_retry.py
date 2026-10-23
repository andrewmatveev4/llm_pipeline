import pytest

from llm_pipeline.llm_client import LLMAPIError
from llm_pipeline.retry import with_retry

TEMP = LLMAPIError("временная", retryable=True)
FATAL = LLMAPIError("фатальная", retryable=False)


def make_func(outcomes):
    calls = []
    items = iter(outcomes)

    def func():
        calls.append(1)
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return item

    return func, calls


def test_success_first_try():
    func, calls = make_func(["ok"])
    sleeps = []
    assert with_retry(func, sleep=sleeps.append) == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_temporary_errors_then_success():
    func, calls = make_func([TEMP, TEMP, "ok"])
    sleeps = []
    assert with_retry(func, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_last_attempt():
    func, calls = make_func([TEMP, TEMP, TEMP])
    sleeps = []
    with pytest.raises(LLMAPIError):
        with_retry(func, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_fatal_error_not_retried():
    func, calls = make_func([FATAL, "ok"])
    sleeps = []
    with pytest.raises(LLMAPIError):
        with_retry(func, sleep=sleeps.append)
    assert len(calls) == 1
    assert sleeps == []

import logging
import time

from llm_pipeline.llm_client import LLMAPIError

logger = logging.getLogger(__name__)


def with_retry(func, attempts: int = 3, base_delay: float = 1.0, sleep=None):
    if sleep is None:
        sleep = time.sleep
    for attempt in range(1, attempts + 1):
        ...
        try:
            return func()
        except LLMAPIError as e:
            if not e.retryable:
                logger.error("Фатальная ошибка: %s", e)
                raise
            if attempt == attempts:
                logger.error("Превышено количество попыток: %s", e)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Повтор через %.1f с (попытка %s из %s): %s",
                           delay, attempt, attempts, e)
            sleep(delay)

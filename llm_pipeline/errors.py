class LLMOutputError(Exception):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


class LLMAPIError(Exception):
    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class ConfigError(Exception):
    pass

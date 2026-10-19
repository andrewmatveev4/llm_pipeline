import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("llm_pipeline").setLevel(level)
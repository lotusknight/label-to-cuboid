import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging(log_level: str) -> None:
    """Configure JSON logging for container-friendly logs."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(filename)s %(lineno)d"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


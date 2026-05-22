"""Console and daily-rotating file logging for the BalaGAN engine."""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_LOG_FILENAME = "balagan.log"


def setup_logging(log_dir: Path | str, level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler and a daily-rotating
    file handler (14-day retention) writing into ``log_dir``.

    Safe to call more than once; existing root handlers are replaced.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        log_dir / _LOG_FILENAME,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

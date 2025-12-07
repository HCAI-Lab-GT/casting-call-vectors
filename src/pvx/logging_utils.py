import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "persona",
    log_dir: str = "logs/runtime",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that writes to stdout and a rotating-ish timestamped file.
    Safe to call multiple times; handlers are added only once per logger name.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{name}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Logging initialized. Writing to %s", log_path)
    return logger


class Heartbeat:
    """
    Periodically emit a log message while long-running work progresses.
    Usage:
        with Heartbeat(logger, "generating dataset...", interval=30):
            do_work()
    """

    def __init__(self, logger: logging.Logger, message: str, interval: int = 30):
        self.logger = logger
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        while not self._stop.wait(self.interval):
            self.logger.info("[heartbeat] %s", self.message)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval)

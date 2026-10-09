"""Logging: quiet on the terminal by default, complete in the log file."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_NOISY = ("PIL", "urllib3", "httpx", "httpcore", "transformers", "huggingface_hub", "filelock")


def silence_third_party() -> None:
    """Call before importing torch/transformers/mediapipe to cut their startup chatter."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def configure_logging(console: Console, verbosity: int = 0, log_file: Path | None = None) -> None:
    """verbosity: 0 = warnings only, 1 = info, 2+ = debug (terminal). File is always DEBUG."""
    root = logging.getLogger("media_culler")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False

    terminal_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    handler = RichHandler(
        console=console,
        level=terminal_level,
        show_time=False,
        show_path=False,
        markup=False,
        rich_tracebacks=verbosity >= 2,
    )
    root.addHandler(handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
            )
            file_handler.setLevel(logging.DEBUG)
            root.addHandler(file_handler)
        except OSError:
            root.warning("Cannot write log file %s; continuing without it.", log_file)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

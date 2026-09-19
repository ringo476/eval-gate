"""One place to configure logging. Never logs API keys or full prompts
at INFO -- prompts/responses are only logged at DEBUG, and even then the
API key itself is never in scope to log."""
from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("harness")
    root.setLevel(level)
    root.handlers = [handler]
    root.propagate = False

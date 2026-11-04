"""Rich-based logging configuration shared by all CLI commands."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a single rich handler.

    Parameters
    ----------
    level : str
        Logging level name (``"DEBUG"``, ``"INFO"``, ...).

    Notes
    -----
    Modules only call ``logging.getLogger(__name__)``; handler configuration happens
    once, here, at the CLI entry point. Calling this twice replaces the handler instead
    of duplicating output.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )

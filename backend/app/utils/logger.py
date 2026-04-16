from __future__ import annotations

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
LOG_BACKUP_COUNT = 5


def setup_logging(log_level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging for the entire application."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[*shared_processors, renderer]
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(LOG_DIR / "backend.log"),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        json_fmt = structlog.stdlib.ProcessorFormatter(
            processors=[*shared_processors, structlog.processors.JSONRenderer()]
        )
        file_handler.setFormatter(json_fmt)
        root.addHandler(file_handler)
    except Exception:
        pass

    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

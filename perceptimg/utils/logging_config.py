"""Logging helpers for structured, JSON-friendly output."""

from __future__ import annotations

import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Serialize log records into a structured JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach extra fields if provided
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_"):
                continue
            if key in {"args", "msg", "exc_text"}:
                continue
            try:
                # Try to serialize, fall back to string representation
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(
    *,
    level: int = logging.INFO,
    json_output: bool = True,
    logger_names: tuple[str, ...] = (),
    merge: bool = True,
) -> None:
    """Configure global logging with optional JSON formatting."""

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(level)
    if not merge:
        root.handlers.clear()
    else:
        # Remove existing JsonFormatter handlers to avoid duplicates
        root.handlers = [h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)]
    root.addHandler(handler)

    for name in logger_names:
        logging.getLogger(name).setLevel(level)

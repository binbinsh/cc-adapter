import json
import logging
from typing import Any, Union


def resolve_log_level(value: Union[str, int, None]) -> int:
    """Safely resolve a log level name or numeric value, defaulting to INFO."""
    if value is None:
        return logging.INFO
    text = str(value).upper()
    try:
        level = logging._checkLevel(text)  # type: ignore[attr-defined]
    except Exception:
        return logging.INFO
    return level if isinstance(level, int) else logging.INFO


def log_payload(logger: logging.Logger, heading: str, payload: Any) -> None:
    """Emit structured payload details only when debug logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        logger.debug("%s: %r", heading, payload)
        return
    logger.debug("%s:\n%s", heading, serialized)

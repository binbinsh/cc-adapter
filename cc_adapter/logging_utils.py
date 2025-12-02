import json
import logging
from typing import Any, Union


VERBOSE_LEVEL = 15


def ensure_verbose_logging() -> int:
    """Register a VERBOSE log level and convenience method if missing."""
    if logging.getLevelName(VERBOSE_LEVEL) != "VERBOSE":
        logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")
    if not hasattr(logging, "VERBOSE"):
        logging.VERBOSE = VERBOSE_LEVEL  # type: ignore[attr-defined]
    if not hasattr(logging.Logger, "verbose"):
        # Attach Logger.verbose for structured use at the custom level.
        def verbose(self, message, *args, **kwargs):
            if self.isEnabledFor(VERBOSE_LEVEL):
                self._log(VERBOSE_LEVEL, message, args, **kwargs)

        logging.Logger.verbose = verbose  # type: ignore[attr-defined]
    return VERBOSE_LEVEL


def resolve_log_level(value: Union[str, int, None]) -> int:
    """Safely resolve a log level name or numeric value, defaulting to INFO."""
    ensure_verbose_logging()
    if value is None:
        return logging.INFO
    try:
        level = logging._checkLevel(str(value).upper())  # type: ignore[attr-defined]
    except Exception:
        return logging.INFO
    return level if isinstance(level, int) else logging.INFO


def log_payload(logger: logging.Logger, heading: str, payload: Any) -> None:
    """Emit structured payload details only when VERBOSE logging is enabled."""
    ensure_verbose_logging()
    if not logger.isEnabledFor(VERBOSE_LEVEL):
        return
    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        logger.verbose("%s: %r", heading, payload)
        return
    logger.verbose("%s:\n%s", heading, serialized)

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
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


def _parse_int_env(keys: list[str], default: int) -> int:
    for key in keys:
        raw = (os.getenv(key, "") or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= 0:
            return value
    return default


def _log_file_from_env() -> str:
    value = (os.getenv("CC_ADAPTER_LOG_FILE") or os.getenv("LOG_FILE") or "").strip()
    if not value:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(value))
    return expanded.strip()


def file_handler_from_env(
    *,
    fmt: str = "%(asctime)s %(levelname)s %(message)s",
    max_bytes_default: int = 10 * 1024 * 1024,
    backup_count_default: int = 3,
) -> logging.Handler | None:
    """
    Return a rotating file handler if configured via env, otherwise None.

    Env vars:
      - CC_ADAPTER_LOG_FILE or LOG_FILE
      - CC_ADAPTER_LOG_MAX_BYTES or LOG_FILE_MAX_BYTES
      - CC_ADAPTER_LOG_BACKUP_COUNT or LOG_FILE_BACKUP_COUNT
    """
    log_file = _log_file_from_env()
    if not log_file:
        return None

    max_bytes = _parse_int_env(["CC_ADAPTER_LOG_MAX_BYTES", "LOG_FILE_MAX_BYTES"], max_bytes_default)
    backup_count = _parse_int_env(["CC_ADAPTER_LOG_BACKUP_COUNT", "LOG_FILE_BACKUP_COUNT"], backup_count_default)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def configure_root_logging() -> None:
    """
    Configure root logging to stderr and (optionally) a rotating log file.

    This is intended for the CLI server entrypoint. GUI configures logging separately.
    """
    fmt = "%(asctime)s %(levelname)s %(message)s"
    level = resolve_log_level(os.getenv("LOG_LEVEL", "INFO"))
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_handler = file_handler_from_env(fmt=fmt)
    if file_handler is not None:
        handlers.append(file_handler)
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

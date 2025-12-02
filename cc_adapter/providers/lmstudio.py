import logging
import requests
from typing import Dict, Any, Optional, List
from http.server import BaseHTTPRequestHandler

from ..config import Settings
from ..context_limits import enforce_context_limits
from ..streaming import stream_openai_response
import copy
from ..logging_utils import log_payload


logger = logging.getLogger(__name__)


ALLOWED_TOP_LEVEL = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "tools",
    "tool_choice",
}


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: copy.deepcopy(v) for k, v in payload.items() if k in ALLOWED_TOP_LEVEL}
    cleaned.pop("reasoning", None)

    msgs: List[Dict[str, Any]] = []
    for msg in cleaned.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        msg = dict(msg)
        msg.pop("cache_control", None)
        # Strip cache_control from any nested content blocks
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict):
                    part = dict(part)
                    part.pop("cache_control", None)
                new_content.append(part)
            msg["content"] = new_content
        msgs.append(msg)
    cleaned["messages"] = msgs
    return cleaned


def send(payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    clean_payload, trim_meta = enforce_context_limits(_sanitize_payload(payload), settings, payload.get("model", settings.lmstudio_model))
    if trim_meta.get("dropped"):
        # LM Studio server returns 400 on over-length prompts; prune to avoid.
        pass
    log_payload(logger, f"LM Studio request -> {clean_payload.get('model') or settings.lmstudio_model}", clean_payload)
    resp = requests.post(
        settings.lmstudio_base,
        json=clean_payload,
        timeout=settings.lmstudio_timeout,
        proxies=settings.resolved_proxies(),
        stream=False,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | body={resp.text}") from exc
    data = resp.json()
    log_payload(logger, "LM Studio raw response", data)
    return data


def stream(
    payload: Dict[str, Any],
    settings: Settings,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    clean_payload, trim_meta = enforce_context_limits(_sanitize_payload(payload), settings, requested_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for LM Studio context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    log_payload(logger, f"LM Studio stream request -> {requested_model}", clean_payload)
    resp = requests.post(
        settings.lmstudio_base,
        json=clean_payload,
        timeout=settings.lmstudio_timeout,
        proxies=settings.resolved_proxies(),
        stream=True,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | body={resp.text}") from exc

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    stream_openai_response(resp, requested_model, incoming, handler, logger)

import requests
from typing import Dict, Any, Optional, List
from http.server import BaseHTTPRequestHandler

from ..config import Settings
from ..streaming import stream_openai_response
import copy


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
    clean_payload = _sanitize_payload(payload)

    resp = requests.post(
        settings.lmstudio_base,
        json=clean_payload,
        timeout=settings.lmstudio_timeout,
        stream=False,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | body={resp.text}") from exc
    return resp.json()


def stream(
    payload: Dict[str, Any],
    settings: Settings,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    clean_payload = _sanitize_payload(payload)

    resp = requests.post(
        settings.lmstudio_base,
        json=clean_payload,
        timeout=settings.lmstudio_timeout,
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

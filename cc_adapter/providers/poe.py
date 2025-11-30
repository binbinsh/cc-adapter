import copy
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional

import requests

from ..config import Settings
from ..converters import openai_to_anthropic
from ..streaming import stream_openai_response
from ..context_limits import enforce_context_limits

# Poe supports an OpenAI-compatible /v1/chat/completions endpoint. We forward
# cleaned OpenAI payloads and bridge the streaming response into Anthropic SSE.
ALLOWED_TOP_LEVEL = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "n",
    "logprobs",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "extra_body",
}

logger = logging.getLogger(__name__)


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {k: copy.deepcopy(v) for k, v in payload.items() if k in ALLOWED_TOP_LEVEL}
    # Poe ignores/errs on unknown top-level keys; drop proto-reasoning hints.
    cleaned.pop("reasoning", None)

    msgs = []
    for msg in cleaned.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        msg = dict(msg)
        msg.pop("cache_control", None)
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


def send(payload: Dict[str, Any], settings: Settings, target_model: str, incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not settings.poe_api_key:
        raise RuntimeError("POE_API_KEY not set")
    clean_payload, trim_meta = enforce_context_limits(_sanitize_payload(payload), settings, target_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Poe context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    headers = {"Authorization": f"Bearer {settings.poe_api_key}"}
    resp = requests.post(
        settings.poe_base_url,
        json=clean_payload,
        headers=headers,
        timeout=settings.lmstudio_timeout,
        proxies=settings.resolved_proxies(),
        stream=False,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc} | body={resp.text}") from exc

    data = resp.json()
    return openai_to_anthropic(data, target_model, incoming)


def stream(
    payload: Dict[str, Any],
    settings: Settings,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    if not settings.poe_api_key:
        raise RuntimeError("POE_API_KEY not set")
    clean_payload, trim_meta = enforce_context_limits(_sanitize_payload(payload), settings, requested_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Poe context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    headers = {"Authorization": f"Bearer {settings.poe_api_key}"}
    resp = requests.post(
        settings.poe_base_url,
        json=clean_payload,
        headers=headers,
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

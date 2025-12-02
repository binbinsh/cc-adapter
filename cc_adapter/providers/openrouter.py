import logging
import requests
from typing import Dict, Any, Optional
from http.server import BaseHTTPRequestHandler

from ..config import Settings
from ..streaming import stream_openai_response
from ..context_limits import enforce_context_limits
from ..logging_utils import log_payload

logger = logging.getLogger(__name__)


def send(payload: Dict[str, Any], settings: Settings, target_model: str) -> Dict[str, Any]:
    if not settings.openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload, trim_meta = enforce_context_limits(payload, settings, target_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for OpenRouter context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )
    log_payload(logger, f"OpenRouter request -> {target_model}", payload)
    headers = {"Authorization": f"Bearer {settings.openrouter_key}"}
    resp = requests.post(
        settings.openrouter_base,
        json=payload,
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
    log_payload(logger, "OpenRouter raw response", data)
    if not data.get("usage"):
        header_map = {k.lower(): v for k, v in resp.headers.items()}
        prompt = header_map.get("x-openrouter-usage-prompt-tokens") or header_map.get(
            "x-openai-usage-prompt-tokens"
        )
        completion = header_map.get("x-openrouter-usage-completion-tokens") or header_map.get(
            "x-openai-usage-completion-tokens"
        )
        usage: Dict[str, Any] = {}
        try:
            if prompt is not None:
                usage["prompt_tokens"] = int(prompt)
            if completion is not None:
                usage["completion_tokens"] = int(completion)
        except ValueError:
            usage = {}
        if usage:
            data["usage"] = usage
    return data


def stream(
    payload: Dict[str, Any],
    settings: Settings,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    if not settings.openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload, trim_meta = enforce_context_limits(payload, settings, requested_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for OpenRouter context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )
    log_payload(logger, f"OpenRouter request -> {requested_model}", payload)
    headers = {"Authorization": f"Bearer {settings.openrouter_key}"}
    resp = requests.post(
        settings.openrouter_base,
        json=payload,
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

import copy
import logging
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from ..config import Settings
from ..converters import openai_to_anthropic, build_poe_params
from ..model_registry import default_extra_body_for
from ..streaming import stream_openai_response
from ..context_limits import enforce_context_limits
from ..logging_utils import log_payload

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
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


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


def _merge_extra_body(
    payload: Dict[str, Any], incoming: Optional[Dict[str, Any]], defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(payload)
    extra_body: Dict[str, Any] = {}
    if defaults:
        extra_body.update(defaults)
    if isinstance(merged.get("extra_body"), dict):
        extra_body.update(merged["extra_body"])  # type: ignore[arg-type]

    derived = build_poe_params(incoming or {})
    # Merge order: defaults -> derived from request -> explicit payload extras win last.
    final_extra = {**defaults} if defaults else {}
    final_extra.update(derived)
    final_extra.update(extra_body)
    if final_extra:
        merged["extra_body"] = final_extra
    else:
        merged.pop("extra_body", None)
    return merged


def _prepare_payload(
    payload: Dict[str, Any],
    incoming: Optional[Dict[str, Any]],
    settings: Settings,
    target_model: str,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    defaults = default_extra_body_for(f"poe:{target_model}")
    enriched = _merge_extra_body(payload, incoming, defaults)
    clean_payload = _sanitize_payload(enriched)
    return enforce_context_limits(clean_payload, settings, target_model)


def _build_retry_session(settings: Settings) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=settings.poe_max_retries,
        backoff_factor=settings.poe_retry_backoff,
        status_forcelist=RETRYABLE_STATUS_CODES,
        allowed_methods={"POST"},
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    proxies = settings.resolved_proxies()
    if proxies:
        session.proxies.update(proxies)
    return session


def _response_body_snippet(resp: Optional[requests.Response], limit: int = 600) -> str:
    if not resp:
        return ""
    try:
        body = resp.text or ""
    except Exception:
        return ""
    body = " ".join(body.split())
    if not body:
        return ""
    if len(body) > limit:
        return f"{body[:limit]}..."
    return body


def _post_with_retries(
    payload: Dict[str, Any], settings: Settings, stream: bool = False
) -> Tuple[requests.Session, requests.Response]:
    session = _build_retry_session(settings)
    resp: Optional[requests.Response] = None
    try:
        resp = session.post(
            settings.poe_base_url,
            json=payload,
            headers={"Authorization": f"Bearer {settings.poe_api_key}"},
            timeout=settings.lmstudio_timeout,
            stream=stream,
        )
        resp.raise_for_status()
        return session, resp
    except requests.HTTPError as exc:
        snippet = _response_body_snippet(resp or getattr(exc, "response", None))
        if resp:
            resp.close()
        session.close()
        msg = str(exc)
        if snippet:
            msg = f"{msg} | body_snippet={snippet}"
        raise requests.HTTPError(msg) from exc
    except requests.RequestException:
        if resp:
            resp.close()
        session.close()
        raise


def send(payload: Dict[str, Any], settings: Settings, target_model: str, incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not settings.poe_api_key:
        raise RuntimeError("POE_API_KEY not set")
    clean_payload, trim_meta = _prepare_payload(payload, incoming, settings, target_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Poe context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    log_payload(logger, f"Poe request -> {target_model}", clean_payload)
    session, resp = _post_with_retries(clean_payload, settings, stream=False)
    try:
        data = resp.json()
        log_payload(logger, "Poe raw response", data)
        return openai_to_anthropic(data, target_model, incoming)
    finally:
        resp.close()
        session.close()


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
    clean_payload, trim_meta = _prepare_payload(payload, incoming, settings, requested_model)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Poe context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    log_payload(logger, f"Poe stream request -> {requested_model}", clean_payload)
    session, resp = _post_with_retries(clean_payload, settings, stream=True)
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    try:
        stream_openai_response(resp, requested_model, incoming, handler, logger)
    finally:
        try:
            resp.close()
        finally:
            session.close()

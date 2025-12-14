import copy
import logging
import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..codex_instructions import get_codex_instructions
from ..codex_oauth import (
    CodexOAuthTokens,
    extract_chatgpt_account_id,
    load_tokens,
    refresh_access_token,
    save_tokens,
)
from ..config import Settings
from ..context_limits import enforce_context_limits
from ..logging_utils import log_payload
from ..model_registry import default_extra_body_for
from ..streaming import stream_responses_response

logger = logging.getLogger(__name__)

OPENAI_BETA_HEADER = "OpenAI-Beta"
OPENAI_BETA_VALUE = "responses=experimental"
CHATGPT_ACCOUNT_HEADER = "chatgpt-account-id"
ORIGINATOR_HEADER = "originator"
ORIGINATOR_VALUE = "codex_cli_rs"

_CODEX_GLOBAL_DEFAULTS: Dict[str, Any] = {
    "reasoning": {"effort": "medium", "summary": "auto"},
    "text": {"verbosity": "medium"},
    "include": ["reasoning.encrypted_content"],
    "store": False,
}


def _codex_model_key(settings: Settings, target_model: str) -> str:
    selected = (getattr(settings, "model", "") or "").strip()
    if selected.lower().startswith("codex:"):
        _, name = selected.split(":", 1)
        if name.strip():
            return f"codex:{name.strip()}"
    return f"codex:{target_model}"


def _tokens_from_settings(settings: Settings) -> Optional[CodexOAuthTokens]:
    access = getattr(settings, "codex_access_token", "") or ""
    refresh = getattr(settings, "codex_refresh_token", "") or ""
    expires = int(getattr(settings, "codex_expires_at_ms", 0) or 0)
    if access and refresh and expires > 0:
        return CodexOAuthTokens(access=access, refresh=refresh, expires_at_ms=expires)
    return None


def _normalized_auth_mode(settings: Settings) -> str:
    mode = str(getattr(settings, "codex_auth", "") or "").strip().lower()
    if mode in {"oauth", "login", "file", "stored"}:
        return "oauth"
    if mode in {"env", "token", "tokens"}:
        return "env"
    return "auto"


def _apply_tokens_to_settings(settings: Settings, tokens: CodexOAuthTokens) -> None:
    try:
        settings.codex_access_token = tokens.access
        settings.codex_refresh_token = tokens.refresh
        settings.codex_expires_at_ms = tokens.expires_at_ms
    except Exception:
        return


def _resolve_codex_auth(settings: Settings) -> Tuple[CodexOAuthTokens, str]:
    auth_mode = _normalized_auth_mode(settings)
    tokens: Optional[CodexOAuthTokens] = None
    from_file = False
    if auth_mode in {"auto", "env"}:
        tokens = _tokens_from_settings(settings)
    if not tokens and auth_mode in {"auto", "oauth"}:
        tokens = load_tokens()
        from_file = True
    if not tokens:
        if auth_mode == "env":
            raise RuntimeError(
                "OpenAI Codex OAuth env tokens not configured (set OPENAI_CODEX_ACCESS_TOKEN/OPENAI_CODEX_REFRESH_TOKEN/OPENAI_CODEX_EXPIRES_AT_MS)"
            )
        raise RuntimeError("OpenAI Codex OAuth not configured (run cc-adapter-codex-login)")

    if tokens.expired():
        tokens = refresh_access_token(
            tokens.refresh,
            proxies=settings.resolved_proxies(),
            timeout=float(settings.lmstudio_timeout),
        )
        _apply_tokens_to_settings(settings, tokens)
        if from_file:
            save_tokens(tokens)
    else:
        _apply_tokens_to_settings(settings, tokens)

    account_id = extract_chatgpt_account_id(tokens.access)
    if not account_id:
        raise RuntimeError("Failed to extract chatgpt_account_id from OAuth token")
    return tokens, account_id


def _headers(account_id: str, access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        CHATGPT_ACCOUNT_HEADER: account_id,
        OPENAI_BETA_HEADER: OPENAI_BETA_VALUE,
        ORIGINATOR_HEADER: ORIGINATOR_VALUE,
        "accept": "text/event-stream",
    }


def _responses_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(tools, list):
        return None
    out: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        entry: Dict[str, Any] = {"type": "function", "name": str(name)}
        desc = func.get("description")
        if desc:
            entry["description"] = str(desc)
        params = func.get("parameters")
        if params is not None:
            entry["parameters"] = params
        out.append(entry)
    return out or None


def _responses_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                return {"type": "function", "name": fn.get("name")}
            if tool_choice.get("name"):
                return {"type": "function", "name": tool_choice.get("name")}
    return tool_choice


def _responses_content_parts(
    content: Any,
    *,
    text_part_type: str = "input_text",
    include_images: bool = True,
) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if content is None:
        return parts
    if isinstance(content, str):
        if content:
            parts.append({"type": text_part_type, "text": content})
        return parts
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append({"type": text_part_type, "text": item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("text", "input_text", "output_text") and item.get("text") is not None:
                text = str(item.get("text") or "")
                if text:
                    parts.append({"type": text_part_type, "text": text})
                continue
            if item_type == "image_url":
                if not include_images:
                    continue
                image_url = item.get("image_url") or {}
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                else:
                    url = image_url
                if url:
                    parts.append({"type": "input_image", "image_url": str(url)})
                continue
            if "text" in item and item.get("text") is not None:
                text = str(item.get("text") or "")
                if text:
                    parts.append({"type": text_part_type, "text": text})
    if isinstance(content, dict) and "text" in content:
        text = str(content.get("text") or "")
        if text:
            parts.append({"type": text_part_type, "text": text})
    return parts


def _flatten_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: List[str] = []
        for item in content:
            if isinstance(item, str):
                bits.append(item)
            elif isinstance(item, dict):
                if "text" in item and item.get("text") is not None:
                    bits.append(str(item.get("text") or ""))
        return "\n".join([b for b in bits if b])
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text") or "")
    return str(content)


def _messages_to_responses_input(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    developer_parts: List[str] = []
    items: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "").strip()
        if role == "system":
            text = _flatten_text(msg.get("content"))
            if text:
                developer_parts.append(text)
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id") or msg.get("id")
            output = _flatten_text(msg.get("content"))
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call_id),
                        "output": output,
                    }
                )
            else:
                # Best-effort fallback: surface tool output as text context.
                if output:
                    items.append(
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": output}],
                        }
                    )
            continue

        content_parts = _responses_content_parts(
            msg.get("content"),
            text_part_type="output_text" if role == "assistant" else "input_text",
            include_images=(role != "assistant"),
        )
        if content_parts:
            items.append({"type": "message", "role": role or "user", "content": content_parts})

        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                func = call.get("function") or {}
                if not isinstance(func, dict):
                    continue
                call_id = call.get("id") or func.get("name") or "tool_call"
                name = func.get("name") or "tool"
                arguments = func.get("arguments") or ""
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call_id),
                        "name": str(name),
                        "arguments": str(arguments),
                    }
                )

    developer_prompt = "\n\n".join([p for p in developer_parts if p]).strip()
    return developer_prompt, items


def _parse_final_response(resp: requests.Response) -> Dict[str, Any]:
    for line in resp.iter_lines(decode_unicode=False):
        if not line:
            continue
        if not line.startswith(b"data:"):
            continue
        raw = line[len(b"data:") :].strip()
        if raw == b"[DONE]":
            break
        try:
            event_obj = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        etype = str(event_obj.get("type") or "")
        if etype in ("response.completed", "response.done") and isinstance(event_obj.get("response"), dict):
            return event_obj["response"]
        if etype == "error":
            raise RuntimeError(str(event_obj.get("message") or event_obj.get("error") or "Unknown error"))
    raise RuntimeError("Codex backend stream ended without a final response event")


def _responses_to_chat_completions(response_obj: Dict[str, Any]) -> Dict[str, Any]:
    output_items = response_obj.get("output") or []
    tool_calls: List[Dict[str, Any]] = []
    text_bits: List[str] = []

    if isinstance(response_obj.get("output_text"), str) and response_obj.get("output_text"):
        text_bits.append(str(response_obj.get("output_text")))

    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                content = item.get("content")
                if isinstance(content, str):
                    if content:
                        text_bits.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text") is not None:
                            maybe = str(part.get("text") or "")
                            if maybe:
                                text_bits.append(maybe)
            elif itype == "function_call":
                call_id = item.get("call_id") or item.get("id") or "tool_call"
                name = item.get("name") or "tool"
                args = item.get("arguments") or ""
                tool_calls.append(
                    {
                        "id": str(call_id),
                        "type": "function",
                        "function": {"name": str(name), "arguments": str(args)},
                    }
                )

    text = "\n".join([t for t in text_bits if t]).strip()
    message: Dict[str, Any] = {"role": "assistant", "content": [{"type": "text", "text": text}] if text else ""}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"
    chat: Dict[str, Any] = {
        "id": response_obj.get("id") or "resp",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    usage = response_obj.get("usage") or {}
    if isinstance(usage, dict):
        mapped: Dict[str, Any] = {}
        if "input_tokens" in usage:
            mapped["prompt_tokens"] = usage.get("input_tokens")
        if "output_tokens" in usage:
            mapped["completion_tokens"] = usage.get("output_tokens")
        if mapped:
            chat["usage"] = mapped
    return chat


def _request_body(
    payload: Dict[str, Any],
    settings: Settings,
    *,
    model_key: Optional[str] = None,
    force_refresh_instructions: bool = False,
) -> Dict[str, Any]:
    developer_prompt, input_items = _messages_to_responses_input(payload.get("messages") or [])
    model_defaults = default_extra_body_for(model_key) if model_key else {}

    timeout = min(30.0, float(settings.lmstudio_timeout))
    instructions = get_codex_instructions(
        payload.get("model") or "",
        proxies=settings.resolved_proxies(),
        timeout=timeout,
        force_refresh=force_refresh_instructions,
    )

    if developer_prompt:
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": developer_prompt}],
            },
            *input_items,
        ]

    body: Dict[str, Any] = {
        "model": payload.get("model") or "",
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
    }

    if isinstance(model_defaults.get("include"), list) and model_defaults.get("include"):
        body["include"] = model_defaults.get("include")
    if isinstance(payload.get("include"), list) and payload.get("include"):
        body["include"] = payload.get("include")

    # ChatGPT Codex backend rejects sampling parameters such as temperature/top_p.

    reasoning = copy.deepcopy(_CODEX_GLOBAL_DEFAULTS.get("reasoning") or {})
    if isinstance(payload.get("reasoning"), dict) and payload.get("reasoning"):
        reasoning.update(payload.get("reasoning"))
    if isinstance(model_defaults.get("reasoning"), dict) and model_defaults.get("reasoning"):
        # Preset defaults override per-request hints.
        reasoning.update(model_defaults.get("reasoning"))
    if reasoning:
        body["reasoning"] = reasoning

    text = copy.deepcopy(_CODEX_GLOBAL_DEFAULTS.get("text") or {})
    if isinstance(payload.get("text"), dict) and payload.get("text"):
        text.update(payload.get("text"))
    if isinstance(model_defaults.get("text"), dict) and model_defaults.get("text"):
        # Preset defaults override per-request hints.
        text.update(model_defaults.get("text"))
    if text:
        body["text"] = text

    tools = _responses_tools(payload.get("tools"))
    if tools:
        body["tools"] = tools

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        body["tool_choice"] = _responses_tool_choice(tool_choice)

    return body


def send(payload: Dict[str, Any], settings: Settings, target_model: str) -> Dict[str, Any]:
    tokens, account_id = _resolve_codex_auth(settings)
    model_key = _codex_model_key(settings, target_model)
    trimmed_payload, trim_meta = enforce_context_limits(payload, settings, model_key)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Codex context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    req_payload = dict(trimmed_payload)
    req_payload["model"] = target_model
    body = _request_body(req_payload, settings, model_key=model_key)
    log_payload(logger, f"Codex request -> {target_model}", body)

    resp = requests.post(
        settings.codex_base_url,
        json=body,
        headers=_headers(account_id, tokens.access),
        timeout=float(settings.lmstudio_timeout),
        proxies=settings.resolved_proxies(),
        stream=True,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body_text = resp.text or ""
        if resp.status_code == 400 and "Instructions are not valid" in body_text:
            try:
                resp.close()
            except Exception:
                pass
            logger.warning("Codex backend rejected instructions; refreshing and retrying once.")
            body = _request_body(req_payload, settings, model_key=model_key, force_refresh_instructions=True)
            resp = requests.post(
                settings.codex_base_url,
                json=body,
                headers=_headers(account_id, tokens.access),
                timeout=float(settings.lmstudio_timeout),
                proxies=settings.resolved_proxies(),
                stream=True,
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc2:
                raise requests.HTTPError(f"{exc2} | body={resp.text}") from exc2
        else:
            raise requests.HTTPError(f"{exc} | body={body_text}") from exc

    try:
        final = _parse_final_response(resp)
        chat = _responses_to_chat_completions(final)
        log_payload(logger, "Codex final response (parsed)", final)
        return chat
    finally:
        try:
            resp.close()
        except Exception:
            pass


def stream(
    payload: Dict[str, Any],
    settings: Settings,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    tokens, account_id = _resolve_codex_auth(settings)
    model_key = _codex_model_key(settings, requested_model)
    trimmed_payload, trim_meta = enforce_context_limits(payload, settings, model_key)
    if trim_meta.get("dropped"):
        logger.warning(
            "Trimmed %s message(s) for Codex context (est %s -> %s tokens, budget=%s)",
            trim_meta["dropped"],
            trim_meta.get("before", 0),
            trim_meta.get("after", 0),
            trim_meta.get("budget", 0),
        )

    req_payload = dict(trimmed_payload)
    req_payload["model"] = requested_model
    body = _request_body(req_payload, settings, model_key=model_key)
    log_payload(logger, f"Codex stream request -> {requested_model}", body)

    resp = requests.post(
        settings.codex_base_url,
        json=body,
        headers=_headers(account_id, tokens.access),
        timeout=float(settings.lmstudio_timeout),
        proxies=settings.resolved_proxies(),
        stream=True,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body_text = resp.text or ""
        if resp.status_code == 400 and "Instructions are not valid" in body_text:
            try:
                resp.close()
            except Exception:
                pass
            logger.warning("Codex backend rejected instructions; refreshing and retrying once.")
            body = _request_body(req_payload, settings, model_key=model_key, force_refresh_instructions=True)
            resp = requests.post(
                settings.codex_base_url,
                json=body,
                headers=_headers(account_id, tokens.access),
                timeout=float(settings.lmstudio_timeout),
                proxies=settings.resolved_proxies(),
                stream=True,
            )
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc2:
                raise requests.HTTPError(f"{exc2} | body={resp.text}") from exc2
        else:
            raise requests.HTTPError(f"{exc} | body={body_text}") from exc

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

    stream_responses_response(resp, requested_model, incoming, handler, logger)

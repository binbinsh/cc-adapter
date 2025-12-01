import logging
from typing import Any, Dict, List, Tuple

from .config import Settings

logger = logging.getLogger(__name__)


def _token_estimator():
    """Return a callable that estimates tokens for a text blob."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        return lambda text: max(1, (len(text) // 4) + 1)


def _flatten_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    nested = _flatten_tool_content(item.get("content"))
                    if nested:
                        parts.append(nested)
        return "\n".join([p for p in parts if p])
    if content is None:
        return ""
    return str(content)


def _normalize_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    allow_tool = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            allow_tool = bool(msg.get("tool_calls"))
            normalized.append(msg)
        elif role == "tool":
            if allow_tool:
                normalized.append(msg)
            else:
                text = _flatten_tool_content(msg.get("content"))
                if not text:
                    allow_tool = False
                    continue
                prefix = msg.get("tool_call_id")
                if prefix:
                    text = f"[tool:{prefix}] {text}".strip()
                normalized.append({"role": "user", "content": text})
                allow_tool = False
        else:
            allow_tool = False
            normalized.append(msg)
    return normalized


def _message_token_count(msg: Dict[str, Any], estimate) -> int:
    tokens = 0
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                tokens += estimate(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    tokens += estimate(text_val)
                elif isinstance(text_val, list):
                    tokens += sum(estimate(str(item or "")) for item in text_val)
    elif isinstance(content, dict) and "text" in content:
        tokens += estimate(str(content.get("text", "")))

    for call in msg.get("tool_calls") or []:
        func = call.get("function") or {}
        tokens += estimate(func.get("name") or "")
        tokens += estimate(str(func.get("arguments") or ""))
    return tokens


def _truncate_text(text: str, max_tokens: int, estimate) -> str:
    if max_tokens <= 0:
        return ""
    if estimate(text) <= max_tokens:
        return text
    max_chars = max(8, max_tokens * 4)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    truncated = f"{head}\n...[truncated for context limits]...\n{tail}"
    if estimate(truncated) <= max_tokens or max_tokens <= 4:
        return truncated
    # If our heuristic is still over budget, shrink again with a tighter cap.
    return _truncate_text(truncated, max_tokens // 2, estimate)


def _truncate_system_message(msg: Dict[str, Any], max_tokens: int, estimate) -> Dict[str, Any]:
    truncated = dict(msg)
    content = truncated.get("content")
    if isinstance(content, str):
        truncated["content"] = _truncate_text(content, max_tokens, estimate)
    elif isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        joined = "\n".join(text_parts)
        truncated["content"] = _truncate_text(joined, max_tokens, estimate)
    return truncated


def _prune_messages_for_budget(
    messages: List[Dict[str, Any]], prompt_budget: int, estimate
) -> Tuple[List[Dict[str, Any]], int, int]:
    if not messages or prompt_budget <= 0:
        return messages, 0, 0

    system_msgs: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if idx == 0 and msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            others.append(msg)

    system_tokens = sum(_message_token_count(m, estimate) for m in system_msgs)
    if system_msgs and system_tokens > prompt_budget:
        truncated_system = _truncate_system_message(system_msgs[0], prompt_budget, estimate)
        return [truncated_system], len(messages) - 1, _message_token_count(truncated_system, estimate)

    remaining_budget = max(prompt_budget - system_tokens, 0)
    kept: List[Dict[str, Any]] = []
    used = 0
    dropped = 0
    for msg in reversed(others):
        cost = max(1, _message_token_count(msg, estimate))
        if used + cost > remaining_budget:
            dropped += 1
            continue
        kept.append(msg)
        used += cost
    kept.reverse()
    total_tokens = system_tokens + used
    dropped += len(others) - len(kept)
    return system_msgs + kept, dropped, total_tokens


def enforce_context_limits(
    payload: Dict[str, Any], settings: Settings, target_model: str
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    messages = _normalize_tool_messages(payload.get("messages") or [])
    working_payload = dict(payload)
    working_payload["messages"] = messages
    budget = settings.resolved_context_window(target_model)
    if budget <= 0 or not messages:
        return working_payload, {"dropped": 0, "before": 0, "after": 0, "budget": budget}

    estimator = _token_estimator()
    before_tokens = sum(_message_token_count(m, estimator) for m in messages)

    max_completion = payload.get("max_tokens") or payload.get("max_completion_tokens") or 0
    try:
        completion_tokens = int(max_completion)
    except (TypeError, ValueError):
        completion_tokens = 0
    if completion_tokens >= budget:
        completion_tokens = budget // 2
        working_payload["max_tokens"] = completion_tokens

    prompt_budget = max(1, budget - completion_tokens)
    trimmed_messages, dropped, _ = _prune_messages_for_budget(messages, prompt_budget, estimator)
    trimmed_messages = _normalize_tool_messages(trimmed_messages)
    final_after_tokens = sum(_message_token_count(m, estimator) for m in trimmed_messages)
    trimmed = dropped > 0 or final_after_tokens < before_tokens or final_after_tokens > prompt_budget
    working_payload["messages"] = trimmed_messages

    if not trimmed:
        return working_payload, {
            "dropped": 0,
            "before": before_tokens,
            "after": final_after_tokens,
            "budget": prompt_budget,
        }

    new_payload = dict(working_payload)
    return new_payload, {
        "dropped": dropped,
        "before": before_tokens,
        "after": final_after_tokens,
        "budget": prompt_budget,
    }

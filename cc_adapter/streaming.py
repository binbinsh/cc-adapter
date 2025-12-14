import json
import logging
from typing import Any, Dict, Optional, Tuple
from http.server import BaseHTTPRequestHandler
from .logging_utils import log_payload


def _estimate_tokens_from_chars(char_count: int) -> int:
    # Rough heuristic: ~4 chars per token; ensure at least 1
    return max(1, (char_count // 4) + 1)


def _collect_prompt_chars(incoming: Optional[Dict[str, Any]]) -> int:
    if not incoming:
        return 0
    total = 0

    def _add_content(content: Any):
        nonlocal total
        if content is None:
            return
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for item in content:
                _add_content(item)
        elif isinstance(content, dict):
            # common Anthropic shapes
            if "text" in content and isinstance(content.get("text"), str):
                total += len(content.get("text") or "")
            if "content" in content:
                _add_content(content.get("content"))

    _add_content(incoming.get("system"))
    for msg in incoming.get("messages", []):
        if isinstance(msg, dict):
            _add_content(msg.get("content"))
    return total


def estimate_prompt_tokens(incoming: Optional[Dict[str, Any]]) -> int:
    """Heuristic prompt token estimator for Anthropic-style payloads."""
    return _estimate_tokens_from_chars(_collect_prompt_chars(incoming))


def stream_openai_response(
    resp,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    debug_enabled = bool(logger) and logger.isEnabledFor(logging.DEBUG)
    encoder = lambda event, data: f"event: {event}\ndata: {json.dumps(data)}\n\n".encode(
        "utf-8"
    )

    def _send(event: str, payload: Dict[str, Any]) -> None:
        if debug_enabled:
            log_payload(logger, f"SSE -> {event}", payload)
        handler.wfile.write(encoder(event, payload))

    sent_start = False
    text_block_open = False
    thinking_block_open = False
    thinking_index = 0
    text_index = 0
    tool_blocks: Dict[str, Tuple[int, list]] = {}
    next_index = 0
    usage_state = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    output_char_count = 0

    def _init_usage_from_headers():
        header_map = {k.lower(): v for k, v in getattr(resp, "headers", {}).items()}
        prompt = header_map.get("x-openrouter-usage-prompt-tokens") or header_map.get(
            "x-openai-usage-prompt-tokens"
        )
        completion = header_map.get("x-openrouter-usage-completion-tokens") or header_map.get(
            "x-openai-usage-completion-tokens"
        )
        try:
            if prompt is not None:
                usage_state["input_tokens"] = int(prompt)
            if completion is not None:
                usage_state["output_tokens"] = int(completion)
        except ValueError:
            pass

    _init_usage_from_headers()

    def open_text_block():
        nonlocal text_block_open, text_index, next_index
        if not text_block_open:
            text_index = next_index
            next_index += 1
            _send(
                "content_block_start",
                {"type": "content_block_start", "index": text_index, "content_block": {"type": "text"}},
            )
            text_block_open = True

    def close_text_block():
        nonlocal text_block_open
        if text_block_open:
            _send(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_index},
            )
            text_block_open = False

    def open_thinking_block():
        nonlocal thinking_block_open, thinking_index, next_index
        if not thinking_block_open:
            thinking_index = next_index
            next_index += 1
            _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": thinking_index,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
            )
            thinking_block_open = True

    def close_thinking_block():
        nonlocal thinking_block_open
        if thinking_block_open:
            _send(
                "content_block_stop",
                {"type": "content_block_stop", "index": thinking_index},
            )
            thinking_block_open = False

    def get_tool_block(tool_id: str, name: str) -> Tuple[int, list]:
        nonlocal next_index
        if tool_id not in tool_blocks:
            idx = next_index
            next_index += 1
            tool_blocks[tool_id] = (idx, [])
            _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": {},
                    },
                },
            )
        return tool_blocks[tool_id]

    try:
        for line in resp.iter_lines(decode_unicode=False):
            if not line:
                continue
            if line.startswith(b"data:"):
                data = line[len(b"data:") :].strip()
                if data == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if debug_enabled:
                    log_payload(logger, "Provider stream chunk", chunk)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason")
                chunk_usage = chunk.get("usage") or {}
                for key in ("prompt_tokens", "completion_tokens"):
                    if key in chunk_usage:
                        mapped = (
                            "input_tokens" if key == "prompt_tokens" else "output_tokens"
                        )
                        usage_state[mapped] = usage_state.get(mapped, 0) + chunk_usage[key]

                if not sent_start:
                    msg_id = chunk.get("id", "stream-msg")
                    _send(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "model": requested_model or "",
                                "metadata": incoming.get("metadata") if incoming else None,
                                "cache_control": incoming.get("cache_control") if incoming else None,
                            },
                        },
                    )
                    sent_start = True

                for part in delta.get("content") or []:
                    if isinstance(part, str):
                        text = part
                        if text:
                            output_char_count += len(text)
                            open_text_block()
                            _send(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_index,
                                    "delta": {"type": "text_delta", "text": text},
                                },
                            )
                        continue

                    if part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            output_char_count += len(text)
                            open_text_block()
                            _send(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_index,
                                    "delta": {"type": "text_delta", "text": text},
                                },
                            )
                    elif part.get("type") == "reasoning":
                        text = part.get("text", "")
                        if text:
                            output_char_count += len(text)
                            open_thinking_block()
                            _send(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_index,
                                    "delta": {"type": "thinking_delta", "thinking": text},
                                },
                            )

                reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_delta:
                    reasoning_parts = []
                    if isinstance(reasoning_delta, str):
                        reasoning_parts.append(reasoning_delta)
                    elif isinstance(reasoning_delta, dict):
                        maybe_text = reasoning_delta.get("text") or reasoning_delta.get("thinking")
                        if maybe_text:
                            reasoning_parts.append(str(maybe_text))
                        if isinstance(reasoning_delta.get("content"), list):
                            reasoning_parts.extend(
                                [
                                    item.get("text", "")
                                    for item in reasoning_delta.get("content", [])
                                    if isinstance(item, dict)
                                ]
                            )
                    elif isinstance(reasoning_delta, list):
                        reasoning_parts.extend(
                            [
                                item.get("text", "")
                                for item in reasoning_delta
                                if isinstance(item, dict)
                            ]
                        )
                    for text in reasoning_parts:
                        if not text:
                            continue
                        output_char_count += len(text)
                        open_thinking_block()
                        _send(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": thinking_index,
                                "delta": {"type": "thinking_delta", "thinking": text},
                            },
                        )

                for tool in delta.get("tool_calls") or []:
                    func = tool.get("function") or {}
                    tid = tool.get("id") or func.get("name") or "tool"
                    name = func.get("name") or "tool"
                    idx, buffer = get_tool_block(tid, name)
                    args = func.get("arguments")
                    if args:
                        buffer.append(str(args))
                        _send(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": "".join(buffer),
                                },
                            },
                        )

                if finish_reason:
                    close_text_block()
                    close_thinking_block()
                    for tid, (idx, _) in tool_blocks.items():
                        _send(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": idx},
                        )
                    if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
                        usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
                        usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
                    stop_reason = _map_finish_reason(finish_reason)
                    _send(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason,
                                "stop_sequence": None,
                            },
                            "usage": usage_state,
                        },
                    )
                    _send("message_stop", {"type": "message_stop"})
                    handler.wfile.flush()
                    if debug_enabled:
                        logger.debug(
                            "Finished streaming to client (stop_reason=%s, usage=%s, output_chars=%s)",
                            stop_reason,
                            usage_state,
                            output_char_count,
                        )
                    return

                handler.wfile.flush()

        close_text_block()
        close_thinking_block()
        for tid, (idx, _) in tool_blocks.items():
            _send(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )
        if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
            usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
            usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
        _send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": usage_state,
            },
        )
        _send("message_stop", {"type": "message_stop"})
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        logger.info("Client disconnected during stream")
        handler.close_connection = True
    except Exception as exc:
        logger.exception("Error while streaming to client: %s", exc)
        try:
            if not sent_start:
                _send(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "stream-error",
                            "type": "message",
                            "role": "assistant",
                            "model": requested_model or "",
                            "metadata": incoming.get("metadata") if incoming else None,
                            "cache_control": incoming.get("cache_control") if incoming else None,
                        },
                    },
                )
                sent_start = True

            close_text_block()
            close_thinking_block()
            for tid, (idx, _) in tool_blocks.items():
                _send(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": idx},
                )

            if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
                usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
                usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))

            _send(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "error", "stop_sequence": None},
                    "usage": usage_state,
                },
            )
            _send(
                "error",
                {
                    "type": "error",
                    "message": str(exc),
                },
            )
            _send("message_stop", {"type": "message_stop"})
            handler.wfile.flush()
        except Exception:
            pass
        handler.close_connection = True
    else:
        # If provider didn't return usage, estimate roughly to avoid always-zero metrics
        if usage_state["input_tokens"] == 0 and usage_state["output_tokens"] == 0:
            output_tokens = _estimate_tokens_from_chars(output_char_count)
            input_tokens = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
            usage_state["input_tokens"] = input_tokens
            usage_state["output_tokens"] = output_tokens
        if debug_enabled:
            logger.debug(
                "Finished streaming to client (usage=%s, output_chars=%s)",
                usage_state,
                output_char_count,
            )
    finally:
        try:
            resp.close()
        except Exception:
            pass


def stream_responses_response(
    resp,
    requested_model: str,
    incoming: Optional[Dict[str, Any]],
    handler: BaseHTTPRequestHandler,
    logger,
):
    """Bridge OpenAI Responses API SSE -> Anthropic SSE events."""
    debug_enabled = bool(logger) and logger.isEnabledFor(logging.DEBUG)
    encoder = lambda event, data: f"event: {event}\ndata: {json.dumps(data)}\n\n".encode(
        "utf-8"
    )

    def _send(event: str, payload: Dict[str, Any]) -> None:
        if debug_enabled:
            log_payload(logger, f"SSE -> {event}", payload)
        handler.wfile.write(encoder(event, payload))

    sent_start = False
    text_block_open = False
    thinking_block_open = False
    thinking_index = 0
    text_index = 0
    tool_blocks: Dict[str, Tuple[int, list]] = {}
    tool_item_to_call: Dict[str, str] = {}
    tool_call_names: Dict[str, str] = {}
    next_index = 0
    usage_state = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    output_char_count = 0

    def open_text_block():
        nonlocal text_block_open, text_index, next_index
        if not text_block_open:
            text_index = next_index
            next_index += 1
            _send(
                "content_block_start",
                {"type": "content_block_start", "index": text_index, "content_block": {"type": "text"}},
            )
            text_block_open = True

    def close_text_block():
        nonlocal text_block_open
        if text_block_open:
            _send(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_index},
            )
            text_block_open = False

    def open_thinking_block():
        nonlocal thinking_block_open, thinking_index, next_index
        if not thinking_block_open:
            thinking_index = next_index
            next_index += 1
            _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": thinking_index,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
            )
            thinking_block_open = True

    def close_thinking_block():
        nonlocal thinking_block_open
        if thinking_block_open:
            _send(
                "content_block_stop",
                {"type": "content_block_stop", "index": thinking_index},
            )
            thinking_block_open = False

    def get_tool_block(call_id: str, name: str) -> Tuple[int, list]:
        nonlocal next_index
        if call_id not in tool_blocks:
            idx = next_index
            next_index += 1
            tool_blocks[call_id] = (idx, [])
            _send(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {},
                    },
                },
            )
        return tool_blocks[call_id]

    def _ensure_started(event_obj: Dict[str, Any]) -> None:
        nonlocal sent_start
        if sent_start:
            return
        response_id = event_obj.get("response_id")
        response = event_obj.get("response") if isinstance(event_obj.get("response"), dict) else None
        msg_id = response_id or (response.get("id") if response else None) or "stream-msg"
        _send(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": requested_model or "",
                    "metadata": incoming.get("metadata") if incoming else None,
                    "cache_control": incoming.get("cache_control") if incoming else None,
                },
            },
        )
        sent_start = True

    def _ingest_usage(event_obj: Dict[str, Any]) -> None:
        response = event_obj.get("response") if isinstance(event_obj.get("response"), dict) else None
        usage = None
        if response and isinstance(response.get("usage"), dict):
            usage = response.get("usage")
        elif isinstance(event_obj.get("usage"), dict):
            usage = event_obj.get("usage")
        if not isinstance(usage, dict):
            return
        if "input_tokens" in usage:
            usage_state["input_tokens"] = usage.get("input_tokens") or 0
        if "output_tokens" in usage:
            usage_state["output_tokens"] = usage.get("output_tokens") or 0
        if "cache_read_input_tokens" in usage:
            usage_state["cache_read_input_tokens"] = usage.get("cache_read_input_tokens") or 0

    try:
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

            if debug_enabled:
                log_payload(logger, "Provider responses event", event_obj)

            etype = str(event_obj.get("type") or "")
            _ensure_started(event_obj)

            if etype == "response.output_text.delta":
                delta = event_obj.get("delta") or ""
                if delta:
                    output_char_count += len(str(delta))
                    open_text_block()
                    _send(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": str(delta)},
                        },
                    )
            elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_summary_part.added"):
                delta = event_obj.get("delta")
                if delta:
                    output_char_count += len(str(delta))
                    open_thinking_block()
                    _send(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": thinking_index,
                            "delta": {"type": "thinking_delta", "thinking": str(delta)},
                        },
                    )
            elif etype == "response.output_item.added":
                item = event_obj.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "function_call":
                    item_id = item.get("id") or ""
                    call_id = item.get("call_id") or item_id or "tool_call"
                    name = item.get("name") or "tool"
                    tool_item_to_call[str(item_id)] = str(call_id)
                    tool_call_names[str(call_id)] = str(name)
                    idx, buffer = get_tool_block(str(call_id), str(name))
                    args = item.get("arguments")
                    if args:
                        buffer.append(str(args))
                        _send(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": "".join(buffer),
                                },
                            },
                        )
            elif etype == "response.function_call_arguments.delta":
                item_id = str(event_obj.get("item_id") or "")
                call_id = tool_item_to_call.get(item_id) or item_id or "tool_call"
                name = tool_call_names.get(call_id) or "tool"
                delta = event_obj.get("delta") or ""
                if delta:
                    idx, buffer = get_tool_block(call_id, name)
                    buffer.append(str(delta))
                    _send(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": "".join(buffer),
                            },
                        },
                    )
            elif etype == "response.function_call_arguments.done":
                item_id = str(event_obj.get("item_id") or "")
                call_id = tool_item_to_call.get(item_id) or item_id or "tool_call"
                name = tool_call_names.get(call_id) or "tool"
                args = event_obj.get("arguments")
                if args is not None:
                    idx, buffer = get_tool_block(call_id, name)
                    buffer[:] = [str(args)]
                    _send(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": "".join(buffer)},
                        },
                    )
            elif etype == "response.output_item.done":
                item = event_obj.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "function_call":
                    item_id = str(item.get("id") or "")
                    call_id = str(item.get("call_id") or tool_item_to_call.get(item_id) or item_id or "tool_call")
                    name = str(item.get("name") or tool_call_names.get(call_id) or "tool")
                    tool_item_to_call[item_id] = call_id
                    tool_call_names[call_id] = name
                    args = item.get("arguments")
                    if args is not None:
                        idx, buffer = get_tool_block(call_id, name)
                        if "".join(buffer) != str(args):
                            buffer[:] = [str(args)]
                            _send(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": "".join(buffer),
                                    },
                                },
                            )
            elif etype in ("response.completed", "response.done", "response.finished", "response.failed"):
                _ingest_usage(event_obj)
                close_text_block()
                close_thinking_block()
                for _, (idx, _) in tool_blocks.items():
                    _send(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
                if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
                    usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
                    usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
                stop_reason = "tool_use" if tool_blocks else "end_turn"
                _send(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": usage_state,
                    },
                )
                _send("message_stop", {"type": "message_stop"})
                handler.wfile.flush()
                return
            elif etype == "error":
                raise RuntimeError(str(event_obj.get("message") or event_obj.get("error") or "Unknown error"))

            handler.wfile.flush()

        close_text_block()
        close_thinking_block()
        for _, (idx, _) in tool_blocks.items():
            _send("content_block_stop", {"type": "content_block_stop", "index": idx})
        if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
            usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
            usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
        stop_reason = "tool_use" if tool_blocks else "end_turn"
        _send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage_state,
            },
        )
        _send("message_stop", {"type": "message_stop"})
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        logger.info("Client disconnected during stream")
        handler.close_connection = True
    except Exception as exc:
        logger.exception("Error while streaming to client: %s", exc)
        try:
            if not sent_start:
                _send(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "stream-error",
                            "type": "message",
                            "role": "assistant",
                            "model": requested_model or "",
                            "metadata": incoming.get("metadata") if incoming else None,
                            "cache_control": incoming.get("cache_control") if incoming else None,
                        },
                    },
                )
                sent_start = True

            close_text_block()
            close_thinking_block()
            for _, (idx, _) in tool_blocks.items():
                _send("content_block_stop", {"type": "content_block_stop", "index": idx})
            if usage_state.get("input_tokens", 0) == 0 and usage_state.get("output_tokens", 0) == 0:
                usage_state["output_tokens"] = _estimate_tokens_from_chars(output_char_count)
                usage_state["input_tokens"] = _estimate_tokens_from_chars(_collect_prompt_chars(incoming))
            _send(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "error", "stop_sequence": None},
                    "usage": usage_state,
                },
            )
            _send("error", {"type": "error", "message": str(exc)})
            _send("message_stop", {"type": "message_stop"})
            handler.wfile.flush()
        except Exception:
            pass
        handler.close_connection = True
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _map_finish_reason(reason: Any) -> Any:
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "tool_call": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "content_filter",
        None: "end_turn",
    }
    return mapping.get(reason, reason)

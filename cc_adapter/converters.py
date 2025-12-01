import json
from typing import Any, Dict, List, Optional, Set


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "input_text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "tool_result":
                    parts.append(_flatten_text(item.get("content")))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text", ""))
    return ""


def _handle_system(system: Any) -> Optional[Dict[str, str]]:
    system_prompt = _flatten_text(system)
    if system_prompt:
        return {"role": "system", "content": system_prompt}
    return None


def _handle_image(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = part.get("source") or {}
    media_type = source.get("media_type") or source.get("type")
    if not media_type:
        return None
    if source.get("type") == "base64" and source.get("data"):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source['data']}"},
        }
    if source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _reasoning_effort(budget: Optional[int]) -> str:
    if not budget:
        return "medium"
    if budget <= 4000:
        return "low"
    if budget <= 16000:
        return "medium"
    return "high"


def _map_thinking(thinking: Any) -> Optional[Dict[str, Any]]:
    if thinking is True:
        return {"effort": "medium"}
    if isinstance(thinking, dict):
        budget = thinking.get("budget_tokens") or thinking.get("max_tokens")
        return {"effort": _reasoning_effort(budget)}
    return None


def anthropic_to_openai(body: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    pending_tool_calls = False
    pending_tool_call_ids: Set[str] = set()
    last_assistant_message: Optional[Dict[str, Any]] = None

    system_msg = _handle_system(body.get("system"))
    if system_msg:
        if isinstance(body.get("cache_control"), dict):
            system_msg["cache_control"] = body.get("cache_control")
        messages.append(system_msg)

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            user_text: List[str] = []
            tool_results: List[Dict[str, Any]] = []
            images: List[Dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        if isinstance(part, str):
                            user_text.append(part)
                        continue
                    if part.get("type") == "tool_result":
                        tool_use_id = part.get("tool_use_id") or part.get("id")
                        if pending_tool_call_ids and tool_use_id in pending_tool_call_ids:
                            tool_results.append(part)
                        else:
                            flattened = _flatten_text(part.get("content")) or ""
                            if flattened:
                                user_text.append(flattened)
                    elif part.get("type") == "text":
                        user_text.append(str(part.get("text", "")))
                    elif part.get("type") == "image":
                        maybe_img = _handle_image(part)
                        if maybe_img:
                            images.append(maybe_img)
            elif isinstance(content, str):
                user_text.append(content)

            def _append_user_message():
                mixed: List[Any] = []
                if user_text:
                    mixed.append({"type": "text", "text": "\n".join(user_text)})
                mixed.extend(images)
                if not mixed:
                    return
                user_msg: Dict[str, Any] = {"role": "user", "content": mixed}
                if msg.get("cache_control"):
                    user_msg["cache_control"] = msg.get("cache_control")
                messages.append(user_msg)

            if tool_results:
                if pending_tool_call_ids:
                    emitted_tool_message = False
                    for result in tool_results:
                        tool_id = (
                            result.get("tool_use_id")
                            or result.get("id")
                            or "tool_call"
                        )
                        if pending_tool_call_ids and tool_id not in pending_tool_call_ids:
                            flattened = _flatten_text(result.get("content")) or ""
                            if flattened:
                                user_text.append(flattened)
                            continue
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": _flatten_text(result.get("content")) or "",
                            }
                        )
                        emitted_tool_message = True
                    pending_tool_calls = False
                    pending_tool_call_ids.clear()
                    if (
                        not emitted_tool_message
                        and last_assistant_message
                        and "tool_calls" in last_assistant_message
                    ):
                        last_assistant_message.pop("tool_calls", None)
                    _append_user_message()
                else:
                    for result in tool_results:
                        flattened = _flatten_text(result.get("content")) or ""
                        if flattened:
                            user_text.append(flattened)
                    _append_user_message()
                    if pending_tool_calls and last_assistant_message and "tool_calls" in last_assistant_message:
                        # We expected tool outputs but none arrived; drop stale tool_calls.
                        last_assistant_message.pop("tool_calls", None)
                        pending_tool_calls = False
                        pending_tool_call_ids.clear()
            else:
                _append_user_message()
                if pending_tool_calls:
                    # Assistant tool_calls must be followed by tool results; drop stale tool_calls.
                    if last_assistant_message and "tool_calls" in last_assistant_message:
                        last_assistant_message.pop("tool_calls", None)
                    pending_tool_calls = False
                    pending_tool_call_ids.clear()
        elif role == "assistant":
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
                        elif part.get("type") == "tool_use":
                            try:
                                arg_text = json.dumps(part.get("input", {}))
                            except TypeError:
                                arg_text = "{}"
                            tool_calls.append(
                                {
                                    "id": part.get("id")
                                    or part.get("tool_use_id")
                                    or part.get("name")
                                    or "tool_call",
                                    "type": "function",
                                    "function": {
                                        "name": part.get("name") or "tool",
                                        "arguments": arg_text,
                                    },
                                }
                            )
                    elif isinstance(part, str):
                        text_parts.append(part)
            elif isinstance(content, str):
                text_parts.append(content)
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_message["content"] = "\n".join(text_parts)
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                pending_tool_calls = True
                pending_tool_call_ids = {call["id"] for call in tool_calls if call.get("id")}
            else:
                pending_tool_calls = False
                pending_tool_call_ids.clear()
            if msg.get("cache_control"):
                assistant_message["cache_control"] = msg.get("cache_control")
            messages.append(assistant_message)
            last_assistant_message = assistant_message

    if pending_tool_calls and last_assistant_message:
        # No tool results observed after tool_calls; drop them to avoid provider 400.
        last_assistant_message.pop("tool_calls", None)
        if not last_assistant_message.get("content"):
            last_assistant_message["content"] = ""
    openai_payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    if body.get("max_tokens") or body.get("max_output_tokens"):
        openai_payload["max_tokens"] = body.get("max_tokens") or body.get(
            "max_output_tokens"
        )
    if body.get("temperature") is not None:
        openai_payload["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        openai_payload["top_p"] = body.get("top_p")
    if body.get("stop_sequences"):
        openai_payload["stop"] = body.get("stop_sequences")

    tools = _convert_tools(body.get("tools"))
    if tools:
        openai_payload["tools"] = tools

    tool_choice = body.get("tool_choice")
    if tool_choice:
        if isinstance(tool_choice, str):
            if tool_choice in ("none", "auto", "any"):
                openai_payload["tool_choice"] = (
                    "auto" if tool_choice == "any" else tool_choice
                )
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool" and tool_choice.get("name"):
                openai_payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            elif tool_choice.get("type") in ("none", "auto", "any"):
                openai_payload["tool_choice"] = (
                    "auto" if tool_choice.get("type") == "any" else tool_choice.get("type")
                )

    thinking = body.get("thinking")
    mapped_thinking = _map_thinking(thinking)
    if mapped_thinking:
        openai_payload["reasoning"] = mapped_thinking

    return openai_payload


def _convert_tools(tools: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not name or not schema:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
        )
    return converted


def _openai_content_to_blocks(content: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if content is None:
        return blocks
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item:
                    blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    blocks.append({"type": "text", "text": str(item.get("text", ""))})
                elif "text" in item:
                    blocks.append({"type": "text", "text": str(item.get("text", ""))})
    elif isinstance(content, dict) and "text" in content:
        blocks.append({"type": "text", "text": str(content.get("text", ""))})
    return blocks


def _parse_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, (dict, list)):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
    return {"raw": arguments}


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


def _estimate_tokens_from_chars(char_count: int) -> int:
    return max(1, (char_count // 4) + 1)


def openai_to_anthropic(
    data: Dict[str, Any], requested_model: str, original_request: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LM Studio returned no choices")

    all_blocks: List[Dict[str, Any]] = []
    stop_reason = None
    usage_block: Dict[str, Any] = {}
    output_chars = 0

    def _extract_thinking(reasoning: Any) -> str:
        if reasoning is None:
            return ""
        if isinstance(reasoning, str):
            return reasoning
        if isinstance(reasoning, list):
            return "".join(
                [
                    rc.get("text", "")
                    for rc in reasoning
                    if isinstance(rc, dict) and "text" in rc
                ]
            )
        if isinstance(reasoning, dict):
            if "thinking" in reasoning:
                return str(reasoning.get("thinking") or "")
            if "content" in reasoning and isinstance(reasoning["content"], list):
                return "".join(
                    [
                        item.get("text", "")
                        for item in reasoning["content"]
                        if isinstance(item, dict)
                    ]
                )
        return ""

    for idx, choice in enumerate(choices):
        message = choice.get("message") or {}

        blocks = _openai_content_to_blocks(message.get("content"))

        reasoning_content = message.get("reasoning_content") or message.get("reasoning")
        thinking_text = _extract_thinking(reasoning_content)
        if thinking_text:
            blocks.insert(
                0,
                {
                    "type": "thinking",
                    "thinking": thinking_text,
                    "signature": "",
                },
            )

        for call in message.get("tool_calls") or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") or call.get("function", {}).get("name", "tool"),
                    "name": (call.get("function") or {}).get("name") or "tool",
                    "input": _parse_arguments((call.get("function") or {}).get("arguments")),
                }
            )

        if idx > 0:
            all_blocks.append({"type": "text", "text": f"[alternative choice {idx}]"})
        all_blocks.extend(blocks)
        for blk in blocks:
            if blk.get("type") in ("text", "thinking"):
                output_chars += len(blk.get("text") or blk.get("thinking") or "")
        if stop_reason is None:
            stop_reason = _map_finish_reason(choice.get("finish_reason"))

    usage = data.get("usage") or {}
    if "prompt_tokens" in usage:
        usage_block["input_tokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        usage_block["output_tokens"] = usage["completion_tokens"]
    if not usage_block:
        usage_block["output_tokens"] = _estimate_tokens_from_chars(output_chars)
        if original_request:
            # very rough prompt estimation from raw text parts
            prompt_chars = 0
            system = original_request.get("system")
            prompt_chars += len(_flatten_text(system))
            for msg in original_request.get("messages", []):
                if isinstance(msg, dict):
                    prompt_chars += len(_flatten_text(msg.get("content")))
            usage_block["input_tokens"] = _estimate_tokens_from_chars(prompt_chars)

    if not all_blocks:
        all_blocks = [{"type": "text", "text": ""}]

    response: Dict[str, Any] = {
        "id": data.get("id", "local-msg"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": all_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
    }
    if usage_block:
        response["usage"] = usage_block
    if original_request:
        if original_request.get("metadata") is not None:
            response["metadata"] = original_request.get("metadata")
        if original_request.get("cache_control") is not None:
            response["cache_control"] = original_request.get("cache_control")
    return response


def build_poe_params(body: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    thinking = body.get("thinking") or {}
    budget = thinking.get("budget_tokens")
    if budget:
        params["thinking_budget"] = budget

    tools = body.get("tools") or []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("function", {}).get("name") == "web_search":
            params["web_search"] = True
            break

    return params

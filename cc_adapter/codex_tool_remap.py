import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ToolInfo:
    name: str
    schema: Dict[str, Any]


def _flatten_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item.get("text") or ""))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict) and content.get("text") is not None:
        return str(content.get("text") or "")
    return str(content)


def extract_working_directory(system: Any) -> Optional[str]:
    text = _flatten_text(system)
    match = re.search(r"^\s*Working directory:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _normalize_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = tool.get("input_schema")
    return schema if isinstance(schema, dict) else {}


def _index_tools(tools: Any) -> Dict[str, ToolInfo]:
    out: Dict[str, ToolInfo] = {}
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_tool_name(name)
        if not key or key in out:
            continue
        out[key] = ToolInfo(name=name, schema=_tool_schema(tool))
    return out


def _resolve_tool(tools_index: Dict[str, ToolInfo], candidates: Sequence[str]) -> Optional[ToolInfo]:
    for cand in candidates:
        key = _normalize_tool_name(cand)
        if key in tools_index:
            return tools_index[key]
    for cand in candidates:
        key = _normalize_tool_name(cand)
        if not key:
            continue
        for k, info in tools_index.items():
            if key in k:
                return info
    return None


def _schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def _schema_required(schema: Dict[str, Any]) -> List[str]:
    req = schema.get("required")
    if isinstance(req, list):
        return [str(x) for x in req if isinstance(x, str)]
    return []


def _pick_key(props: Dict[str, Any], candidates: Sequence[str]) -> str:
    for cand in candidates:
        if cand in props:
            return cand
    normalized = {_normalize_tool_name(k): k for k in props.keys()}
    for cand in candidates:
        key = _normalize_tool_name(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = _normalize_tool_name(cand)
        for nk, original in normalized.items():
            if key and key in nk:
                return original
    return ""


def _parse_jsonish(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": value}
        except Exception:
            return {"raw": value}
    return {"raw": value}


def _build_bash_input(tool: ToolInfo, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    props = _schema_properties(tool.schema)
    required = _schema_required(tool.schema)
    command = args.get("command") or args.get("cmd") or args.get("shell_command")
    if not command:
        return None

    command_key = _pick_key(props, ["command", "cmd"])
    if not command_key:
        return None
    out: Dict[str, Any] = {command_key: command}

    workdir_value = args.get("workdir") or args.get("cwd") or args.get("dir")
    if workdir_value:
        workdir_key = _pick_key(props, ["workdir", "cwd", "directory"])
        if workdir_key:
            out[workdir_key] = workdir_value

    timeout_value = args.get("timeout_ms") or args.get("timeout")
    if timeout_value is not None:
        timeout_key = _pick_key(props, ["timeout_ms", "timeout"])
        if timeout_key:
            out[timeout_key] = timeout_value

    if "description" in required:
        desc_key = _pick_key(props, ["description", "desc"])
        if desc_key and desc_key not in out:
            out[desc_key] = "Runs a shell command"

    return out


def _build_write_input(tool: ToolInfo, file_path: str, content: str) -> Optional[Dict[str, Any]]:
    props = _schema_properties(tool.schema)
    required = _schema_required(tool.schema)
    path_key = _pick_key(
        props,
        [
            "filePath",
            "file_path",
            "file",
            "path",
            "filepath",
            "filename",
        ],
    )
    content_key = _pick_key(props, ["content", "text", "data", "contents"])
    if not path_key or not content_key:
        return None
    out: Dict[str, Any] = {path_key: file_path, content_key: content}

    if "overwrite" in required and "overwrite" not in out:
        out["overwrite"] = True
    return out


def _build_delete_input(tool: ToolInfo, file_path: str) -> Optional[Dict[str, Any]]:
    props = _schema_properties(tool.schema)
    path_key = _pick_key(props, ["filePath", "file_path", "path", "filepath", "filename"])
    if not path_key:
        return None
    return {path_key: file_path}


def _safe_abspath(base_dir: Path, path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        resolved = raw
    else:
        resolved = (base_dir / raw).resolve()
    base_resolved = base_dir.resolve()
    try:
        resolved.relative_to(base_resolved)
    except Exception:
        # Disallow escaping the base directory for safety.
        raise ValueError(f"Refusing to access path outside working directory: {path_str}")
    return resolved


def _split_patch(text: str) -> List[str]:
    return (text or "").replace("\r\n", "\n").split("\n")


def _parse_apply_patch(patch_text: str) -> List[Dict[str, Any]]:
    lines = _split_patch(patch_text)
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("apply_patch payload missing '*** Begin Patch' header")
    if "*** End Patch" not in [l.strip() for l in lines]:
        raise ValueError("apply_patch payload missing '*** End Patch' footer")

    ops: List[Dict[str, Any]] = []
    i = 1
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.strip() == "*** End Patch":
            break
        if line.startswith("*** Add File: "):
            path = line[len("*** Add File: ") :].strip()
            i += 1
            content_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                raw = lines[i]
                if raw.startswith("+"):
                    content_lines.append(raw[1:])
                elif raw == "":
                    content_lines.append("")
                else:
                    # Best-effort: tolerate missing '+' prefix.
                    content_lines.append(raw)
                i += 1
            ops.append({"action": "add", "path": path, "content": "\n".join(content_lines)})
            continue
        if line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: ") :].strip()
            i += 1
            ops.append({"action": "delete", "path": path})
            continue
        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: ") :].strip()
            i += 1
            move_to = None
            if i < len(lines) and lines[i].startswith("*** Move to: "):
                move_to = lines[i][len("*** Move to: ") :].strip()
                i += 1
            patch_lines: List[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                patch_lines.append(lines[i])
                i += 1
            ops.append({"action": "update", "path": path, "move_to": move_to, "patch_lines": patch_lines})
            continue
        i += 1
    return ops


def _find_subsequence(haystack: List[str], needle: List[str]) -> Optional[int]:
    if not needle:
        return 0
    for start in range(0, len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def _apply_patch_lines_to_text(original: str, patch_lines: List[str]) -> str:
    original = (original or "").replace("\r\n", "\n")
    ends_with_newline = original.endswith("\n")
    lines = original.splitlines()

    hunks: List[List[str]] = []
    current: List[str] = []
    for raw in patch_lines:
        if raw.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        if not raw:
            continue
        if raw[0] in (" ", "+", "-"):
            current.append(raw)
        else:
            current.append(" " + raw)
    if current:
        hunks.append(current)

    out_lines = list(lines)
    for hunk in hunks:
        old_seq = [ln[1:] for ln in hunk if ln and ln[0] in (" ", "-")]
        new_seq = [ln[1:] for ln in hunk if ln and ln[0] in (" ", "+")]
        if not old_seq and not new_seq:
            continue
        pos = _find_subsequence(out_lines, old_seq)
        if pos is None:
            raise ValueError("Failed to apply patch hunk (context not found)")
        out_lines = out_lines[:pos] + new_seq + out_lines[pos + len(old_seq) :]

    rendered = "\n".join(out_lines)
    if ends_with_newline:
        rendered += "\n"
    return rendered


def _extract_patch_text(args: Dict[str, Any]) -> str:
    if isinstance(args.get("patch"), str):
        return args["patch"]
    if isinstance(args.get("text"), str):
        return args["text"]
    if isinstance(args.get("command"), list) and len(args["command"]) >= 2:
        maybe = args["command"][1]
        if isinstance(maybe, str):
            return maybe
    if isinstance(args.get("command"), str):
        return args["command"]
    if isinstance(args.get("input"), str):
        return args["input"]
    return ""


def remap_codex_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: Any,
    incoming: Optional[Dict[str, Any]],
) -> Optional[List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Return a list of (new_call_id, new_tool_name, new_input) if remapping is needed.

    Remapping is best-effort and only targets common Codex CLI tool names.
    """
    if not incoming:
        return None
    tools_index = _index_tools(incoming.get("tools"))
    if not tools_index:
        return None

    available_normalized = set(tools_index.keys())
    normalized_name = _normalize_tool_name(name)
    if normalized_name in available_normalized:
        # Tool exists (maybe different casing); let it pass through.
        actual = tools_index[normalized_name].name
        if actual == name:
            return None
        parsed = _parse_jsonish(arguments)
        return [(call_id, actual, parsed)]

    args = _parse_jsonish(arguments)

    # shell_command -> bash
    if normalized_name == "shellcommand":
        bash_tool = _resolve_tool(tools_index, ["bash"])
        if not bash_tool:
            return None
        built = _build_bash_input(bash_tool, args)
        if not built:
            return None
        return [(call_id, bash_tool.name, built)]

    # update_plan/read_plan -> todowrite/todoread (best-effort)
    if normalized_name == "updateplan":
        todo = _resolve_tool(tools_index, ["todowrite", "todo_write", "todo"])
        if not todo:
            return None
        props = _schema_properties(todo.schema)
        out: Dict[str, Any] = {}
        key = _pick_key(props, ["todos", "items", "plan"])
        if key:
            out[key] = args.get("plan") if isinstance(args.get("plan"), list) else args.get(key) or []
        if not out:
            # Fallback: pass through only known keys.
            for k in props.keys():
                if k in args:
                    out[k] = args[k]
        return [(call_id, todo.name, out or args)]

    if normalized_name == "readplan":
        todo_read = _resolve_tool(tools_index, ["todoread", "todo_read"])
        if not todo_read:
            return None
        return [(call_id, todo_read.name, {})]

    # apply_patch -> write/delete operations (best-effort)
    if normalized_name == "applypatch":
        patch_text = _extract_patch_text(args)
        if "*** Begin Patch" not in (patch_text or ""):
            return None

        base = extract_working_directory(incoming.get("system")) or os.getcwd()
        base_dir = Path(base)

        write_tool = _resolve_tool(tools_index, ["write"])
        delete_tool = _resolve_tool(tools_index, ["delete", "remove", "rm", "unlink"])
        bash_tool = _resolve_tool(tools_index, ["bash"])
        if not write_tool and not delete_tool and not bash_tool:
            return None

        ops = _parse_apply_patch(patch_text)
        remapped: List[Tuple[str, str, Dict[str, Any]]] = []
        next_suffix = 0

        for op in ops:
            action = op.get("action")
            rel_path = str(op.get("path") or "").strip()
            if not rel_path:
                continue
            abs_path = _safe_abspath(base_dir, rel_path)

            if action == "add":
                if not write_tool:
                    continue
                content = str(op.get("content") or "")
                write_input = _build_write_input(write_tool, str(abs_path), content)
                if not write_input:
                    continue
                new_id = call_id if next_suffix == 0 else f"{call_id}:{next_suffix}"
                next_suffix += 1
                remapped.append((new_id, write_tool.name, write_input))
                continue

            if action == "delete":
                if delete_tool:
                    delete_input = _build_delete_input(delete_tool, str(abs_path))
                    if delete_input:
                        new_id = call_id if next_suffix == 0 else f"{call_id}:{next_suffix}"
                        next_suffix += 1
                        remapped.append((new_id, delete_tool.name, delete_input))
                        continue
                if bash_tool:
                    bash_input = _build_bash_input(bash_tool, {"command": f'rm -f "{abs_path}"'})
                    if bash_input:
                        new_id = call_id if next_suffix == 0 else f"{call_id}:{next_suffix}"
                        next_suffix += 1
                        remapped.append((new_id, bash_tool.name, bash_input))
                        continue
                continue

            if action == "update":
                if not write_tool:
                    continue
                patch_lines = op.get("patch_lines") or []
                if not isinstance(patch_lines, list):
                    continue
                old_text = abs_path.read_text(encoding="utf-8", errors="replace")
                new_text = _apply_patch_lines_to_text(old_text, [str(x) for x in patch_lines])
                dest_rel = str(op.get("move_to") or "").strip()
                dest_path = _safe_abspath(base_dir, dest_rel) if dest_rel else abs_path
                write_input = _build_write_input(write_tool, str(dest_path), new_text)
                if not write_input:
                    continue
                new_id = call_id if next_suffix == 0 else f"{call_id}:{next_suffix}"
                next_suffix += 1
                remapped.append((new_id, write_tool.name, write_input))
                if dest_path != abs_path:
                    if delete_tool:
                        delete_input = _build_delete_input(delete_tool, str(abs_path))
                        if delete_input:
                            remapped.append((f"{call_id}:{next_suffix}", delete_tool.name, delete_input))
                            next_suffix += 1
                    elif bash_tool:
                        bash_input = _build_bash_input(bash_tool, {"command": f'rm -f "{abs_path}"'})
                        if bash_input:
                            remapped.append((f"{call_id}:{next_suffix}", bash_tool.name, bash_input))
                            next_suffix += 1
                continue

        return remapped or None

    return None

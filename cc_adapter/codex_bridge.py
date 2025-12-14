import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _normalize_mode(value: str) -> str:
    return (value or "").strip().lower()


def _truthy(value: str) -> bool:
    return _normalize_mode(value) in {"1", "true", "yes", "on", "always"}


def _falsy(value: str) -> bool:
    return _normalize_mode(value) in {"0", "false", "no", "off", "disable", "disabled", "never"}


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return str(tool["function"].get("name") or "").strip()
    return str(tool.get("name") or "").strip()


def _normalize_tool_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _unique_tool_names(tools: Any) -> List[str]:
    names: List[str] = []
    seen = set()
    for tool in tools or []:
        name = _tool_name(tool)
        if not name:
            continue
        key = _normalize_tool_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return sorted(names, key=lambda s: s.lower())


def _find_tool_name(tool_names: Sequence[str], candidates: Sequence[str]) -> str:
    normalized = {_normalize_tool_name(n): n for n in tool_names}
    for cand in candidates:
        key = _normalize_tool_name(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = _normalize_tool_name(cand)
        if not key:
            continue
        for k, original in normalized.items():
            if key in k:
                return original
    return ""


def looks_like_claude_code_system(prompt: str) -> bool:
    """
    Heuristic detector for Claude Code's *default* (very large) system prompt.

    Claude Code also uses smaller, task-specific system prompts (e.g. command
    file-path extraction). In `auto` strip mode we only want to strip the large
    default prompt to reduce conflicts with Codex CLI instructions, while keeping
    smaller specialized prompts intact.
    """
    raw = prompt or ""
    text = raw.lower()
    if not text.strip():
        return False
    # The default Claude Code system prompt is extremely large. Using a size gate
    # avoids stripping small helper prompts that happen to mention "Claude Code".
    if len(text) < 4000:
        return False
    return (
        "todowrite tool" in text
        or "webfetch tool" in text
        or "available tools" in text
        or "claude-code-guide" in text
        or "anthropics/claude-code" in text
        or "you are claude code" in text
    )


def extract_user_instruction_blocks(system_prompt: str) -> str:
    """
    Best-effort extraction of project/user instruction blocks embedded by clients.

    Currently keeps any "Instructions from:" sections (used by some CLIs) verbatim.
    """
    text = system_prompt or ""
    lines = text.splitlines()
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Instructions from:"):
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith("Instructions from:"):
                i += 1
            block = "\n".join(lines[start:i]).strip()
            if block:
                blocks.append(block)
            continue
        i += 1
    return "\n\n".join(blocks).strip()


def load_bridge_prompt(path: str) -> str:
    candidate = (path or "").strip()
    if not candidate:
        return ""
    expanded = os.path.expanduser(candidate)
    p = Path(expanded)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def should_inject_bridge(mode: str, tools: Any) -> bool:
    text = _normalize_mode(mode)
    if _falsy(text):
        return False
    if _truthy(text):
        return True
    # auto
    return bool(_unique_tool_names(tools))


def should_strip_system(mode: str, system_prompt: str) -> bool:
    text = _normalize_mode(mode)
    if _falsy(text):
        return False
    if _truthy(text):
        return True
    # auto
    return looks_like_claude_code_system(system_prompt)


def build_claude_code_bridge_prompt(tools: Any) -> str:
    tool_names = _unique_tool_names(tools)
    bash_tool = _find_tool_name(tool_names, ["bash"])
    edit_tool = _find_tool_name(tool_names, ["edit"])
    write_tool = _find_tool_name(tool_names, ["write"])
    read_tool = _find_tool_name(tool_names, ["read"])
    todo_write = _find_tool_name(tool_names, ["todowrite", "todo_write", "todo"])
    todo_read = _find_tool_name(tool_names, ["todoread", "todo_read"])

    mapping_lines: List[str] = []
    if bash_tool:
        mapping_lines.append(f"- `shell_command` → `{bash_tool}`")
    if edit_tool:
        mapping_lines.append(f"- `apply_patch` → `{edit_tool}`")
    elif write_tool:
        mapping_lines.append(f"- `apply_patch` → `{write_tool}`")
    if todo_write:
        if todo_read:
            mapping_lines.append(f"- `update_plan`/`read_plan` → `{todo_write}`/`{todo_read}`")
        else:
            mapping_lines.append(f"- `update_plan` → `{todo_write}`")

    tools_line = ", ".join([f"`{n}`" for n in tool_names]) if tool_names else "(none)"

    lines = [
        "# Codex via Claude Code (cc-adapter)",
        "",
        "You are running OpenAI Codex through cc-adapter inside Claude Code.",
        "The Codex CLI system prompt may mention tools that do NOT exist here (e.g. `apply_patch`, `update_plan`, `shell_command`).",
        "Use ONLY the tools provided in this session, and follow each tool's JSON schema exactly.",
        "",
    ]
    if mapping_lines:
        lines.extend(
            [
                "Codex-CLI tool name mapping (if you were about to use them):",
                *mapping_lines,
                "",
            ]
        )

    lines.extend(
        [
            f"Available tools: {tools_line}",
            "",
            "Rules:",
            "- Never call a tool that is not in `Available tools`.",
            "- If a capability is missing, ask the user instead of guessing tool names or schemas.",
        ]
    )

    # Keep the prompt short; avoid repeating tool schemas (they are provided separately).
    return "\n".join(lines).strip()


def split_system_prompt(system_prompt: str, strip_mode: str) -> Tuple[str, str]:
    """
    Return (kept_system_prompt, extracted_user_instructions).

    In strip modes, tries to preserve explicit "Instructions from:" blocks while
    removing the rest of the system prompt to reduce conflicts with Codex CLI instructions.
    """
    prompt = system_prompt or ""
    if not should_strip_system(strip_mode, prompt):
        return prompt, ""
    extracted = extract_user_instruction_blocks(prompt)
    return "", extracted

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cc_adapter.config import Settings
from cc_adapter.converters import openai_to_anthropic
from cc_adapter.providers import codex


class CodexBridgeInjectionTestCase(unittest.TestCase):
    def test_request_body_prepends_claude_bridge_developer_message(self):
        payload = {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "run shell commands",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            ],
            "stream": True,
        }
        settings = Settings(lmstudio_timeout=1, codex_bridge="on")
        with mock.patch.object(codex, "get_codex_instructions", return_value="codex-cli-prompt"):
            body = codex._request_body(payload, settings, model_key="codex:gpt-5.2-medium")
        self.assertEqual(body["instructions"], "codex-cli-prompt")
        self.assertIsInstance(body.get("input"), list)
        self.assertGreaterEqual(len(body["input"]), 2)
        first = body["input"][0]
        self.assertEqual(first.get("type"), "message")
        self.assertEqual(first.get("role"), "developer")
        text = (first.get("content") or [{}])[0].get("text") or ""
        self.assertIn("Codex via Claude Code", text)


class CodexToolRemapTestCase(unittest.TestCase):
    def test_shell_command_is_remapped_to_bash_tool(self):
        original_request = {
            "system": "Working directory: /tmp",
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run shell commands",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
        }
        data = {
            "id": "cmpl-1",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "shell_command",
                                    "arguments": json.dumps({"command": "echo hi"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        out = openai_to_anthropic(data, "codex:gpt-5.2", original_request)
        tool_blocks = [b for b in out.get("content", []) if b.get("type") == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "Bash")
        self.assertEqual(tool_blocks[0]["input"], {"command": "echo hi"})

    def test_apply_patch_is_translated_into_write_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / "hello.txt"
            target.write_text("hello\n", encoding="utf-8")

            patch_text = "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: hello.txt",
                    "@@",
                    "-hello",
                    "+hi",
                    "*** End Patch",
                ]
            )

            original_request = {
                "system": f"Working directory: {base}",
                "tools": [
                    {
                        "name": "Write",
                        "description": "Write a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "filePath": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filePath", "content"],
                        },
                    }
                ],
            }
            data = {
                "id": "cmpl-1",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "apply_patch",
                                        "arguments": json.dumps({"command": ["apply_patch", patch_text]}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }

            out = openai_to_anthropic(data, "codex:gpt-5.2", original_request)
            tool_blocks = [b for b in out.get("content", []) if b.get("type") == "tool_use"]
            self.assertEqual(len(tool_blocks), 1)
            self.assertEqual(tool_blocks[0]["name"], "Write")
            expected_path = str(target.resolve())
            self.assertEqual(tool_blocks[0]["input"]["filePath"], expected_path)
            self.assertEqual(tool_blocks[0]["input"]["content"], "hi\n")


if __name__ == "__main__":
    unittest.main()


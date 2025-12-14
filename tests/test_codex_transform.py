import unittest

import base64
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from cc_adapter.providers import codex


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def b64url(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode("utf-8")).decode("utf-8").rstrip("=")

    return f"{b64url(header)}.{b64url(payload)}.sig"


class CodexTransformTestCase(unittest.TestCase):
    def test_messages_to_responses_input_uses_output_text_for_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]

        _, items = codex._messages_to_responses_input(messages)
        self.assertEqual(items[0]["type"], "message")
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[0]["content"][0]["type"], "input_text")
        self.assertEqual(items[0]["content"][0]["text"], "hi")

        self.assertEqual(items[1]["type"], "message")
        self.assertEqual(items[1]["role"], "assistant")
        self.assertEqual(items[1]["content"][0]["type"], "output_text")
        self.assertEqual(items[1]["content"][0]["text"], "hello")

    def test_messages_to_responses_input_maps_tools(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "do_it", "arguments": "{\"x\":1}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]

        developer_prompt, items = codex._messages_to_responses_input(messages)
        self.assertIn("sys", developer_prompt)

        function_calls = [i for i in items if i.get("type") == "function_call"]
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0]["call_id"], "call_1")
        self.assertEqual(function_calls[0]["name"], "do_it")
        self.assertEqual(function_calls[0]["arguments"], "{\"x\":1}")

        outputs = [i for i in items if i.get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["call_id"], "call_1")
        self.assertEqual(outputs[0]["output"], "done")

    def test_responses_tools_flattens_chat_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        out = codex._responses_tools(tools)
        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["name"], "web_search")
        self.assertIn("parameters", out[0])

    def test_resolve_codex_auth_env_mode_uses_env_tokens(self):
        from cc_adapter.config import Settings
        from cc_adapter.codex_oauth import JWT_ACCOUNT_ID_FIELD, JWT_CLAIM_PATH

        token = _jwt({JWT_CLAIM_PATH: {JWT_ACCOUNT_ID_FIELD: "account-123"}})
        settings = Settings(
            model="codex:gpt-5.1-codex",
            codex_auth="env",
            codex_access_token=token,
            codex_refresh_token="refresh",
            codex_expires_at_ms=9_999_999_999_999,
            lmstudio_timeout=1,
        )

        tokens, account_id = codex._resolve_codex_auth(settings)
        self.assertEqual(account_id, "account-123")
        self.assertEqual(tokens.access, token)

    def test_resolve_codex_auth_oauth_mode_uses_token_file(self):
        from cc_adapter.config import Settings
        from cc_adapter.codex_oauth import DEFAULT_TOKEN_FILENAME, JWT_ACCOUNT_ID_FIELD, JWT_CLAIM_PATH

        token = _jwt({JWT_CLAIM_PATH: {JWT_ACCOUNT_ID_FIELD: "account-456"}})
        with tempfile.TemporaryDirectory() as cfgdir:
            with mock.patch.dict(os.environ, {"CC_ADAPTER_CONFIG_DIR": cfgdir}, clear=False):
                Path(cfgdir, DEFAULT_TOKEN_FILENAME).write_text(
                    json.dumps(
                        {
                            "type": "oauth",
                            "access": token,
                            "refresh": "refresh",
                            "expires_at_ms": 9_999_999_999_999,
                        }
                    ),
                    encoding="utf-8",
                )
                settings = Settings(model="codex:gpt-5.1-codex", codex_auth="oauth", lmstudio_timeout=1)

                tokens, account_id = codex._resolve_codex_auth(settings)
                self.assertEqual(account_id, "account-456")
                self.assertEqual(tokens.access, token)

    def test_request_body_applies_preset_defaults(self):
        from cc_adapter.config import Settings

        payload = {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        settings = Settings(lmstudio_timeout=1)
        with mock.patch.object(codex, "get_codex_instructions", return_value="codex-cli-prompt"):
            body = codex._request_body(payload, settings, model_key="codex:gpt-5.2-high")
        self.assertEqual(body["model"], "gpt-5.2")
        self.assertEqual(body["instructions"], "codex-cli-prompt")
        self.assertEqual(body["reasoning"]["effort"], "high")
        self.assertEqual(body["reasoning"]["summary"], "detailed")
        self.assertEqual(body["text"]["verbosity"], "medium")
        self.assertIs(body["store"], False)
        self.assertIn("reasoning.encrypted_content", body.get("include", []))

    def test_request_body_includes_reasoning_summary_and_text_defaults(self):
        from cc_adapter.config import Settings

        payload = {
            "model": "gpt-5.1-codex",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        settings = Settings(lmstudio_timeout=1)
        with mock.patch.object(codex, "get_codex_instructions", return_value="codex-cli-prompt"):
            body = codex._request_body(payload, settings, model_key="codex:gpt-5.1-codex")
        self.assertEqual(body["reasoning"]["effort"], "medium")
        self.assertEqual(body["reasoning"]["summary"], "auto")
        self.assertEqual(body["text"]["verbosity"], "medium")

    def test_request_body_drops_temperature_and_top_p(self):
        from cc_adapter.config import Settings

        payload = {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.2,
            "top_p": 0.9,
        }
        settings = Settings(lmstudio_timeout=1)
        with mock.patch.object(codex, "get_codex_instructions", return_value="codex-cli-prompt"):
            body = codex._request_body(payload, settings, model_key="codex:gpt-5.2-medium")
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_request_body_preset_overrides_payload_reasoning(self):
        from cc_adapter.config import Settings

        payload = {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "high"},
        }
        settings = Settings(lmstudio_timeout=1)
        with mock.patch.object(codex, "get_codex_instructions", return_value="codex-cli-prompt"):
            body = codex._request_body(payload, settings, model_key="codex:gpt-5.2-low")
        self.assertEqual(body["reasoning"]["effort"], "low")
        self.assertEqual(body["reasoning"]["summary"], "auto")
        self.assertEqual(body["text"]["verbosity"], "medium")


if __name__ == "__main__":
    unittest.main()

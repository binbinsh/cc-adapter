import json
import unittest

from cc_adapter import converters


class ConvertersTestCase(unittest.TestCase):
    def test_anthropic_to_openai_maps_thinking_and_tools(self):
        body = {
            "model": "lmstudio:gpt-oss-120b",
            "thinking": {"budget_tokens": 20000},
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "name": "web_search",
                    "description": "search the web",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "web_search"},
        }

        out = converters.anthropic_to_openai(body, "gpt-oss-120b")

        self.assertEqual(out["model"], "gpt-oss-120b")
        self.assertEqual(out["reasoning"], {"effort": "high"})
        self.assertEqual(out["tool_choice"], {"type": "function", "function": {"name": "web_search"}})
        self.assertEqual(len(out["tools"]), 1)
        self.assertEqual(out["tools"][0]["function"]["name"], "web_search")

    def test_anthropic_to_openai_flattens_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "do_it",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "run tool"},
                        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "done"}]},
                    ],
                }
            ]
        }

        out = converters.anthropic_to_openai(body, "gpt-oss-120b")
        tool_messages = [m for m in out["messages"] if m["role"] == "tool"]
        self.assertEqual(out["messages"][0]["role"], "assistant")
        self.assertEqual(out["messages"][1]["role"], "tool")
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "t1")
        self.assertEqual(tool_messages[0]["content"], "done")

    def test_anthropic_to_openai_places_tool_result_before_user_text(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "do_it",
                            "input": {"x": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "note before"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "done"}],
                        },
                        {"type": "text", "text": "note after"},
                    ],
                },
            ]
        }

        out = converters.anthropic_to_openai(body, "gpt-oss-120b")
        self.assertEqual([m["role"] for m in out["messages"][:3]], ["assistant", "tool", "user"])
        self.assertEqual(out["messages"][1]["tool_call_id"], "call-1")
        self.assertEqual(out["messages"][1]["content"], "done")
        user_msg = out["messages"][2]
        self.assertEqual(user_msg["content"], [{"type": "text", "text": "note before\nnote after"}])

    def test_anthropic_to_openai_folds_orphan_tool_results_into_user(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "missing", "content": "done"},
                        {"type": "text", "text": "follow up"},
                    ],
                }
            ]
        }

        out = converters.anthropic_to_openai(body, "gpt-oss-120b")
        self.assertEqual(len(out["messages"]), 1)
        self.assertEqual(out["messages"][0]["role"], "user")
        user_content = out["messages"][0]["content"]
        self.assertEqual(user_content, [{"type": "text", "text": "done\nfollow up"}])

    def test_openai_to_anthropic_includes_thinking_and_tools(self):
        data = {
            "id": "cmpl-1",
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": "hi"}],
                        "reasoning_content": [{"text": "thinking..."}],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "web_search", "arguments": json.dumps({"q": "abc"})},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }

        result = converters.openai_to_anthropic(data, "lmstudio:gpt-oss-120b")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["input_tokens"], 5)
        self.assertEqual(result["usage"]["output_tokens"], 7)

        # thinking block should be first
        self.assertEqual(result["content"][0]["type"], "thinking")
        tool_blocks = [block for block in result["content"] if block["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "web_search")
        self.assertEqual(tool_blocks[0]["input"], {"q": "abc"})

    def test_anthropic_to_openai_drops_stale_tool_calls_without_results(self):
        # Assistant emits a tool_call but no tool result follows; user message interrupts.
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "do_it",
                            "input": {"x": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": "hi again",
                },
            ]
        }
        out = converters.anthropic_to_openai(body, "gpt-oss-120b")
        assistant_messages = [m for m in out["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(assistant_messages), 1)
        # tool_calls should be removed to satisfy OpenAI/Poe constraint
        self.assertNotIn("tool_calls", assistant_messages[0])

    def test_anthropic_to_openai_keeps_tool_calls_when_results_follow(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "do_it",
                            "input": {"x": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "done"},
                    ],
                },
            ]
        }
        out = converters.anthropic_to_openai(body, "gpt-oss-120b")
        assistant_messages = [m for m in out["messages"] if m["role"] == "assistant"]
        tool_messages = [m for m in out["messages"] if m["role"] == "tool"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertIn("tool_calls", assistant_messages[0])
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call-1")


if __name__ == "__main__":
    unittest.main()

import unittest

from cc_adapter.providers import poe


class PoePayloadMergeTestCase(unittest.TestCase):
    def test_merge_adds_thinking_budget_without_overwriting_extra_body(self):
        payload = {"messages": [], "extra_body": {"foo": "bar"}}
        incoming = {"thinking": {"budget_tokens": 256}}

        merged = poe._merge_extra_body(payload, incoming, defaults={"web_search": True})

        self.assertIn("extra_body", merged)
        self.assertEqual(merged["extra_body"]["foo"], "bar")
        self.assertEqual(merged["extra_body"].get("thinking_budget"), 256)
        self.assertTrue(merged["extra_body"].get("web_search"))

    def test_merge_sets_web_search_only_when_requested(self):
        merged = poe._merge_extra_body({}, {}, defaults={})
        self.assertNotIn("extra_body", merged)

        incoming = {
            "tools": [
                {"function": {"name": "web_search"}, "description": "", "input_schema": {"type": "object"}}
            ]
        }
        merged = poe._merge_extra_body({}, incoming, defaults={})
        self.assertTrue(merged["extra_body"].get("web_search"))

    def test_merge_preserves_explicit_web_search(self):
        payload = {"extra_body": {"web_search": False}}
        incoming = {
            "tools": [
                {"function": {"name": "web_search"}, "description": "", "input_schema": {"type": "object"}}
            ]
        }

        merged = poe._merge_extra_body(payload, incoming, defaults={"web_search": True})

        self.assertEqual(merged["extra_body"].get("web_search"), False)

    def test_default_web_search_applied_only_to_claude(self):
        claude_defaults = {"web_search": True}
        merged_claude = poe._merge_extra_body({}, {}, defaults=claude_defaults)
        self.assertTrue(merged_claude["extra_body"].get("web_search"))

        deepseek_defaults = {}
        merged_deepseek = poe._merge_extra_body({}, {}, defaults=deepseek_defaults)
        self.assertNotIn("extra_body", merged_deepseek)


if __name__ == "__main__":
    unittest.main()

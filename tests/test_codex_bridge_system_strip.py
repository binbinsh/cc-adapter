import unittest

from cc_adapter.codex_bridge import split_system_prompt


class CodexBridgeSystemStripTestCase(unittest.TestCase):
    def test_auto_does_not_strip_small_claude_code_helper_prompt(self):
        helper = "\n".join(
            [
                "You are Claude Code, Anthropic's official CLI for Claude.",
                "Extract any file paths that this command reads or modifies.",
                "",
                "Format your response as:",
                "<filepaths>",
                "</filepaths>",
            ]
        )
        kept, extracted = split_system_prompt(helper, "auto")
        self.assertEqual(kept, helper)
        self.assertEqual(extracted, "")

    def test_auto_strips_large_default_claude_code_prompt_but_keeps_instructions_blocks(self):
        large = "\n".join(
            [
                "You are Claude Code, Anthropic's official CLI for Claude.",
                "Use the TodoWrite tool to plan tasks.",
                "Available tools: Bash, Read, Edit, TodoWrite",
                "",
                "x" * 5000,
                "",
                "Instructions from: user",
                "- Always write docs in English",
            ]
        )
        kept, extracted = split_system_prompt(large, "auto")
        self.assertEqual(kept, "")
        self.assertIn("Instructions from: user", extracted)
        self.assertIn("- Always write docs in English", extracted)


if __name__ == "__main__":
    unittest.main()


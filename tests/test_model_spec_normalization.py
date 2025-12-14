import unittest

from cc_adapter.models import normalize_model_spec


class ModelSpecNormalizationTestCase(unittest.TestCase):
    def test_normalize_model_spec_expands_provider_only(self):
        self.assertEqual(normalize_model_spec("codex:"), "codex:gpt-5.1-codex")
        self.assertEqual(normalize_model_spec("codex"), "codex:gpt-5.1-codex")
        self.assertEqual(normalize_model_spec("poe:"), "poe:claude-opus-4.5")
        self.assertEqual(normalize_model_spec("openrouter:"), "openrouter:claude-opus-4.5")
        self.assertEqual(normalize_model_spec("lmstudio:"), "lmstudio:gpt-oss-120b")

    def test_normalize_model_spec_keeps_fully_qualified(self):
        self.assertEqual(normalize_model_spec("codex:gpt-5.2-high"), "codex:gpt-5.2-high")
        self.assertEqual(normalize_model_spec("poe:deepseek-v3.2"), "poe:deepseek-v3.2")


if __name__ == "__main__":
    unittest.main()


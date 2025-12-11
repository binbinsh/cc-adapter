import unittest

from cc_adapter.config import Settings, default_context_window_for


class ContextDefaultsTestCase(unittest.TestCase):
    def test_default_context_window_for_known_models(self):
        self.assertEqual(default_context_window_for("poe:gpt-5.2-pro"), 400_000)
        self.assertEqual(default_context_window_for("poe:claude-sonnet-4.5"), 1_000_000)
        self.assertEqual(default_context_window_for("poe:claude-haiku-4.5"), 200_000)
        self.assertEqual(default_context_window_for("poe:deepseek-v3.2"), 163_840)
        self.assertEqual(default_context_window_for("poe:glm-4.6"), 202_752)
        self.assertEqual(default_context_window_for("openrouter:gpt-5.2-pro"), 400_000)
        self.assertEqual(default_context_window_for("openrouter:claude-opus-4.5"), 200_000)
        self.assertEqual(default_context_window_for("openrouter:gpt-5.2"), 400_000)
        self.assertEqual(default_context_window_for("lmstudio:gpt-oss-120b"), 131_072)

    def test_default_context_window_for_unknown_models_is_zero(self):
        self.assertEqual(default_context_window_for("somevendor:new-model"), 0)

    def test_resolved_context_window_prefers_override(self):
        settings = Settings(context_window=1234, model="poe:claude-sonnet-4.5")
        self.assertEqual(settings.resolved_context_window("poe:claude-opus-4.5"), 1234)

    def test_resolved_context_window_falls_back_to_model_default(self):
        settings = Settings(context_window=0, model="poe:claude-opus-4.5")
        self.assertEqual(settings.resolved_context_window(), 200_000)


if __name__ == "__main__":
    unittest.main()

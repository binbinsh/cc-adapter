import unittest

from cc_adapter import streaming
from cc_adapter.config import Settings
from cc_adapter.models import resolve_provider_model
from cc_adapter import server


class ServerHelpersTestCase(unittest.TestCase):
    def test_resolve_provider_model_prefixed(self):
        settings = Settings(model="poe:claude-opus-4.5")
        provider, name = resolve_provider_model("poe:claude-sonnet-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-sonnet-4.5")

    def test_resolve_provider_model_uses_config_when_unprefixed(self):
        settings = Settings(model="poe:claude-opus-4.5")
        provider, name = resolve_provider_model("claude-opus-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-opus-4.5")

    def test_available_models_reflect_keys(self):
        settings = Settings(
            lmstudio_model="local-model",
            poe_api_key="k1",
            openrouter_key="k2",
        )
        models = server._available_models(settings)
        self.assertIn("lmstudio:local-model", models)
        self.assertIn("poe:claude-opus-4.5", models)
        self.assertIn("openrouter:claude-sonnet-4.5", models)

    def test_is_allowed_model_supports_anthropic_prefix(self):
        settings = Settings(openrouter_key="k2")
        allowed = server._is_allowed_model("openrouter", "anthropic/claude-sonnet-4.5", settings)
        self.assertTrue(allowed)

    def test_estimate_prompt_tokens_counts_text(self):
        incoming = {
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ],
        }
        tokens = streaming.estimate_prompt_tokens(incoming)
        self.assertGreaterEqual(tokens, 1)


if __name__ == "__main__":
    unittest.main()

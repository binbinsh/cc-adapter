import os
import unittest
from unittest import mock

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

    def test_resolved_proxies_prefer_specific_over_all(self):
        settings = Settings(
            http_proxy="http://http-proxy:8080",
            https_proxy="http://https-proxy:8080",
            all_proxy="socks5://fallback:1080",
        )
        proxies = settings.resolved_proxies()
        self.assertEqual(proxies["http"], "http://http-proxy:8080")
        self.assertEqual(proxies["https"], "http://https-proxy:8080")

    def test_resolved_proxies_fall_back_to_all(self):
        settings = Settings(
            http_proxy="",
            https_proxy="",
            all_proxy="socks5://fallback:1080",
        )
        proxies = settings.resolved_proxies()
        self.assertEqual(proxies["http"], "socks5://fallback:1080")
        self.assertEqual(proxies["https"], "socks5://fallback:1080")

    def test_apply_no_proxy_env_sets_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings(no_proxy="127.0.0.1,localhost")
            settings.apply_no_proxy_env()
            self.assertEqual(os.environ["NO_PROXY"], "127.0.0.1,localhost")
            self.assertEqual(os.environ["no_proxy"], "127.0.0.1,localhost")


if __name__ == "__main__":
    unittest.main()

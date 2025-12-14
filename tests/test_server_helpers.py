import json
import os
import socket
import tempfile
import unittest
from unittest import mock

os.environ["CC_ADAPTER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="cc-adapter-tests-")

from cc_adapter import streaming
from cc_adapter.config import Settings
from cc_adapter.models import available_models, resolve_provider_model
from cc_adapter import server


class ServerHelpersTestCase(unittest.TestCase):
    def test_resolve_provider_model_prefixed(self):
        settings = Settings(model="poe:claude-opus-4.5")
        provider, name = resolve_provider_model("poe:claude-sonnet-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-opus-4.5")

    def test_resolve_provider_model_uses_config_when_unprefixed(self):
        settings = Settings(model="poe:claude-opus-4.5")
        provider, name = resolve_provider_model("claude-opus-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-opus-4.5")

    def test_resolve_provider_model_auto_prefers_poe_haiku(self):
        settings = Settings(model="poe:gpt-5.2-pro", poe_api_key="k1")
        provider, name = resolve_provider_model("claude-haiku-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-haiku-4.5")

    def test_resolve_provider_model_auto_normalizes_haiku_alias(self):
        settings = Settings(model="poe:gpt-5.2-pro", poe_api_key="k1")
        provider, name = resolve_provider_model("claude-haiku-4-5-20251001", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-haiku-4.5")

    def test_resolve_provider_model_routes_non_haiku_to_selected_model(self):
        settings = Settings(model="poe:deepseek-v3.2")
        provider, name = resolve_provider_model("openrouter:claude-sonnet-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "deepseek-v3.2")

    def test_resolve_provider_model_honors_requested_provider_for_haiku(self):
        settings = Settings(model="poe:deepseek-v3.2", openrouter_key="k2")
        provider, name = resolve_provider_model("openrouter:claude-haiku-4.5", settings)
        self.assertEqual(provider, "openrouter")
        self.assertEqual(name, "anthropic/claude-haiku-4.5")

    def test_resolve_provider_model_prefers_selected_provider_openrouter(self):
        settings = Settings(model="openrouter:claude-opus-4.5", poe_api_key="k1", openrouter_key="k2")
        provider, name = resolve_provider_model("claude-haiku-4.5", settings)
        self.assertEqual(provider, "openrouter")
        self.assertEqual(name, "anthropic/claude-haiku-4.5")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_resolve_provider_model_requires_provider_for_haiku_when_unprefixed(self):
        settings = Settings(model="poe:claude-opus-4.5", poe_api_key="", openrouter_key="k2")
        provider, name = resolve_provider_model("claude-haiku-4.5", settings)
        self.assertEqual(provider, "poe")
        self.assertEqual(name, "claude-haiku-4.5")

    def test_resolve_provider_model_maps_haiku_to_lmstudio_default(self):
        settings = Settings(model="lmstudio:gpt-oss-120b", lmstudio_model="local-model")
        provider, name = resolve_provider_model("claude-haiku-4.5", settings)
        self.assertEqual(provider, "lmstudio")
        self.assertEqual(name, "gpt-oss-120b")

    def test_resolve_provider_model_maps_anthropic_haiku_to_lmstudio_default(self):
        settings = Settings(model="lmstudio:gpt-oss-120b", lmstudio_model="offline-fallback")
        provider, name = resolve_provider_model("anthropic/claude-haiku-4.5", settings)
        self.assertEqual(provider, "lmstudio")
        self.assertEqual(name, "gpt-oss-120b")

    def test_effective_codex_settings_maps_haiku_to_codex_mini(self):
        settings = Settings(model="codex:gpt-5.2-xhigh")
        effective, model = server._effective_codex_settings(settings, "claude-haiku-4.5", "gpt-5.2")
        self.assertEqual(model, server.CODEX_HAIKU_FALLBACK_MODEL)
        self.assertEqual(effective.model, f"codex:{server.CODEX_HAIKU_FALLBACK_MODEL}")

    def test_effective_codex_settings_maps_haiku_to_codex_mini_codex_max(self):
        settings = Settings(model="codex:gpt-5.1-codex-max-xhigh")
        effective, model = server._effective_codex_settings(
            settings, "claude-haiku-4-5-20251001", "gpt-5.1-codex-max"
        )
        self.assertEqual(model, server.CODEX_HAIKU_FALLBACK_MODEL)
        self.assertEqual(effective.model, f"codex:{server.CODEX_HAIKU_FALLBACK_MODEL}")

    def test_effective_codex_settings_noop_for_non_haiku(self):
        settings = Settings(model="codex:gpt-5.2-xhigh")
        effective, model = server._effective_codex_settings(settings, "claude-opus-4.5", "gpt-5.2")
        self.assertEqual(model, "gpt-5.2")
        self.assertEqual(effective.model, "codex:gpt-5.2-xhigh")

    def test_effective_codex_settings_noop_when_not_codex(self):
        settings = Settings(model="poe:claude-opus-4.5")
        effective, model = server._effective_codex_settings(settings, "claude-haiku-4.5", "gpt-5.2")
        self.assertEqual(model, "gpt-5.2")
        self.assertEqual(effective.model, "poe:claude-opus-4.5")

    def test_available_models_reflect_keys(self):
        settings = Settings(
            lmstudio_model="local-model",
            poe_api_key="k1",
            openrouter_key="k2",
        )
        models = available_models(settings)
        self.assertIn("lmstudio:local-model", models)
        self.assertIn("poe:gpt-5.2-pro", models)
        self.assertIn("poe:claude-haiku-4.5", models)
        self.assertIn("poe:claude-opus-4.5", models)
        self.assertIn("poe:deepseek-v3.2", models)
        self.assertIn("poe:glm-4.6", models)
        self.assertIn("openrouter:gpt-5.2-pro", models)
        self.assertIn("openrouter:claude-haiku-4.5", models)
        self.assertIn("openrouter:claude-sonnet-4.5", models)
        self.assertIn("openrouter:gpt-5.2", models)
        self.assertIn("openrouter:glm-4.6", models)
        self.assertNotIn("poe:gpt-5.1-codex", models)
        self.assertNotIn("poe:gpt-5.1-codex-max", models)
        self.assertNotIn("openrouter:gpt-5.1-codex", models)

    def test_available_models_includes_explicit_default(self):
        settings = Settings(model="poe:custom-model", poe_api_key="k1")
        models = available_models(settings)
        self.assertIn("poe:custom-model", models)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_settings_default_model_is_opus(self):
        settings = Settings()
        self.assertEqual(settings.model, "poe:claude-opus-4.5")

    def test_available_models_order_custom_then_claude_priority(self):
        settings = Settings(model="poe:my-bot", poe_api_key="k1", openrouter_key="k2")
        models = available_models(settings)
        self.assertEqual(models[0], "poe:my-bot")

        codex_positions = [
            models.index(m)
            for m in models
            if m.endswith("gpt-5.2") or "deepseek-v3.2" in m
        ]
        glm_positions = [models.index(m) for m in models if "glm-4.6" in m]
        gpt5pro_positions = [models.index(m) for m in models if "gpt-5.2-pro" in m]
        opus_positions = [models.index(m) for m in models if "claude-opus" in m]
        sonnet_positions = [models.index(m) for m in models if "claude-sonnet" in m]
        haiku_positions = [models.index(m) for m in models if "claude-haiku" in m]

        self.assertTrue(
            codex_positions
            and glm_positions
            and gpt5pro_positions
            and opus_positions
            and sonnet_positions
            and haiku_positions
        )
        # Claudes (opus/sonnet/haiku) should lead the frontier stack, followed by codex/deepseek,
        # then GLM, and finally gpt-5.2-pro at the tail.
        self.assertLess(max(opus_positions), min(sonnet_positions))
        self.assertLess(max(sonnet_positions), min(haiku_positions))
        self.assertLess(max(haiku_positions), min(codex_positions))
        self.assertLess(max(codex_positions), min(glm_positions))
        self.assertLess(max(glm_positions), min(gpt5pro_positions))

    def test_available_models_respects_missing_provider_keys(self):
        settings = Settings(model="", poe_api_key="", openrouter_key="", lmstudio_model="local")
        models = available_models(settings)
        self.assertEqual(models, ["lmstudio:local"])

    def test_canonicalize_glm_alias_for_openrouter(self):
        settings = Settings(model="openrouter:glm-4.6", openrouter_key="k2")
        provider, name = resolve_provider_model("openrouter:glm-4.6", settings)
        self.assertEqual(provider, "openrouter")
        self.assertEqual(name, "z-ai/glm-4.6")

    def test_port_available_detects_in_use(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        try:
            self.assertFalse(server.port_available(host, port))
        finally:
            sock.close()

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

    def test_json_response_sends_body(self):
        class DummyHandler:
            def __init__(self):
                self.status = None
                self.headers = []
                self.body = b""
                self.close_connection = False
                self.wfile = self

            def send_response(self, status):
                self.status = status

            def send_header(self, key, value):
                self.headers.append((key, value))

            def end_headers(self):
                return

            def write(self, data):
                self.body += data

        handler = DummyHandler()
        server._json_response(handler, 200, {"ok": True})
        self.assertEqual(handler.status, 200)
        self.assertIn(("Content-Type", "application/json"), handler.headers)
        self.assertEqual(handler.body, json.dumps({"ok": True}).encode("utf-8"))
        # Content-Length header should match the serialized body length.
        self.assertIn(("Content-Length", str(len(handler.body))), handler.headers)
        self.assertFalse(handler.close_connection)

    def test_json_response_handles_broken_pipe(self):
        class DummyHandler:
            def __init__(self):
                self.close_connection = False
                self.wfile = self

            def send_response(self, status):
                return

            def send_header(self, key, value):
                return

            def end_headers(self):
                raise BrokenPipeError()

            def write(self, data):
                return

        handler = DummyHandler()
        server._json_response(handler, 200, {"ok": True})
        self.assertTrue(handler.close_connection)


if __name__ == "__main__":
    unittest.main()

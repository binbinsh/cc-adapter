import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cc_adapter.config import Settings
from cc_adapter.codex_oauth import DEFAULT_TOKEN_FILENAME
from cc_adapter.models import available_models


class CodexModelListingTestCase(unittest.TestCase):
    def test_available_models_includes_codex_when_oauth_file_present(self):
        with tempfile.TemporaryDirectory() as cfgdir:
            with mock.patch.dict(os.environ, {"CC_ADAPTER_CONFIG_DIR": cfgdir}, clear=False):
                Path(cfgdir, DEFAULT_TOKEN_FILENAME).write_text(
                    json.dumps(
                        {
                            "type": "oauth",
                            "access": "a",
                            "refresh": "r",
                            "expires_at_ms": 9_999_999_999_999,
                        }
                    ),
                    encoding="utf-8",
                )
                settings = Settings(model="lmstudio:local", lmstudio_model="local", codex_auth="oauth")
                models = available_models(settings)
                self.assertIn("codex:gpt-5.1-codex", models)
                self.assertIn("codex:gpt-5.2-medium", models)
                self.assertIn("codex:gpt-5.1-codex-max-xhigh", models)

    def test_available_models_hides_codex_when_env_mode_and_no_env_tokens(self):
        with tempfile.TemporaryDirectory() as cfgdir:
            with mock.patch.dict(os.environ, {"CC_ADAPTER_CONFIG_DIR": cfgdir}, clear=False):
                Path(cfgdir, DEFAULT_TOKEN_FILENAME).write_text(
                    json.dumps(
                        {
                            "type": "oauth",
                            "access": "a",
                            "refresh": "r",
                            "expires_at_ms": 9_999_999_999_999,
                        }
                    ),
                    encoding="utf-8",
                )
                settings = Settings(model="lmstudio:local", lmstudio_model="local", codex_auth="env")
                models = available_models(settings)
                self.assertNotIn("codex:gpt-5.1-codex", models)


if __name__ == "__main__":
    unittest.main()

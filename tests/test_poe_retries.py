import unittest
from unittest.mock import patch

import requests

from cc_adapter.config import Settings
from cc_adapter.providers import poe


class PoeRetryConfigTestCase(unittest.TestCase):
    def test_retry_session_configured_for_post_and_statuses(self):
        settings = Settings(poe_api_key="token", poe_max_retries=3, poe_retry_backoff=0.25)
        session = poe._build_retry_session(settings)
        adapter = session.get_adapter(settings.poe_base_url)
        retries = adapter.max_retries

        self.assertEqual(retries.total, 3)
        self.assertEqual(retries.backoff_factor, 0.25)
        self.assertIn("POST", retries.allowed_methods)
        for status in poe.RETRYABLE_STATUS_CODES:
            self.assertIn(status, retries.status_forcelist)
        session.close()

    def test_post_with_retries_wraps_http_error_and_closes(self):
        settings = Settings(poe_api_key="token")

        class DummyResponse:
            def __init__(self):
                self.closed = False
                self.text = "<html>Internal server error</html>"
                self.status_code = 502

            def raise_for_status(self):
                raise requests.HTTPError("502 Server Error", response=self)

            def close(self):
                self.closed = True

        class DummySession:
            def __init__(self, resp):
                self.resp = resp
                self.closed = False
                self.proxies = {}

            def mount(self, *_args, **_kwargs):
                return None

            def post(self, *_args, **_kwargs):
                return self.resp

            def close(self):
                self.closed = True

        dummy_resp = DummyResponse()
        dummy_session = DummySession(dummy_resp)

        with patch("cc_adapter.providers.poe._build_retry_session", return_value=dummy_session):
            with self.assertRaises(requests.HTTPError) as ctx:
                poe._post_with_retries({"messages": []}, settings, stream=False)

        self.assertTrue(dummy_resp.closed)
        self.assertTrue(dummy_session.closed)
        self.assertIn("body_snippet=<html>Internal server error</html>", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

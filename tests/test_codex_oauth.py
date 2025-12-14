import base64
import json
import unittest

from cc_adapter import codex_oauth


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def b64url(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode("utf-8")).decode("utf-8").rstrip("=")

    return f"{b64url(header)}.{b64url(payload)}.sig"


class CodexOAuthTestCase(unittest.TestCase):
    def test_parse_authorization_input_url(self):
        code, state = codex_oauth.parse_authorization_input(
            "http://localhost:1455/auth/callback?code=abc123&state=xyz789"
        )
        self.assertEqual(code, "abc123")
        self.assertEqual(state, "xyz789")

    def test_parse_authorization_input_code_hash_state(self):
        code, state = codex_oauth.parse_authorization_input("abc123#xyz789")
        self.assertEqual(code, "abc123")
        self.assertEqual(state, "xyz789")

    def test_parse_authorization_input_query_string(self):
        code, state = codex_oauth.parse_authorization_input("code=abc123&state=xyz789")
        self.assertEqual(code, "abc123")
        self.assertEqual(state, "xyz789")

    def test_parse_authorization_input_code_only(self):
        code, state = codex_oauth.parse_authorization_input("abc123")
        self.assertEqual(code, "abc123")
        self.assertIsNone(state)

    def test_parse_authorization_input_empty(self):
        code, state = codex_oauth.parse_authorization_input("")
        self.assertIsNone(code)
        self.assertIsNone(state)

    def test_extract_chatgpt_account_id(self):
        token = _jwt(
            {
                codex_oauth.JWT_CLAIM_PATH: {
                    codex_oauth.JWT_ACCOUNT_ID_FIELD: "account-123",
                }
            }
        )
        self.assertEqual(codex_oauth.extract_chatgpt_account_id(token), "account-123")


if __name__ == "__main__":
    unittest.main()


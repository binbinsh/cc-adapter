import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import platformdirs
import requests

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"

JWT_CLAIM_PATH = "https://api.openai.com/auth"
JWT_ACCOUNT_ID_FIELD = "chatgpt_account_id"

DEFAULT_TOKEN_FILENAME = "openai_codex_oauth.json"
OAUTH_CALLBACK_PORT = 1455

SUCCESS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CC Adapter - OAuth Success</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>OAuth complete</h2>
  <p>You can close this tab and return to <code>cc-adapter</code>.</p>
</body>
</html>
"""


@dataclass(frozen=True)
class CodexOAuthTokens:
    access: str
    refresh: str
    expires_at_ms: int

    def expired(self, skew_seconds: int = 60) -> bool:
        return self.expires_at_ms <= int(time.time() * 1000) + (skew_seconds * 1000)

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": "oauth",
                "access": self.access,
                "refresh": self.refresh,
                "expires_at_ms": self.expires_at_ms,
            },
            indent=2,
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CodexOAuthTokens":
        access = str(data.get("access") or "")
        refresh = str(data.get("refresh") or "")
        expires_at_ms = int(data.get("expires_at_ms") or 0)
        if not access or not refresh or expires_at_ms <= 0:
            raise ValueError("Invalid token payload")
        return CodexOAuthTokens(access=access, refresh=refresh, expires_at_ms=expires_at_ms)


def default_token_path() -> Path:
    override = os.getenv("CC_ADAPTER_CONFIG_DIR", "").strip()
    if override:
        return Path(override) / DEFAULT_TOKEN_FILENAME
    return Path(platformdirs.user_config_dir("cc-adapter")) / DEFAULT_TOKEN_FILENAME


def load_tokens(path: Optional[Path] = None) -> Optional[CodexOAuthTokens]:
    token_path = path or default_token_path()
    if not token_path.exists():
        return None
    try:
        raw = json.loads(token_path.read_text(encoding="utf-8"))
        return CodexOAuthTokens.from_dict(raw)
    except Exception:
        return None


def save_tokens(tokens: CodexOAuthTokens, path: Optional[Path] = None) -> Path:
    token_path = path or default_token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = token_path.with_suffix(token_path.suffix + ".tmp")
    tmp_path.write_text(tokens.to_json(), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    tmp_path.replace(token_path)
    return token_path


def delete_tokens(path: Optional[Path] = None) -> None:
    token_path = path or default_token_path()
    try:
        token_path.unlink()
    except FileNotFoundError:
        return


def create_state() -> str:
    return secrets.token_hex(16)


def _b64url_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def generate_pkce_pair() -> Tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    if len(verifier) > 128:
        verifier = verifier[:128]
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = _b64url_no_pad(digest)
    return verifier, challenge


def build_authorization_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def parse_authorization_input(value: str) -> Tuple[Optional[str], Optional[str]]:
    raw = (value or "").strip()
    if not raw:
        return None, None

    try:
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query or "")
        code = (qs.get("code") or [None])[0]
        state = (qs.get("state") or [None])[0]
        if code or state:
            return (code, state)
    except Exception:
        pass

    if "#" in raw:
        code, state = raw.split("#", 1)
        return (code or None, state or None)

    if "code=" in raw:
        try:
            qs = parse_qs(raw)
            return (qs.get("code", [None])[0], qs.get("state", [None])[0])
        except Exception:
            pass

    return raw, None


def exchange_authorization_code(
    code: str,
    code_verifier: str,
    redirect_uri: str = REDIRECT_URI,
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> CodexOAuthTokens:
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
        },
        timeout=timeout,
        proxies=proxies,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    try:
        expires_in_int = int(expires_in)
    except Exception:
        expires_in_int = 0
    if not access or not refresh or expires_in_int <= 0:
        raise RuntimeError("Token exchange response missing required fields")
    return CodexOAuthTokens(
        access=str(access),
        refresh=str(refresh),
        expires_at_ms=int(time.time() * 1000) + (expires_in_int * 1000),
    )


def refresh_access_token(
    refresh_token: str,
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> CodexOAuthTokens:
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
        timeout=timeout,
        proxies=proxies,
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    try:
        expires_in_int = int(expires_in)
    except Exception:
        expires_in_int = 0
    if not access or not refresh or expires_in_int <= 0:
        raise RuntimeError("Token refresh response missing required fields")
    return CodexOAuthTokens(
        access=str(access),
        refresh=str(refresh),
        expires_at_ms=int(time.time() * 1000) + (expires_in_int * 1000),
    )


def decode_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    parts = (token or "").split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_b64 + padding)
        return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None


def extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    payload = decode_jwt_payload(access_token) or {}
    claim = payload.get(JWT_CLAIM_PATH) or {}
    if isinstance(claim, dict):
        account_id = claim.get(JWT_ACCOUNT_ID_FIELD)
        if account_id:
            return str(account_id)
    return None


class _CallbackServer(ThreadingHTTPServer):
    expected_state: str
    code_event: threading.Event
    code_value: Optional[str]


def start_local_callback_server(state: str, host: str = "127.0.0.1") -> _CallbackServer:
    code_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not Found")
                return

            qs = parse_qs(parsed.query or "")
            received_state = (qs.get("state") or [None])[0]
            code = (qs.get("code") or [None])[0]
            if received_state != self.server.expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                return
            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code")
                return
            self.server.code_value = str(code)
            self.server.code_event.set()
            body = SUCCESS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _CallbackServer((host, OAUTH_CALLBACK_PORT), Handler)
    server.expected_state = state
    server.code_event = code_event
    server.code_value = None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def wait_for_callback_code(server: _CallbackServer, timeout_seconds: int = 300) -> Optional[str]:
    if server.code_event.wait(timeout=timeout_seconds):
        return server.code_value
    return None

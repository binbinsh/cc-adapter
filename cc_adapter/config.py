import os
from dataclasses import dataclass
from typing import Dict, Optional

from .model_registry import default_context_window_for


@dataclass
class Settings:
    host: str = os.getenv("ADAPTER_HOST", "127.0.0.1")
    port: int = int(os.getenv("ADAPTER_PORT", "8005"))
    model: str = os.getenv("CC_ADAPTER_MODEL", "poe:claude-opus-4.5")
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "0"))
    lmstudio_base: str = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1/chat/completions")
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "gpt-oss-120b")
    lmstudio_timeout: int = int(os.getenv("LMSTUDIO_TIMEOUT", "3600"))

    poe_base_url: str = os.getenv("POE_BASE_URL", "https://api.poe.com/v1/chat/completions")
    poe_api_key: str = os.getenv("POE_API_KEY", "")
    poe_max_retries: int = int(os.getenv("POE_MAX_RETRIES", "2"))
    poe_retry_backoff: float = float(os.getenv("POE_RETRY_BACKOFF", "0.5"))

    openrouter_base: str = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1/chat/completions")
    openrouter_key: str = os.getenv("OPENROUTER_API_KEY", "")
    codex_base_url: str = os.getenv("CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex/responses")
    codex_auth: str = os.getenv("CODEX_AUTH", "auto")
    codex_access_token: str = os.getenv("OPENAI_CODEX_ACCESS_TOKEN", "")
    codex_refresh_token: str = os.getenv("OPENAI_CODEX_REFRESH_TOKEN", "")
    codex_expires_at_ms: int = int(os.getenv("OPENAI_CODEX_EXPIRES_AT_MS", "0"))
    http_proxy: str = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""
    https_proxy: str = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or ""
    all_proxy: str = os.getenv("ALL_PROXY") or os.getenv("all_proxy") or ""
    no_proxy: str = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""

    def resolved_proxies(self) -> Optional[Dict[str, str]]:
        """Build a requests-compatible proxies mapping from settings."""
        proxies: Dict[str, str] = {}
        if self.all_proxy:
            proxies.setdefault("http", self.all_proxy)
            proxies.setdefault("https", self.all_proxy)
        if self.http_proxy:
            proxies["http"] = self.http_proxy
        if self.https_proxy:
            proxies["https"] = self.https_proxy
        return proxies or None

    def apply_no_proxy_env(self) -> None:
        """Propagate an explicit no_proxy setting to the environment."""
        if self.no_proxy:
            os.environ["NO_PROXY"] = self.no_proxy
            os.environ["no_proxy"] = self.no_proxy

    def resolved_context_window(self, model: Optional[str] = None) -> int:
        """Return the effective context window, falling back to model defaults."""
        if self.context_window and self.context_window > 0:
            return self.context_window
        target = model or self.model
        return default_context_window_for(target)


def load_settings() -> Settings:
    return Settings()


def apply_overrides(settings: Settings, overrides: dict) -> Settings:
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings

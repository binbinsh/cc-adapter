import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Settings:
    host: str = os.getenv("ADAPTER_HOST", "0.0.0.0")
    port: int = int(os.getenv("ADAPTER_PORT", "8000"))
    model: str = os.getenv("CC_ADAPTER_MODEL", "")
    lmstudio_base: str = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1/chat/completions")
    lmstudio_model: str = os.getenv("LMSTUDIO_MODEL", "gpt-oss-120b")
    lmstudio_timeout: float = float(os.getenv("LMSTUDIO_TIMEOUT", "120"))

    poe_base_url: str = os.getenv("POE_BASE_URL", "https://api.poe.com/v1/chat/completions")
    poe_api_key: str = os.getenv("POE_API_KEY", "")

    openrouter_base: str = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1/chat/completions")
    openrouter_key: str = os.getenv("OPENROUTER_API_KEY", "")
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


def load_settings() -> Settings:
    return Settings()


def apply_overrides(settings: Settings, overrides: dict) -> Settings:
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings

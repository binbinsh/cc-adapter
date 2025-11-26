import os
from dataclasses import dataclass


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


def load_settings() -> Settings:
    return Settings()


def apply_overrides(settings: Settings, overrides: dict) -> Settings:
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings

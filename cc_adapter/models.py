from typing import Optional, Tuple
from .config import Settings


def resolve_provider_model(model: Optional[str], settings: Settings) -> Tuple[str, str]:
    """Return (provider, model). Model must be explicitly specified with provider prefix."""
    target_model = model or settings.model
    if not target_model:
        raise ValueError("Model is required and must include provider prefix (e.g., poe:claude-opus-4.5)")

    # If client passed an unprefixed model but server is configured with a prefixed model, honor server config
    if ":" not in target_model and settings.model:
        target_model = settings.model

    if ":" in target_model:
        provider, name = target_model.split(":", 1)
        provider = provider.lower()
        if provider == "poe":
            return "poe", name
        if provider == "lmstudio":
            return "lmstudio", name
        if provider == "openrouter":
            return "openrouter", name
        raise ValueError(f"Unsupported provider prefix: {provider}")

    raise ValueError("Model must include provider prefix (e.g., poe:claude-opus-4.5)")

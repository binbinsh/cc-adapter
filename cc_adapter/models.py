from typing import Optional, Tuple

from .config import Settings
from .model_registry import canonicalize_model, find_model, provider_models


def _display_slug(provider: str, name: str) -> str:
    match = find_model(provider, name)
    return match.slug if match else name


def available_models(settings: Settings) -> list[str]:
    """
    Build a user-facing list of provider-prefixed models based on available credentials.
    Ordering prefers higher-capability tiers first, while keeping any configured default at the top.
    """
    models: list[str] = []
    seen: set[str] = set()

    def add(provider: str, slug: str):
        key = f"{provider}:{slug}"
        if key not in seen:
            models.append(key)
            seen.add(key)

    # Always bubble the configured default to the top if it includes a prefix.
    if settings.model and ":" in settings.model:
        provider, name = settings.model.split(":", 1)
        provider = provider.lower()
        add(provider, _display_slug(provider, name))

    # Always show the LM Studio default so local-only users see an option.
    add("lmstudio", _display_slug("lmstudio", settings.lmstudio_model))

    # Include known provider offerings gated by available credentials.
    eligible: list[Tuple[int, str, str]] = []
    if settings.poe_api_key:
        for info in provider_models("poe"):
            eligible.append((info.priority, info.provider, info.slug))
    if settings.openrouter_key:
        for info in provider_models("openrouter"):
            eligible.append((info.priority, info.provider, info.slug))

    for _, provider, slug in sorted(eligible, key=lambda item: (item[0], item[1], item[2])):
        add(provider, slug)

    return models


def resolve_provider_model(model: Optional[str], settings: Settings) -> Tuple[str, str]:
    """Return (provider, upstream_model) while favoring the selected model by default."""
    default_model = settings.model
    if not default_model:
        raise ValueError("Model is required and must include provider prefix (e.g., poe:claude-opus-4.5)")
    if ":" not in default_model:
        raise ValueError("Model must include provider prefix (e.g., poe:claude-opus-4.5)")

    default_provider, default_name = default_model.split(":", 1)
    default_provider = default_provider.lower()
    if default_provider not in {"poe", "lmstudio", "openrouter"}:
        raise ValueError(f"Unsupported provider prefix: {default_provider}")

    requested = (model or "").strip()
    if requested:
        req_provider = default_provider
        req_name = requested
        if ":" in requested:
            req_provider, req_name = requested.split(":", 1)
            req_provider = req_provider.lower()

        if req_provider in {"poe", "openrouter"} and "claude-haiku" in req_name.lower():
            return req_provider, canonicalize_model(req_provider, req_name)

    return default_provider, canonicalize_model(default_provider, default_name)

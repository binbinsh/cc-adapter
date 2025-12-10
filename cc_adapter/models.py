from typing import Optional, Tuple

from .config import Settings
from .model_registry import canonicalize_model, find_model, provider_models


def _default_provider(settings: Settings) -> str:
    if settings.model and ":" in settings.model:
        return settings.model.split(":", 1)[0].lower()
    return ""


def _display_slug(provider: str, name: str) -> str:
    match = find_model(provider, name)
    return match.slug if match else name


def _select_haiku_provider(settings: Settings) -> str:
    preferred = _default_provider(settings)
    if preferred == "poe" and settings.poe_api_key:
        return "poe"
    if preferred == "openrouter" and settings.openrouter_key:
        return "openrouter"
    if preferred == "lmstudio":
        return "lmstudio"
    raise ValueError("No valid provider configured for Claude Haiku model.")


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
    """Return (provider, upstream_model). Model may be prefixed or rely on the configured default provider."""
    target_model = model or settings.model
    if not target_model:
        raise ValueError("Model is required and must include provider prefix (e.g., poe:claude-opus-4.5)")

    if ":" in target_model:
        provider, name = target_model.split(":", 1)
        provider = provider.lower()
        if provider not in {"poe", "lmstudio", "openrouter"}:
            raise ValueError(f"Unsupported provider prefix: {provider}")

        if provider == "lmstudio" and "claude-haiku" in name.lower():
            return "lmstudio", settings.lmstudio_model
        return provider, canonicalize_model(provider, name)

    lowered = target_model.lower()
    if "claude-haiku" in lowered:
        provider = _select_haiku_provider(settings)
        if provider == "lmstudio":
            return "lmstudio", settings.lmstudio_model
        return provider, canonicalize_model(provider, target_model)

    provider = _default_provider(settings)
    if provider in {"poe", "lmstudio", "openrouter"}:
        if provider == "lmstudio":
            return provider, target_model
        return provider, canonicalize_model(provider, target_model)

    raise ValueError("Model must include provider prefix (e.g., poe:claude-opus-4.5)")

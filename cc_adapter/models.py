from typing import Optional, Tuple
from .config import Settings


def resolve_provider_model(model: Optional[str], settings: Settings) -> Tuple[str, str]:
    """Return (provider, model). Model must be explicitly specified with provider prefix."""
    target_model = model or settings.model
    if not target_model:
        raise ValueError("Model is required and must include provider prefix (e.g., poe:claude-opus-4.5)")

    if ":" in target_model:
        provider, name = target_model.split(":", 1)
        provider = provider.lower()
        if provider == "poe":
            return "poe", canonicalize_model("poe", name)
        if provider == "lmstudio":
            return "lmstudio", name
        if provider == "openrouter":
            return "openrouter", canonicalize_model("openrouter", name)
        raise ValueError(f"Unsupported provider prefix: {provider}")

    lowered = target_model.lower()
    if lowered.startswith("claude-haiku"):
        selected_provider = ""
        if settings.model and ":" in settings.model:
            selected_provider = settings.model.split(":", 1)[0].lower()

        if selected_provider == "poe" and settings.poe_api_key:
            return "poe", canonicalize_model("poe", target_model)
        elif selected_provider == "openrouter" and settings.openrouter_key:
            return "openrouter", canonicalize_model("openrouter", target_model)
        else:
            raise ValueError("No valid provider configured for Claude Haiku model.")

    # Fall back to server default provider if configured with a prefix
    if settings.model and ":" in settings.model:
        provider, name = settings.model.split(":", 1)
        provider = provider.lower()
        if provider in {"poe", "lmstudio", "openrouter"}:
            normalized_name = (
                canonicalize_model(provider, name) if provider in {"poe", "openrouter"} else name
            )
            return provider, normalized_name

    raise ValueError("Model must include provider prefix (e.g., poe:claude-opus-4.5)")


def canonicalize_model(provider: str, model: str) -> str:
    """
    Normalize known model aliases to canonical IDs for the selected provider.

    Currently collapses all Claude Haiku 4.5 variants to the canonical 4.5 name.
    """
    lowered = model.lower()
    if "claude-haiku" in lowered:
        base = "claude-haiku-4.5"
        if lowered.startswith("anthropic/"):
            return f"anthropic/{base}"
        return base
    return model

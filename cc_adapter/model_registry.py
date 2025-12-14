import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a provider model."""

    provider: str
    slug: str
    upstream: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    context_window: int = 0
    priority: int = 100
    expose: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> str:
        """Return the provider-facing model identifier."""
        return self.upstream or self.slug


# Ordering favors higher-cost/capability models first so dropdowns and /v1/models
# responses present sensible defaults.
MODEL_ENTRIES: Tuple[ModelInfo, ...] = (
    # LM Studio
    ModelInfo(
        provider="lmstudio",
        slug="gpt-oss-120b",
        context_window=131_072,
        priority=60,
    ),
    # OpenAI Codex (ChatGPT OAuth)
    ModelInfo(
        provider="codex",
        slug="gpt-5.2",
        context_window=400_000,
        priority=5,
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-max",
        aliases=("gpt-5-codex-max",),
        context_window=400_000,
        priority=10,
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex",
        aliases=("gpt-5-codex",),
        context_window=400_000,
        priority=20,
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-mini",
        aliases=("gpt-5-codex-mini", "codex-mini-latest"),
        context_window=400_000,
        priority=30,
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1",
        aliases=("gpt-5",),
        context_window=400_000,
        priority=40,
    ),
    # Codex presets (mirrors ref/opencode-openai-codex-auth/config/full-opencode.json)
    ModelInfo(
        provider="codex",
        slug="gpt-5.2-low",
        upstream="gpt-5.2",
        context_window=400_000,
        priority=6,
        extra_body={
            "reasoning": {"effort": "low", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.2-medium",
        upstream="gpt-5.2",
        context_window=400_000,
        priority=7,
        extra_body={
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.2-high",
        upstream="gpt-5.2",
        context_window=400_000,
        priority=8,
        extra_body={
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.2-xhigh",
        upstream="gpt-5.2",
        context_window=400_000,
        priority=9,
        extra_body={
            "reasoning": {"effort": "xhigh", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-max-low",
        upstream="gpt-5.1-codex-max",
        context_window=400_000,
        priority=11,
        extra_body={
            "reasoning": {"effort": "low", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-max-medium",
        upstream="gpt-5.1-codex-max",
        context_window=400_000,
        priority=12,
        extra_body={
            "reasoning": {"effort": "medium", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-max-high",
        upstream="gpt-5.1-codex-max",
        context_window=400_000,
        priority=13,
        extra_body={
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-max-xhigh",
        upstream="gpt-5.1-codex-max",
        context_window=400_000,
        priority=14,
        extra_body={
            "reasoning": {"effort": "xhigh", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-low",
        upstream="gpt-5.1-codex",
        context_window=400_000,
        priority=21,
        extra_body={
            "reasoning": {"effort": "low", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-medium",
        upstream="gpt-5.1-codex",
        context_window=400_000,
        priority=22,
        extra_body={
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-high",
        upstream="gpt-5.1-codex",
        context_window=400_000,
        priority=23,
        extra_body={
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-mini-medium",
        upstream="gpt-5.1-codex-mini",
        context_window=400_000,
        priority=31,
        extra_body={
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-codex-mini-high",
        upstream="gpt-5.1-codex-mini",
        context_window=400_000,
        priority=32,
        extra_body={
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-low",
        upstream="gpt-5.1",
        context_window=400_000,
        priority=41,
        extra_body={
            "reasoning": {"effort": "low", "summary": "auto"},
            "text": {"verbosity": "low"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-medium",
        upstream="gpt-5.1",
        context_window=400_000,
        priority=42,
        extra_body={
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    ModelInfo(
        provider="codex",
        slug="gpt-5.1-high",
        upstream="gpt-5.1",
        context_window=400_000,
        priority=43,
        extra_body={
            "reasoning": {"effort": "high", "summary": "detailed"},
            "text": {"verbosity": "high"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    ),
    # Poe
    ModelInfo(
        provider="poe",
        slug="claude-opus-4.5",
        aliases=("claude-opus-4-5",),
        context_window=200_000,
        priority=10,
        extra_body={"web_search": True},
    ),
    ModelInfo(
        provider="poe",
        slug="claude-sonnet-4.5",
        aliases=("claude-sonnet-4-5",),
        context_window=1_000_000,
        priority=20,
        extra_body={"web_search": True},
    ),
    ModelInfo(
        provider="poe",
        slug="claude-haiku-4.5",
        aliases=("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        context_window=200_000,
        priority=30,
        extra_body={"web_search": True},
    ),
    ModelInfo(
        provider="poe",
        slug="deepseek-v3.2",
        context_window=163_840,
        priority=40,
    ),
    ModelInfo(
        provider="poe",
        slug="glm-4.6",
        context_window=202_752,
        priority=50,
    ),
    ModelInfo(
        provider="poe",
        slug="gpt-5.2-pro",
        context_window=400_000,
        priority=99,
        extra_body={"web_search": True},
    ),
    # OpenRouter
    ModelInfo(
        provider="openrouter",
        slug="claude-opus-4.5",
        upstream="anthropic/claude-opus-4.5",
        context_window=200_000,
        priority=10,
    ),
    ModelInfo(
        provider="openrouter",
        slug="claude-sonnet-4.5",
        upstream="anthropic/claude-sonnet-4.5",
        context_window=1_000_000,
        priority=20,
    ),
    ModelInfo(
        provider="openrouter",
        slug="claude-haiku-4.5",
        upstream="anthropic/claude-haiku-4.5",
        aliases=("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        context_window=200_000,
        priority=30,
    ),
    ModelInfo(
        provider="openrouter",
        slug="gpt-5.2",
        upstream="openai/gpt-5.2",
        context_window=400_000,
        priority=40,
    ),
    ModelInfo(
        provider="openrouter",
        slug="glm-4.6",
        upstream="z-ai/glm-4.6",
        context_window=202_752,
        priority=50,
    ),
    ModelInfo(
        provider="openrouter",
        slug="gpt-5.2-pro",
        upstream="openai/gpt-5.2-pro",
        context_window=400_000,
        priority=99,
    ),
)

DEFAULT_PROVIDER_MODELS: Dict[str, str] = {
    "lmstudio": "gpt-oss-120b",
    "codex": "gpt-5.1-codex",
    "poe": "claude-opus-4.5",
    "openrouter": "claude-opus-4.5",
}


def _normalize(name: str) -> str:
    return name.strip().lower()


def _names_for(info: ModelInfo) -> List[str]:
    names: List[str] = [info.slug]
    if info.upstream:
        names.append(info.upstream)
    names.extend(info.aliases)
    return [n for n in names if n]


def _build_provider_lookup(entries: Iterable[ModelInfo]) -> Dict[str, Dict[str, ModelInfo]]:
    lookup: Dict[str, Dict[str, ModelInfo]] = {}
    for info in entries:
        provider_map = lookup.setdefault(info.provider, {})
        for name in _names_for(info):
            provider_map.setdefault(_normalize(name), info)
    return lookup


def _build_global_lookup(entries: Iterable[ModelInfo]) -> Dict[str, ModelInfo]:
    # Earlier priorities win ties when aliases collide across providers.
    ordered = sorted(entries, key=lambda i: (i.priority, i.provider, i.slug))
    lookup: Dict[str, ModelInfo] = {}
    for info in ordered:
        for name in _names_for(info):
            key = _normalize(name)
            lookup.setdefault(key, info)
    return lookup


_PROVIDER_MODELS: Dict[str, List[ModelInfo]] = {}
for info in MODEL_ENTRIES:
    _PROVIDER_MODELS.setdefault(info.provider, []).append(info)
for provider in _PROVIDER_MODELS:
    _PROVIDER_MODELS[provider].sort(key=lambda i: (i.priority, i.slug))

_PROVIDER_LOOKUP: Dict[str, Dict[str, ModelInfo]] = _build_provider_lookup(MODEL_ENTRIES)
_GLOBAL_LOOKUP: Dict[str, ModelInfo] = _build_global_lookup(MODEL_ENTRIES)


def provider_models(provider: str, expose_only: bool = True) -> List[ModelInfo]:
    """Return known models for a provider, optionally hiding non-exposed entries."""
    models = _PROVIDER_MODELS.get(provider.lower(), [])
    if expose_only:
        return [m for m in models if m.expose]
    return list(models)


def provider_model_slugs(provider: str, expose_only: bool = True) -> List[str]:
    """List provider models as user-facing slugs (no provider prefix)."""
    return [m.slug for m in provider_models(provider, expose_only=expose_only)]


def find_model(provider: Optional[str], name: str) -> Optional[ModelInfo]:
    """Find a model definition by provider/alias/slug."""
    normalized = _normalize(name)
    if provider:
        return _PROVIDER_LOOKUP.get(provider.lower(), {}).get(normalized)
    return _GLOBAL_LOOKUP.get(normalized)


def canonicalize_model(provider: str, name: str) -> str:
    """Normalize a provider/model string to the upstream identifier."""
    info = find_model(provider, name)
    if info:
        return info.target
    fallback = find_model(None, name)
    return fallback.target if fallback else name


def default_context_window_for(model: Optional[str]) -> int:
    """Return a sensible default context window for known models."""
    if not model:
        return 0
    provider = ""
    name = model
    if ":" in model:
        provider, name = model.split(":", 1)
        provider = provider.lower()

    info = find_model(provider or None, name)
    if info:
        return info.context_window
    return 0


def default_extra_body_for(model: Optional[str]) -> Dict[str, Any]:
    """Return provider-specific default extra_body payload settings, if any."""
    if not model:
        return {}
    provider = ""
    name = model
    if ":" in model:
        provider, name = model.split(":", 1)
        provider = provider.lower()

    info = find_model(provider or None, name)
    if info and info.extra_body:
        return copy.deepcopy(info.extra_body)
    return {}

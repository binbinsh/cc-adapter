import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import platformdirs
import requests

logger = logging.getLogger(__name__)

ModelFamily = Literal["codex-max", "codex", "gpt-5.2", "gpt-5.1"]

GITHUB_API_RELEASES = "https://api.github.com/repos/openai/codex/releases/latest"
GITHUB_HTML_RELEASES = "https://github.com/openai/codex/releases/latest"

# These map to Codex CLI prompt selection (see ref/opencode-openai-codex-auth).
PROMPT_FILES: Dict[ModelFamily, str] = {
    "codex-max": "gpt-5.1-codex-max_prompt.md",
    "codex": "gpt_5_codex_prompt.md",
    "gpt-5.2": "gpt-5.1-codex-max_prompt.md",  # GPT-5.2 uses the same prompt as codex-max.
    "gpt-5.1": "gpt_5_1_prompt.md",
}

CACHE_FILES: Dict[ModelFamily, str] = {
    "codex-max": "codex-max-instructions.md",
    "codex": "codex-instructions.md",
    "gpt-5.2": "gpt-5.2-instructions.md",
    "gpt-5.1": "gpt-5.1-instructions.md",
}

_CACHE_TTL_MS = 15 * 60 * 1000  # 15 minutes
_LOCK = threading.Lock()
_MEMO: Dict[ModelFamily, Tuple[int, str]] = {}


def _config_dir() -> Path:
    override = os.getenv("CC_ADAPTER_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return Path(platformdirs.user_config_dir("cc-adapter"))


def _cache_dir() -> Path:
    return _config_dir() / "cache"


def _cache_paths(model_family: ModelFamily) -> Tuple[Path, Path]:
    cache_file = _cache_dir() / CACHE_FILES[model_family]
    meta_file = cache_file.with_name(cache_file.name.replace(".md", "-meta.json"))
    return cache_file, meta_file


def model_family_for(model: str) -> ModelFamily:
    normalized = (model or "").strip().lower()
    if "codex-max" in normalized:
        return "codex-max"
    if "codex" in normalized or normalized.startswith("codex-"):
        return "codex"
    if "gpt-5.2" in normalized:
        return "gpt-5.2"
    return "gpt-5.1"


def _load_meta(meta_file: Path) -> Dict[str, object]:
    try:
        if meta_file.is_file():
            return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _write_meta(meta_file: Path, payload: Dict[str, object]) -> None:
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_file.with_suffix(meta_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(meta_file)


def _latest_release_tag(
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> str:
    try:
        resp = requests.get(GITHUB_API_RELEASES, timeout=timeout, proxies=proxies)
        if resp.ok:
            data = resp.json() if resp.content else {}
            tag = data.get("tag_name")
            if tag:
                return str(tag)
    except Exception:
        pass

    resp = requests.get(GITHUB_HTML_RELEASES, timeout=timeout, proxies=proxies, allow_redirects=True)
    resp.raise_for_status()

    final_url = str(resp.url or "")
    if "/tag/" in final_url:
        tag = final_url.split("/tag/", 1)[1].split("/", 1)[0].strip()
        if tag:
            return tag

    html = resp.text or ""
    marker = "/openai/codex/releases/tag/"
    idx = html.find(marker)
    if idx != -1:
        start = idx + len(marker)
        end = html.find('"', start)
        if end != -1:
            tag = html[start:end].strip()
            if tag and "/" not in tag:
                return tag

    raise RuntimeError("Failed to determine latest openai/codex release tag")


def _raw_prompt_url(tag: str, prompt_file: str) -> str:
    return f"https://raw.githubusercontent.com/openai/codex/{tag}/codex-rs/core/{prompt_file}"


def get_codex_instructions(
    model: str,
    *,
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    force_refresh: bool = False,
) -> str:
    """
    Return the exact Codex CLI system prompt required by the ChatGPT Codex backend.

    The ChatGPT Codex backend rejects arbitrary `instructions` payloads with:
      {"detail":"Instructions are not valid"}
    """

    family = model_family_for(model)
    cache_file, meta_file = _cache_paths(family)
    now_ms = int(time.time() * 1000)

    with _LOCK:
        if not force_refresh and family in _MEMO:
            cached_at_ms, cached = _MEMO[family]
            if cached and now_ms - cached_at_ms < _CACHE_TTL_MS:
                return cached

    meta = _load_meta(meta_file)
    cached_etag = str(meta.get("etag") or "").strip() or None
    cached_tag = str(meta.get("tag") or "").strip() or None
    last_checked_ms = int(meta.get("last_checked_ms") or 0)

    if (
        not force_refresh
        and cache_file.is_file()
        and last_checked_ms > 0
        and now_ms - last_checked_ms < _CACHE_TTL_MS
    ):
        text = cache_file.read_text(encoding="utf-8")
        with _LOCK:
            _MEMO[family] = (now_ms, text)
        return text

    try:
        latest_tag = _latest_release_tag(proxies=proxies, timeout=timeout)
        prompt_file = PROMPT_FILES[family]
        url = _raw_prompt_url(latest_tag, prompt_file)

        etag = cached_etag
        if force_refresh or (cached_tag and cached_tag != latest_tag):
            etag = None

        headers: Dict[str, str] = {}
        if etag and not force_refresh:
            headers["If-None-Match"] = etag

        resp = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        if resp.status_code == 304 and cache_file.is_file() and not force_refresh:
            text = cache_file.read_text(encoding="utf-8")
            _write_meta(
                meta_file,
                {
                    "etag": cached_etag,
                    "tag": latest_tag,
                    "last_checked_ms": now_ms,
                    "url": url,
                },
            )
            with _LOCK:
                _MEMO[family] = (now_ms, text)
            return text

        resp.raise_for_status()
        text = resp.text or ""
        if not text.strip():
            raise RuntimeError("Fetched Codex instructions were empty")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
        _write_meta(
            meta_file,
            {
                "etag": resp.headers.get("etag") or resp.headers.get("ETag"),
                "tag": latest_tag,
                "last_checked_ms": now_ms,
                "url": url,
            },
        )
        with _LOCK:
            _MEMO[family] = (now_ms, text)
        return text
    except Exception as exc:
        if cache_file.is_file():
            logger.warning(
                "Failed to refresh Codex instructions (%s); using cached version: %s",
                family,
                exc,
            )
            text = cache_file.read_text(encoding="utf-8")
            with _LOCK:
                _MEMO[family] = (now_ms, text)
            return text
        raise RuntimeError(f"Failed to fetch Codex instructions ({family}): {exc}") from exc


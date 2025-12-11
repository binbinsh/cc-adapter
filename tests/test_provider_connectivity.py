import http.client
import json
import os
import threading
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace

import pytest

from cc_adapter import server
from cc_adapter.config import Settings
from cc_adapter.model_registry import provider_model_slugs

# Match GUI dropdown models (provider-prefixed here).
GUI_TEST_MODELS = [f"poe:{name}" for name in provider_model_slugs("poe")] + [
    f"openrouter:{name}" for name in provider_model_slugs("openrouter")
]


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _filter_models_by_keys(models: List[str], poe_key: str, openrouter_key: str) -> List[str]:
    filtered: List[str] = []
    for model in models:
        if model.startswith("poe:") and not poe_key:
            continue
        if model.startswith("openrouter:") and not openrouter_key:
            continue
        filtered.append(model)
    return filtered


def _start_server(settings: Settings):
    http_server = server.build_server(settings)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    return http_server, thread


def _stop_server(http_server, thread):
    http_server.shutdown()
    http_server.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def live_settings() -> Tuple[Settings, List[str]]:
    if os.environ.get("RUN_LIVE_PROVIDER_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_PROVIDER_TESTS=1 to run live provider connectivity tests")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    _load_env_file(env_path)
    poe_key = (os.environ.get("POE_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not poe_key and not openrouter_key:
        pytest.skip("No provider API keys found in environment/.env (POE_API_KEY/OPENROUTER_API_KEY)")
    models = _filter_models_by_keys(GUI_TEST_MODELS, poe_key, openrouter_key)
    settings = Settings(host="127.0.0.1", port=0, model=models[0])
    settings.poe_api_key = poe_key
    settings.openrouter_key = openrouter_key
    return settings, models


def test_gui_models_connectivity(live_settings, capsys):
    base_settings, models = live_settings
    for model in models:
        settings = replace(base_settings, model=model, port=0)
        http_server, thread = _start_server(settings)
        host, port = http_server.server_address
        try:
            conn = http.client.HTTPConnection(host, port, timeout=60)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Who are you? (respond in English)"}],
                "max_tokens": 16,
            }
            body = json.dumps(payload).encode("utf-8")
            conn.request("POST", "/v1/messages", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            conn.close()

            snippet = " ".join(data.split())
            with capsys.disabled():
                print(f"[live-test] {model} -> HTTP {resp.status} | {snippet}")

            if resp.status == 400 and "not a valid model" in data.lower():
                pytest.skip(f"{model} not accepted by provider: {snippet}")

            assert resp.status == 200, f"{model} -> {snippet}"
            parsed = json.loads(data)
            assert "error" not in parsed, f"{model} -> {snippet}"
            content = parsed.get("content") or []
            assert content, f"No content returned for {model}: {parsed}"
        finally:
            _stop_server(http_server, thread)

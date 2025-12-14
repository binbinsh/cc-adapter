# CC Adapter

[![PyPI Version](https://img.shields.io/pypi/v/cc-adapter)](https://pypi.org/project/cc-adapter/)
[![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https://pypistats.org/api/packages/cc-adapter/recent&query=data.last_month&label=downloads/month)](https://pypistats.org/packages/cc-adapter)

cc-adapter lets Claude Code speak Anthropic `/v1/messages` to Poe, OpenRouter, OpenAI Codex subscription or LM Studio. It supports streaming, tool calls, and cache-control passthrough.

## Tested models (quick list)
Choose one of these thoroughly tested models:

- **`codex:gpt-5.2-xhigh` (requires OpenAI Codex Subscription)**
- **`poe:claude-opus-4.5` (best quality; requires Poe key)**
- `poe:claude-sonnet-4.5` (requires Poe key)
- **`poe:deepseek-v3.2` (best value for money; requires Poe key)**
- `poe:glm-4.6` (requires Poe key)
- `openrouter:claude-sonnet-4.5` (requires OpenRouter key)
- `openrouter:claude-opus-4.5` (requires OpenRouter key)
- `openrouter:gpt-5.2` (requires OpenRouter key)
- `openrouter:glm-4.6` (requires OpenRouter key)
- `lmstudio:gpt-oss-120b` (requires LM Studio + gpt-oss-120b)
- `poe:gpt-5.2-pro` (requires Poe key; **extremely expensive!**)
- `openrouter:gpt-5.2-pro` (requires OpenRouter key; **extremely expensive!**)

## Install from PyPI (recommended)
Quickest way to get cc-adapter:

```bash
uv tool install cc-adapter
```

## Install from source
Clone and set up a local dev environment:

```bash
git clone https://github.com/binbinsh/cc-adapter.git
cd cc-adapter/
uv venv --python 3.10
```

## GUI (recommended for daily use)
Launch the Tkinter GUI to configure and start/stop the adapter:

```bash
uv run cc-adapter-gui
```

Set provider/model/API keys in the window, then use `Test Provider` and `Start/Stop`.

<img src="https://raw.githubusercontent.com/binbinsh/cc-adapter/main/screenshot-01.png" alt="CC Adapter GUI" width="800">

## OpenAI Codex (ChatGPT OAuth)

This provider uses **ChatGPT OAuth** (subscription-based) to call the Codex backend. You do **not** need an `OPENAI_API_KEY`.

```bash
cc-adapter-codex-login
uv run cc-adapter --model codex:gpt-5.1-codex-mini
```

GUI: choose provider `OpenAI Codex`, then click `Login` and `Start`.

<img src="https://raw.githubusercontent.com/binbinsh/cc-adapter/main/screenshot-02.png" alt="CC Adapter GUI" width="800">

## Quick start (minimal CLI)
Run the recommended Poe model and point Claude Code to the adapter:

```bash
uv run cc-adapter --host 127.0.0.1 --port 8005 \
  --model poe:claude-opus-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon

export ANTHROPIC_BASE_URL=http://127.0.0.1:8005
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=
export NO_PROXY=127.0.0.1
export DISABLE_TELEMETRY=true
export DISABLE_COST_WARNINGS=true
export API_TIMEOUT_MS=600000
export CLAUDE_CODE_USE_BEDROCK=
claude
```

Notes:
- `--host` defaults to `127.0.0.1`; only use `0.0.0.0` if you must expose it.
- Change `--port` if 8005 is taken.
- Swap `--model` for any model from the provider.

## CLI examples
The CLI accepts any provider-prefixed model string (e.g., `poe:any-model-name`); the GUI offers a curated drop-down list. Common flags: `--host` (default 127.0.0.1), `--port` (default 8005), plus provider-specific API keys.

### Poe
```bash
uv run cc-adapter --host 127.0.0.1 --port 8005 \
  --model poe:claude-opus-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon
```

### OpenRouter
```bash
uv run cc-adapter --host 127.0.0.1 --port 8005 \
  --model openrouter:claude-opus-4.5 \
  --openrouter-api-key YOUR_OPENROUTER_API_KEY \
  --daemon
```

### LM Studio
```bash
uv run cc-adapter --host 127.0.0.1 --port 8005 \
  --model lmstudio:gpt-oss-120b \
  --lmstudio-base http://127.0.0.1:1234/v1/chat/completions \
  --lmstudio-timeout 3600 \
  --daemon
```

## Proxy support (optional)
Only set these if your network blocks the provider URLs:

```bash
export HTTP_PROXY=http://localhost:port
export HTTPS_PROXY=http://localhost:port
export NO_PROXY=127.0.0.1,localhost
uv run cc-adapter --model poe:claude-opus-4.5 --poe-api-key YOUR_POE_API_KEY
```

## Run Claude Code
Point Claude Code to the adapter:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8005
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=
export NO_PROXY=127.0.0.1
export DISABLE_TELEMETRY=true
export DISABLE_COST_WARNINGS=true
export API_TIMEOUT_MS=600000
export CLAUDE_CODE_USE_BEDROCK=

claude
```

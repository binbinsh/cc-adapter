# CC Adapter

[![PyPI Version](https://img.shields.io/pypi/v/cc-adapter)](https://pypi.org/project/cc-adapter/)
[![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https://pypistats.org/api/packages/cc-adapter/recent&query=data.last_month&label=downloads/month)](https://pypistats.org/packages/cc-adapter)

Claude Code-compatible adapter bridging Anthropic `/v1/messages` to LM Studio, Poe, or OpenRouter and back. Streaming, tool calls, and cache-control passthrough are supported.

## Install from PyPI (recommended)
Quickest way to get cc-adapter.

```bash
uv tool install cc-adapter
```

## Install from source
Clone and set up a local dev environment.

```bash
git clone https://github.com/binbinsh/cc-adapter.git
cd cc-adapter/
uv venv --python 3.12
uv sync
```

## GUI (recommended)
Launch the Tkinter GUI for configure/start/stop:
```
uv run cc-adapter-gui
```

Set provider/model/API keys in the window, then use `Test Provider` and `Start/Stop`

<img src="https://raw.githubusercontent.com/binbinsh/cc-adapter/main/screenshot.png" alt="CC Adapter GUI" width="800">

## LM Studio backend (CLI)
Run cc-adapter against an LM Studio OpenAI-compatible server with gpt-oss-120b.

```bash
uv run cc-adapter --host 0.0.0.0 --port 8005 \
  --model lmstudio:gpt-oss-120b \
  --lmstudio-base http://127.0.0.1:1234/v1/chat/completions \
  --lmstudio-timeout 3600 \
  --daemon
```

## Poe backend (CLI)
Run cc-adapter with Poe as the provider and claude-opus-4.5 model,
```bash
uv run cc-adapter --host 0.0.0.0 --port 8005 \
  --model poe:claude-opus-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon
```

or claude-sonnet-4.5 for cheaper model,

```bash
uv run cc-adapter --host 0.0.0.0 --port 8005 \
  --model poe:claude-sonnet-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon
```

## OpenRouter backend (CLI)
Run cc-adapter with OpenRouter as the provider.
```bash
uv run cc-adapter --host 0.0.0.0 --port 8005 \
  --model openrouter:claude-opus-4.5 \
  --openrouter-api-key YOUR_OPENROUTER_API_KEY \
  --daemon
```

or

```bash
uv run cc-adapter --host 0.0.0.0 --port 8005 \
  --model openrouter:claude-sonnet-4.5 \
  --openrouter-api-key YOUR_OPENROUTER_API_KEY \
  --daemon
```

## Proxy support
Use your OS proxy environment variables when VPN/firewall rules block provider calls.
Set `HTTP_PROXY`/`HTTPS_PROXY` (or `ALL_PROXY` for SOCKS) and `NO_PROXY` for hosts to bypass; your VPN/proxy app or system network settings show the host/port (e.g. Clash/Surge local port).

```bash
export HTTP_PROXY=http://localhost:port
export HTTPS_PROXY=http://localhost:port
export NO_PROXY=127.0.0.1,localhost
uv run cc-adapter --model poe:claude-sonnet-4.5 --poe-api-key YOUR_POE_API_KEY
```

## Run Claude Code
Point Claude Code to the adapter using these env vars.

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

## Tested models
Choose one of these thoroughly tested models (provider prefix required).

- `lmstudio:gpt-oss-120b` (requires LM Studio + gpt-oss-120b)
- `poe:claude-sonnet-4.5` (requires Poe key)
- `poe:claude-opus-4.5` (requires Poe key)
- `openrouter:claude-sonnet-4.5` (requires OpenRouter key)
- `openrouter:claude-opus-4.5` (requires OpenRouter key)

# CC Adapter

Claude Code-compatible adapter bridging Anthropic `/v1/messages` to LM Studio, Poe, or OpenRouter and back. Streaming, tool calls, and cache-control passthrough are supported.

## Install from PyPI (recommended)
Quickest way to get cc-adapter.

```bash
uv tool install cc-adapter
```

### Install from source
Clone and set up a local dev environment.

```bash
git clone https://github.com/binbinsh/cc-adapter.git
cd cc-adapter/
uv venv --python 3.12
uv sync
```

## LM Studio backend
Run cc-adapter against an LM Studio OpenAI-compatible server with gpt-oss-120b.

```bash
uv run cc-adapter --host 0.0.0.0 --port 8000 \
  --model lmstudio:gpt-oss-120b \
  --lmstudio-base http://127.0.0.1:1234/v1/chat/completions \
  --lmstudio-timeout 3600 \
  --daemon
```

## Poe backend
Run cc-adapter with Poe as the provider and claude-opus-4.5 model,
```bash
uv run cc-adapter --host 0.0.0.0 --port 8000 \
  --model poe:claude-opus-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon
```

or claude-sonnet-4.5 for cheaper model,

```bash
uv run cc-adapter --host 0.0.0.0 --port 8000 \
  --model poe:claude-sonnet-4.5 \
  --poe-api-key YOUR_POE_API_KEY \
  --daemon
```

## OpenRouter backend
Run cc-adapter with OpenRouter as the provider.
```bash
uv run cc-adapter --host 0.0.0.0 --port 8000 \
  --model openrouter:claude-opus-4.5 \
  --openrouter-api-key YOUR_OPENROUTER_API_KEY \
  --daemon
```

or

```bash
uv run cc-adapter --host 0.0.0.0 --port 8000 \
  --model openrouter:claude-sonnet-4.5 \
  --openrouter-api-key YOUR_OPENROUTER_API_KEY \
  --daemon
```

## Supported models
Choose one of these models (provider prefix required).

- `lmstudio:gpt-oss-120b` (requires LM Studio + gpt-oss-120b)
- `poe:claude-sonnet-4.5` (requires Poe key)
- `poe:claude-opus-4.5` (requires Poe key)
- `openrouter:claude-sonnet-4.5` (requires OpenRouter key)
- `openrouter:claude-opus-4.5` (requires OpenRouter key)

`GET /v1/models` returns the available list based on which keys are configured.

## Run Claude Code
Point Claude Code to the adapter using these env vars.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_API_KEY=
export NO_PROXY=127.0.0.1
export DISABLE_TELEMETRY=true
export DISABLE_COST_WARNINGS=true
export API_TIMEOUT_MS=600000
export CLAUDE_CODE_USE_BEDROCK=

claude
```

## Endpoints
HTTP entrypoints exposed by cc-adapter.
- `/health` – health check
- `/v1/messages` – Anthropic-compatible entrypoint (Claude Code target)
- `/v1/models` – list available models
- `/v1/messages/count_tokens` – rough prompt token estimate for a request body (heuristic)

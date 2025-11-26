#!/usr/bin/env python3
"""
Main adapter server entrypoint.
"""

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
from typing import Any, Dict
from urllib.parse import urlparse
from subprocess import Popen, DEVNULL

from .config import Settings, load_settings, apply_overrides
from .models import resolve_provider_model
from .converters import anthropic_to_openai, openai_to_anthropic
from .providers import lmstudio, poe, openrouter
from . import streaming

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cc-adapter")


def _available_models(settings: Settings):
    models = [f"lmstudio:{settings.lmstudio_model}"]
    if settings.poe_api_key:
        models.extend(
            [
                "poe:claude-sonnet-4.5",
                "poe:claude-opus-4.5",
            ]
        )
    if settings.openrouter_key:
        models.extend(
            [
                "openrouter:claude-sonnet-4.5",
                "openrouter:claude-opus-4.5",
            ]
        )
    return models


def _is_allowed_model(provider: str, target_model: str, settings: Settings) -> bool:
    allowed = set(_available_models(settings))
    full = f"{provider}:{target_model}"
    if full in allowed:
        return True
    # Allow anthropic/ prefix variants for OpenRouter Claude models
    if provider == "openrouter" and target_model.startswith("anthropic/"):
        stripped = target_model.split("/", 1)[1]
        return f"{provider}:{stripped}" in allowed
    return False


def _port_available(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AdapterHTTPServer(ThreadingHTTPServer):
    """HTTP server that suppresses noisy client disconnect tracebacks."""

    def handle_error(self, request, client_address):
        exc_type, exc, _ = sys.exc_info()
        if exc_type is ConnectionResetError:
            logger.debug("Client disconnected during request from %s", client_address)
            return
        return super().handle_error(request, client_address)


class AdapterHandler(BaseHTTPRequestHandler):
    settings: Settings = load_settings()

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], format % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return _json_response(self, 200, {"status": "ok"})
        if parsed.path == "/v1/models":
            return _json_response(
                self,
                200,
                {
                    "data": [{"id": m, "object": "model"} for m in _available_models(self.settings)],
                },
            )
        if parsed.path == "/v1/messages/count_tokens":
            # Lightweight token estimate based on character counts.
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            try:
                incoming = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception as exc:
                logger.exception("Failed to parse count_tokens request")
                return _json_response(self, 400, {"error": f"Invalid JSON: {exc}"})
            prompt_tokens = streaming.estimate_prompt_tokens(incoming)
            return _json_response(self, 200, {"input_tokens": prompt_tokens})
        return _json_response(self, 404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/v1/messages/count_tokens":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length else b"{}"
                incoming = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception as exc:
                logger.exception("Failed to parse count_tokens request")
                return _json_response(self, 400, {"error": f"Invalid JSON: {exc}"})
            prompt_tokens = streaming.estimate_prompt_tokens(incoming)
            return _json_response(self, 200, {"input_tokens": prompt_tokens})
        if parsed.path != "/v1/messages":
            return _json_response(self, 404, {"error": "Not Found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            incoming = json.loads(raw_body.decode("utf-8") or "{}")
        except Exception as exc:
            logger.exception("Failed to parse incoming request")
            return _json_response(self, 400, {"error": f"Invalid JSON: {exc}"})

        try:
            provider, target_model = resolve_provider_model(
                incoming.get("model"), self.settings
            )
        except ValueError as exc:
            return _json_response(self, 400, {"error": str(exc)})

        # Normalize OpenRouter model names: auto-add anthropic/ prefix for Claude models if missing
        if provider == "openrouter":
            if "/" not in target_model and target_model.lower().startswith("claude"):
                target_model = f"anthropic/{target_model}"

        if not _is_allowed_model(provider, target_model, self.settings):
            allowed = ", ".join(_available_models(self.settings))
            return _json_response(
                self,
                400,
                {"error": f"Model not allowed. Provide one of: {allowed} (use CC_ADAPTER_MODEL or --model)."},
            )

        try:
            openai_payload = anthropic_to_openai(incoming, target_model)
        except Exception as exc:
            logger.exception("Failed to translate Anthropic request")
            return _json_response(self, 400, {"error": f"Bad request: {exc}"})

        # Poe direct (OpenAI-compatible API)
        if provider == "poe":
            if not self.settings.poe_api_key:
                return _json_response(self, 400, {"error": "POE_API_KEY not set"})
            if openai_payload.get("stream"):
                return self._handle_poe_stream(openai_payload, target_model, incoming)
            return self._handle_poe(openai_payload, target_model, incoming)

        # OpenRouter
        if provider == "openrouter":
            if not self.settings.openrouter_key:
                return _json_response(self, 400, {"error": "OPENROUTER_API_KEY not set"})
            if openai_payload.get("stream"):
                return self._handle_openrouter_stream(
                    openai_payload, target_model, incoming
                )
            try:
                openrouter_response = openrouter.send(openai_payload, self.settings)
                outgoing = openai_to_anthropic(openrouter_response, target_model, incoming)
                return _json_response(self, 200, outgoing)
            except Exception as exc:
                logger.exception("OpenRouter request failed")
                return _json_response(self, 502, {"error": f"OpenRouter error: {exc}"})

        # LM Studio
        if openai_payload.get("stream"):
            return self._handle_lm_stream(openai_payload, target_model, incoming)

        try:
            lmstudio_response = lmstudio.send(openai_payload, self.settings)
            outgoing = openai_to_anthropic(lmstudio_response, target_model, incoming)
            return _json_response(self, 200, outgoing)
        except Exception as exc:
            logger.exception("LM Studio request failed")
            return _json_response(self, 502, {"error": f"LM Studio error: {exc}"})

    # Provider dispatch helpers
    def _handle_poe(self, payload: Dict[str, Any], bot_name: str, incoming: Dict[str, Any]):
        try:
            poe_response = poe.send(payload, self.settings, bot_name, incoming)
            return _json_response(self, 200, poe_response)
        except Exception as exc:
            logger.exception("Poe request failed")
            return _json_response(self, 502, {"error": f"Poe error: {exc}"})

    def _handle_poe_stream(
        self, payload: Dict[str, Any], bot_name: str, incoming: Dict[str, Any]
    ):
        try:
            return poe.stream(payload, self.settings, bot_name, incoming, self, logger)
        except Exception as exc:
            logger.exception("Poe stream failed")
            return _json_response(self, 502, {"error": f"Poe error: {exc}"})

    def _handle_openrouter_stream(
        self, payload: Dict[str, Any], target_model: str, incoming: Dict[str, Any]
    ):
        try:
            return openrouter.stream(payload, self.settings, target_model, incoming, self, logger)
        except Exception as exc:
            logger.exception("OpenRouter stream failed")
            return _json_response(self, 502, {"error": f"OpenRouter error: {exc}"})

    def _handle_lm_stream(
        self, payload: Dict[str, Any], target_model: str, incoming: Dict[str, Any]
    ):
        try:
            return lmstudio.stream(payload, self.settings, target_model, incoming, self, logger)
        except Exception as exc:
            logger.exception("LM Studio stream failed")
            return _json_response(self, 502, {"error": f"LM Studio error: {exc}"})


def run_server(settings: Settings):
    server = AdapterHTTPServer((settings.host, settings.port), AdapterHandler)
    AdapterHandler.settings = settings  # type: ignore
    logger.info(
        "Adapter listening on http://%s:%s (model=%s)",
        settings.host,
        settings.port,
        settings.model or "client-provided-only",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down adapter")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="LM Studio / Poe / OpenRouter adapter")
    parser.add_argument("--host", default=os.getenv("ADAPTER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ADAPTER_PORT", "8000")))
    parser.add_argument("--daemon", action="store_true", help="run in background")
    parser.add_argument(
        "--model",
        required=False,
        help="Required default model with provider prefix (e.g., poe:claude-opus-4.5); request must include model if this is omitted. Can also set CC_ADAPTER_MODEL.",
    )
    parser.add_argument("--lmstudio-base", help="LM Studio base URL (OpenAI compatible)")
    parser.add_argument("--lmstudio-model", help="LM Studio model name")
    parser.add_argument("--lmstudio-timeout", type=float, help="LM Studio timeout (seconds)")
    parser.add_argument("--poe-api-key", help="Poe API key")
    parser.add_argument("--poe-base-url", help="Poe base URL")
    parser.add_argument("--openrouter-api-key", help="OpenRouter API key")
    parser.add_argument("--openrouter-base", help="OpenRouter base URL")
    args = parser.parse_args()

    if args.daemon:
        if not _port_available(args.host, args.port):
            print(f"Port {args.host}:{args.port} already in use. Is cc-adapter already running?")
            return
        cmd = [sys.executable, "-m", "cc_adapter.server", "--host", args.host, "--port", str(args.port)]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.lmstudio_base:
            cmd.extend(["--lmstudio-base", args.lmstudio_base])
        if args.lmstudio_model:
            cmd.extend(["--lmstudio-model", args.lmstudio_model])
        if args.lmstudio_timeout:
            cmd.extend(["--lmstudio-timeout", str(args.lmstudio_timeout)])
        if args.poe_api_key:
            cmd.extend(["--poe-api-key", args.poe_api_key])
        if args.poe_base_url:
            cmd.extend(["--poe-base-url", args.poe_base_url])
        if args.openrouter_api_key:
            cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
        if args.openrouter_base:
            cmd.extend(["--openrouter-base", args.openrouter_base])
        proc = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL, close_fds=True)
        print(f"Started daemon pid={proc.pid}")
        return

    settings = load_settings()
    overrides = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "lmstudio_base": args.lmstudio_base,
        "lmstudio_model": args.lmstudio_model,
        "lmstudio_timeout": args.lmstudio_timeout,
        "poe_api_key": args.poe_api_key,
        "poe_base_url": args.poe_base_url,
        "openrouter_key": args.openrouter_api_key,
        "openrouter_base": args.openrouter_base,
    }
    settings = apply_overrides(settings, overrides)
    run_server(settings)


if __name__ == "__main__":
    main()

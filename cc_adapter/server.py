#!/usr/bin/env python3
"""
Main adapter server entrypoint.
"""

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from subprocess import Popen, DEVNULL

from .config import Settings, load_settings, apply_overrides
from .models import available_models, normalize_model_spec, resolve_provider_model
from .converters import anthropic_to_openai, openai_to_anthropic
from .providers import lmstudio, poe, openrouter, codex
from . import streaming
from .logging_utils import log_payload, resolve_log_level

logging.basicConfig(
    level=resolve_log_level(os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("cc-adapter")

CODEX_HAIKU_FALLBACK_MODEL = "gpt-5.1-codex-mini"

def _effective_codex_settings(
    settings: Settings,
    requested_model: Any,
    upstream_model: str,
) -> tuple[Settings, str]:
    requested_text = str(requested_model or "")
    if not requested_text or "claude-haiku" not in requested_text.lower():
        return settings, upstream_model
    if not str(getattr(settings, "model", "") or "").lower().startswith("codex:"):
        return settings, upstream_model
    return replace(settings, model=f"codex:{CODEX_HAIKU_FALLBACK_MODEL}"), CODEX_HAIKU_FALLBACK_MODEL


def port_available(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        logger.info("Client disconnected before response could be sent (status=%s)", status)
        handler.close_connection = True
    except Exception:
        handler.close_connection = True
        logger.exception("Failed to send JSON response")


class AdapterHTTPServer(ThreadingHTTPServer):
    """HTTP server that suppresses noisy client disconnect tracebacks."""

    def handle_error(self, request, client_address):
        exc_type, exc, _ = sys.exc_info()
        if exc_type in (ConnectionResetError, BrokenPipeError):
            logger.debug("Client disconnected during request from %s", client_address)
            return
        return super().handle_error(request, client_address)


class AdapterHandler(BaseHTTPRequestHandler):
    settings: Settings = load_settings()

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], format % args)

    def _client_is_loopback(self) -> bool:
        host = str((self.client_address[0] or "")).strip()
        return host == "127.0.0.1" or host == "::1" or host.startswith("127.")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return _json_response(self, 200, {"status": "ok"})
        if parsed.path == "/v1/models":
            return _json_response(
                self,
                200,
                {
                    "data": [{"id": m, "object": "model"} for m in available_models(self.settings)],
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
        if parsed.path == "/shutdown":
            if not self._client_is_loopback():
                return _json_response(self, 403, {"error": "Forbidden"})

            _json_response(self, 200, {"status": "shutting_down"})

            def _shutdown():
                try:
                    self.server.shutdown()
                except Exception:
                    logger.exception("Shutdown request failed")
                try:
                    self.server.server_close()
                except Exception:
                    pass

            threading.Thread(target=_shutdown, daemon=True).start()
            self.close_connection = True
            return
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

        log_payload(logger, "Incoming /v1/messages payload", incoming)

        try:
            requested_model = incoming.get("model")
            provider, target_model = resolve_provider_model(requested_model, self.settings)
            normalized_model = False
        except ValueError as exc:
            return _json_response(self, 400, {"error": str(exc)})

        effective_settings = self.settings
        codex_haiku_override = ""
        if provider == "codex":
            effective_settings, effective_model = _effective_codex_settings(
                self.settings, requested_model, target_model
            )
            if effective_model != target_model:
                codex_haiku_override = effective_model
                target_model = effective_model

        resolution_bits = []
        if requested_model and requested_model != target_model:
            resolution_bits.append(f"requested={requested_model}")
        if normalized_model:
            resolution_bits.append("alias->canonical")
        if (
            provider == "lmstudio"
            and requested_model
            and "claude-haiku" in requested_model.lower()
            and target_model == self.settings.lmstudio_model
        ):
            resolution_bits.append("haiku->lmstudio_default")
        if codex_haiku_override:
            resolution_bits.append(f"haiku->{codex_haiku_override}")
        suffix = f" ({'; '.join(resolution_bits)})" if resolution_bits else ""
        logger.info("Resolved model %s:%s%s", provider, target_model, suffix)

        try:
            openai_payload = anthropic_to_openai(incoming, target_model)
        except Exception as exc:
            logger.exception("Failed to translate Anthropic request")
            return _json_response(self, 400, {"error": f"Bad request: {exc}"})

        log_payload(
            logger,
            f"Sending payload to {provider}:{target_model}",
            openai_payload,
        )

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
                openrouter_response = openrouter.send(openai_payload, self.settings, target_model)
                outgoing = openai_to_anthropic(openrouter_response, target_model, incoming)
                log_payload(logger, "OpenRouter response", openrouter_response)
                log_payload(logger, "Responding to client", outgoing)
                return _json_response(self, 200, outgoing)
            except Exception as exc:
                logger.exception("OpenRouter request failed")
                return _json_response(self, 502, {"error": f"OpenRouter error: {exc}"})

        # OpenAI Codex (ChatGPT OAuth)
        if provider == "codex":
            if openai_payload.get("stream"):
                return self._handle_codex_stream(openai_payload, target_model, incoming, effective_settings)
            try:
                codex_response = codex.send(openai_payload, effective_settings, target_model)
                outgoing = openai_to_anthropic(codex_response, target_model, incoming)
                log_payload(logger, "Codex response", codex_response)
                log_payload(logger, "Responding to client", outgoing)
                return _json_response(self, 200, outgoing)
            except Exception as exc:
                logger.exception("Codex request failed")
                return _json_response(self, 502, {"error": f"Codex error: {exc}"})

        # LM Studio
        if openai_payload.get("stream"):
            return self._handle_lm_stream(openai_payload, target_model, incoming)

        try:
            lmstudio_response = lmstudio.send(openai_payload, self.settings)
            outgoing = openai_to_anthropic(lmstudio_response, target_model, incoming)
            log_payload(logger, "LM Studio response", lmstudio_response)
            log_payload(logger, "Responding to client", outgoing)
            return _json_response(self, 200, outgoing)
        except Exception as exc:
            logger.exception("LM Studio request failed")
            return _json_response(self, 502, {"error": f"LM Studio error: {exc}"})

    # Provider dispatch helpers
    def _handle_poe(self, payload: Dict[str, Any], bot_name: str, incoming: Dict[str, Any]):
        try:
            poe_response = poe.send(payload, self.settings, bot_name, incoming)
            log_payload(logger, "Responding to client", poe_response)
            return _json_response(self, 200, poe_response)
        except Exception as exc:
            logger.exception("Poe request failed")
            return _json_response(self, 502, {"error": f"Poe error: {exc}"})

    def _handle_poe_stream(
        self, payload: Dict[str, Any], bot_name: str, incoming: Dict[str, Any]
    ):
        try:
            return poe.stream(payload, self.settings, bot_name, incoming, self, logger)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during Poe stream")
            self.close_connection = True
            return
        except Exception as exc:
            logger.exception("Poe stream failed")
            return _json_response(self, 502, {"error": f"Poe error: {exc}"})

    def _handle_openrouter_stream(
        self, payload: Dict[str, Any], target_model: str, incoming: Dict[str, Any]
    ):
        try:
            return openrouter.stream(payload, self.settings, target_model, incoming, self, logger)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during OpenRouter stream")
            self.close_connection = True
            return
        except Exception as exc:
            logger.exception("OpenRouter stream failed")
            return _json_response(self, 502, {"error": f"OpenRouter error: {exc}"})

    def _handle_codex_stream(
        self,
        payload: Dict[str, Any],
        target_model: str,
        incoming: Dict[str, Any],
        settings: Optional[Settings] = None,
    ):
        try:
            return codex.stream(payload, settings or self.settings, target_model, incoming, self, logger)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during Codex stream")
            self.close_connection = True
            return
        except Exception as exc:
            logger.exception("Codex stream failed")
            return _json_response(self, 502, {"error": f"Codex error: {exc}"})

    def _handle_lm_stream(
        self, payload: Dict[str, Any], target_model: str, incoming: Dict[str, Any]
    ):
        try:
            return lmstudio.stream(payload, self.settings, target_model, incoming, self, logger)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during LM Studio stream")
            self.close_connection = True
            return
        except Exception as exc:
            logger.exception("LM Studio stream failed")
            return _json_response(self, 502, {"error": f"LM Studio error: {exc}"})


def run_server(settings: Settings):
    server = build_server(settings)
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


def build_server(settings: Settings) -> AdapterHTTPServer:
    """Create a configured AdapterHTTPServer without starting it."""
    settings.apply_no_proxy_env()
    server = AdapterHTTPServer((settings.host, settings.port), AdapterHandler)
    AdapterHandler.settings = settings  # type: ignore
    return server


def _ensure_codex_login_if_needed(settings: Settings, no_browser: bool, allow_interactive: bool) -> None:
    model = (settings.model or "").strip()
    if not model.lower().startswith("codex:"):
        return

    auth_mode = str(getattr(settings, "codex_auth", "") or "").strip().lower()
    env_present = bool(
        getattr(settings, "codex_access_token", "")
        and getattr(settings, "codex_refresh_token", "")
        and int(getattr(settings, "codex_expires_at_ms", 0) or 0) > 0
    )

    from .codex_oauth import load_tokens

    file_present = load_tokens() is not None

    if auth_mode in {"env", "token", "tokens"}:
        if not env_present:
            raise RuntimeError(
                "Codex auth mode is env but tokens are missing (set OPENAI_CODEX_ACCESS_TOKEN/OPENAI_CODEX_REFRESH_TOKEN/OPENAI_CODEX_EXPIRES_AT_MS)."
            )
        return

    if auth_mode in {"oauth", "login", "file", "stored"}:
        if file_present:
            return
        if not allow_interactive:
            raise RuntimeError("OpenAI Codex OAuth not configured (run cc-adapter-codex-login).")
        logger.info("OpenAI Codex OAuth not found. Starting OAuth login flow...")
        from .codex_auth import login

        login(no_browser=no_browser)
        return

    # auto
    if env_present or file_present:
        return
    if not allow_interactive:
        raise RuntimeError("OpenAI Codex OAuth not configured (run cc-adapter-codex-login).")
    logger.info("OpenAI Codex OAuth not found. Starting OAuth login flow...")
    from .codex_auth import login

    login(no_browser=no_browser)


def main():
    parser = argparse.ArgumentParser(description="LM Studio / Poe / OpenRouter / OpenAI Codex adapter")
    parser.add_argument("--host", default=os.getenv("ADAPTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ADAPTER_PORT", "8005")))
    parser.add_argument("--daemon", action="store_true", help="run in background")
    parser.add_argument(
        "--model",
        required=False,
        help="Default model as provider:model (e.g., poe:claude-opus-4.5) or provider-only shorthand (e.g., codex:). Can also set CC_ADAPTER_MODEL.",
    )
    parser.add_argument(
        "--provider",
        required=False,
        help="Provider prefix to pair with an unprefixed --model (poe|lmstudio|openrouter|codex). Example: --provider codex --model gpt-5.2-high",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=int(os.getenv("CONTEXT_WINDOW", "0")),
        help="Context window in tokens (prompt + completion); defaults per model if omitted.",
    )
    parser.add_argument("--lmstudio-base", help="LM Studio base URL (OpenAI compatible)")
    parser.add_argument("--lmstudio-model", help="LM Studio model name")
    parser.add_argument("--lmstudio-timeout", type=float, help="LM Studio timeout (seconds)")
    parser.add_argument("--poe-api-key", help="Poe API key")
    parser.add_argument("--poe-base-url", help="Poe base URL")
    parser.add_argument("--poe-max-retries", type=int, help="Poe retry attempts for upstream errors")
    parser.add_argument("--poe-retry-backoff", type=float, help="Seconds between retry attempts")
    parser.add_argument("--openrouter-api-key", help="OpenRouter API key")
    parser.add_argument("--openrouter-base", help="OpenRouter base URL")
    parser.add_argument("--codex-base-url", help="OpenAI Codex base URL (ChatGPT backend)")
    parser.add_argument(
        "--codex-auth",
        choices=["auto", "oauth", "env"],
        help="Codex auth mode: auto (env tokens or stored OAuth), oauth (stored OAuth only), env (env tokens only).",
    )
    parser.add_argument(
        "--codex-no-browser",
        action="store_true",
        help="Do not attempt to open a browser during Codex OAuth login (manual paste fallback).",
    )
    args = parser.parse_args()

    model_arg = args.model
    provider_arg = str(getattr(args, "provider", "") or "").strip()
    if provider_arg:
        provider = provider_arg.rstrip(":").lower()
        if provider not in {"poe", "lmstudio", "openrouter", "codex"}:
            parser.error(f"Unsupported provider: {provider_arg}")
        if model_arg:
            if ":" in model_arg:
                model_provider = model_arg.split(":", 1)[0].lower()
                if model_provider != provider:
                    parser.error(f"--provider {provider_arg} conflicts with --model {model_arg}")
            else:
                model_arg = f"{provider}:{model_arg}"
        else:
            model_arg = f"{provider}:"
    model_arg = normalize_model_spec(model_arg)

    overrides = {
        "host": args.host,
        "port": args.port,
        "model": model_arg,
        "context_window": args.context_window,
        "lmstudio_base": args.lmstudio_base,
        "lmstudio_model": args.lmstudio_model,
        "lmstudio_timeout": args.lmstudio_timeout,
        "poe_api_key": args.poe_api_key,
        "poe_base_url": args.poe_base_url,
        "poe_max_retries": args.poe_max_retries,
        "poe_retry_backoff": args.poe_retry_backoff,
        "openrouter_key": args.openrouter_api_key,
        "openrouter_base": args.openrouter_base,
        "codex_base_url": args.codex_base_url,
        "codex_auth": args.codex_auth,
    }

    if args.daemon:
        if not port_available(args.host, args.port):
            logger.error(
                "Port %s:%s already in use. Stop the existing adapter or choose another port.",
                args.host,
                args.port,
            )
            sys.exit(1)
        try:
            settings = apply_overrides(load_settings(), overrides)
            settings.model = normalize_model_spec(settings.model) or settings.model
            _ensure_codex_login_if_needed(
                settings,
                no_browser=bool(args.codex_no_browser),
                allow_interactive=sys.stdin.isatty(),
            )
        except Exception as exc:
            logger.error("Codex authentication failed: %s", exc)
            sys.exit(1)
        cmd = [sys.executable, "-m", "cc_adapter.server", "--host", args.host, "--port", str(args.port)]
        if model_arg:
            cmd.extend(["--model", model_arg])
        if args.context_window:
            cmd.extend(["--context-window", str(args.context_window)])
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
        if args.poe_max_retries is not None:
            cmd.extend(["--poe-max-retries", str(args.poe_max_retries)])
        if args.poe_retry_backoff is not None:
            cmd.extend(["--poe-retry-backoff", str(args.poe_retry_backoff)])
        if args.openrouter_api_key:
            cmd.extend(["--openrouter-api-key", args.openrouter_api_key])
        if args.openrouter_base:
            cmd.extend(["--openrouter-base", args.openrouter_base])
        if args.codex_base_url:
            cmd.extend(["--codex-base-url", args.codex_base_url])
        if args.codex_auth:
            cmd.extend(["--codex-auth", args.codex_auth])
        proc = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL, close_fds=True)
        print(f"Started daemon pid={proc.pid}")
        return

    if not port_available(args.host, args.port):
        logger.error(
            "Port %s:%s already in use. Stop the existing adapter or choose another port.",
            args.host,
            args.port,
        )
        sys.exit(1)

    settings = apply_overrides(load_settings(), overrides)
    settings.model = normalize_model_spec(settings.model) or settings.model
    try:
        _ensure_codex_login_if_needed(
            settings,
            no_browser=bool(args.codex_no_browser),
            allow_interactive=sys.stdin.isatty(),
        )
    except Exception as exc:
        logger.error("Codex authentication failed: %s", exc)
        sys.exit(1)
    try:
        run_server(settings)
    except OSError:
        logger.exception("Failed to start server")
        sys.exit(1)


if __name__ == "__main__":
    main()

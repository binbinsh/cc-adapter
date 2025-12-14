import json
import logging
import os
import queue
import threading
import time
import webbrowser
from importlib.resources import as_file, files
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText
from typing import Optional, Tuple

import platformdirs
import requests
from pathlib import Path

from .config import Settings, apply_overrides, default_context_window_for, load_settings
from .codex_oauth import (
    build_authorization_url,
    create_state,
    delete_tokens,
    exchange_authorization_code,
    extract_chatgpt_account_id,
    generate_pkce_pair,
    load_tokens,
    parse_authorization_input,
    save_tokens,
    start_local_callback_server,
    wait_for_callback_code,
)
from .model_registry import DEFAULT_PROVIDER_MODELS, provider_model_slugs
from .server import build_server, port_available
from .logging_utils import resolve_log_level


class LogQueueHandler(logging.Handler):
    """Route log records into a queue so the UI thread can display them safely."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.log_queue.put(message)
        except Exception:
            self.handleError(record)


def _parse_model(default_model: str) -> Tuple[str, str]:
    """Split provider:model into components."""
    if default_model and ":" in default_model:
        provider, name = default_model.split(":", 1)
        return provider, name
    return "", default_model


class AdapterGUI:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.root = tk.Tk()
        self.root.title("CC Adapter GUI")
        self.root.geometry("960x720")
        self._center_window(960, 720)
        self._set_icon()

        provider, model_name = _parse_model(self.settings.model)
        self.provider_display_map = {
            "LM Studio": "lmstudio",
            "OpenAI Codex": "codex",
            "Poe": "poe",
            "OpenRouter": "openrouter",
        }
        self.provider_value_to_display = {v: k for k, v in self.provider_display_map.items()}
        self.host_var = tk.StringVar(value=self.settings.host)
        self.port_var = tk.StringVar(value=str(self.settings.port))
        self.provider_var = tk.StringVar(value=self._provider_display(provider or "lmstudio"))
        self.model_var = tk.StringVar(value=model_name)
        self.context_window_var = tk.StringVar(value="")
        self.log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        default_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if default_log_level not in self.log_level_options:
            default_log_level = "INFO"
        self.log_level_var = tk.StringVar(value=default_log_level)
        self.lmstudio_base_var = tk.StringVar(value=self.settings.lmstudio_base)
        self.lmstudio_timeout_var = tk.StringVar(value=str(self.settings.lmstudio_timeout))
        self.poe_base_var = tk.StringVar(value=self.settings.poe_base_url)
        self.poe_key_var = tk.StringVar(value=self.settings.poe_api_key)
        self.openrouter_base_var = tk.StringVar(value=self.settings.openrouter_base)
        self.openrouter_key_var = tk.StringVar(value=self.settings.openrouter_key)
        self.codex_base_var = tk.StringVar(value=self.settings.codex_base_url)
        self.codex_auth_status_var = tk.StringVar(value=self._codex_status_text())
        self.codex_auth_action_var = tk.StringVar(value=self._codex_action_text())
        self.http_proxy_var = tk.StringVar(value=self.settings.http_proxy)
        self.https_proxy_var = tk.StringVar(value=self.settings.https_proxy)
        self.all_proxy_var = tk.StringVar(value=self.settings.all_proxy)
        self.no_proxy_var = tk.StringVar(value=self.settings.no_proxy)

        self.start_stop_text = tk.StringVar(value="Start")
        self.provider_models = {}
        for provider in self.provider_display_map.values():
            options = []
            if provider == "lmstudio":
                options.append(self.settings.lmstudio_model)
            options.extend(provider_model_slugs(provider))
            if not options and provider in DEFAULT_PROVIDER_MODELS:
                options.append(DEFAULT_PROVIDER_MODELS[provider])
            seen: set[str] = set()
            deduped: list[str] = []
            for name in options:
                if name and name not in seen:
                    deduped.append(name)
                    seen.add(name)
            self.provider_models[provider] = deduped
        self.last_provider = self._current_provider()
        self.server_thread: Optional[threading.Thread] = None
        self.server_instance = None
        self.codex_login_in_progress = False
        self.status_var = tk.StringVar(value="")
        self.max_log_lines = 2000
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.log_handler = LogQueueHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.provider_frames: dict[str, tk.Widget] = {}

        self._load_config_values()
        self.last_context_default = ""
        self._refresh_model_options()
        self._update_context_window_default(force=False)
        self._setup_logging()
        self._build_layout()
        self._format_context_window_var()
        self._set_status("Server stopped")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._present_window)
        self.root.after(200, self._poll_logs)

    def _setup_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(self.log_handler)
        self._apply_log_level()
        logging.getLogger("cc-adapter").propagate = True

    def _center_window(self, width: int, height: int) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _present_window(self) -> None:
        try:
            self.root.update_idletasks()
            current_w = self.root.winfo_width() or 960
            current_h = self.root.winfo_height() or 720
            self._center_window(current_w, current_h)
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(150, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except Exception:
            logging.debug("Could not foreground window", exc_info=True)

    def _set_icon(self) -> None:
        try:
            icon_resource = files(__package__).joinpath("icon.png")
            if not icon_resource.is_file():
                logging.debug("Window icon resource missing: %s", icon_resource)
                return
            with as_file(icon_resource) as icon_path:
                icon_image = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(False, icon_image)
                self._icon_image_ref = icon_image  # prevent GC
        except Exception:
            logging.debug("Could not set window icon", exc_info=True)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(5, weight=1)

        controls = ttk.LabelFrame(container, text="Server settings")
        controls.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        for i in range(6):
            controls.columnconfigure(i, weight=1)

        ttk.Label(controls, text="Host").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.host_var, width=18).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(controls, text="Port").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.port_var, width=10).grid(
            row=0, column=3, sticky="ew", padx=4, pady=4
        )
        ttk.Label(controls, text="Log level").grid(row=0, column=4, sticky="w", padx=4, pady=4)
        log_level_combo = ttk.Combobox(
            controls,
            textvariable=self.log_level_var,
            values=self.log_level_options,
            width=12,
            state="readonly",
        )
        log_level_combo.grid(row=0, column=5, sticky="ew", padx=4, pady=4)
        log_level_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_log_level())

        ttk.Label(controls, text="Provider").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        provider_combo = ttk.Combobox(
            controls,
            textvariable=self.provider_var,
            values=list(self.provider_display_map.keys()),
            width=18,
            state="readonly",
        )
        provider_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        provider_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_provider_change())
        ttk.Label(controls, text="Model name").grid(
            row=1, column=2, sticky="w", padx=4, pady=4
        )
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=self.provider_models.get(self._current_provider(), []),
            width=26,
            state="readonly",
        )
        self.model_combo.grid(row=1, column=3, sticky="ew", padx=4, pady=4)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_model_change())
        ttk.Label(controls, text="Context window").grid(
            row=1, column=4, sticky="w", padx=4, pady=4
        )
        ttk.Entry(controls, textvariable=self.context_window_var, width=18).grid(
            row=1, column=5, sticky="ew", padx=4, pady=4
        )

        self.provider_wrapper = ttk.LabelFrame(container, text="Provider settings")
        self.provider_wrapper.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.provider_wrapper.columnconfigure(0, weight=1)
        self.provider_wrapper.rowconfigure(0, weight=1)
        self._build_provider_frames(self.provider_wrapper)

        proxy_frame = ttk.LabelFrame(container, text="Proxy (optional)")
        proxy_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        for i in range(4):
            proxy_frame.columnconfigure(i, weight=1)
        ttk.Label(proxy_frame, text="HTTP_PROXY").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(proxy_frame, textvariable=self.http_proxy_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(proxy_frame, text="HTTPS_PROXY").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(proxy_frame, textvariable=self.https_proxy_var).grid(
            row=0, column=3, sticky="ew", padx=4, pady=4
        )
        ttk.Label(proxy_frame, text="ALL_PROXY").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(proxy_frame, textvariable=self.all_proxy_var).grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(proxy_frame, text="NO_PROXY").grid(row=1, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(proxy_frame, textvariable=self.no_proxy_var).grid(
            row=1, column=3, sticky="ew", padx=4, pady=4
        )

        controls_bottom = ttk.Frame(container)
        controls_bottom.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        controls_bottom.columnconfigure(0, weight=1)
        controls_bottom.columnconfigure(1, weight=1)
        ttk.Button(controls_bottom, text="Test Provider", command=self.test_provider).grid(
            row=0, column=0, sticky="ew", padx=4
        )
        ttk.Button(controls_bottom, textvariable=self.start_stop_text, command=self.toggle_server).grid(
            row=0, column=1, sticky="ew", padx=4
        )

        log_frame = ttk.LabelFrame(container, text="Logs")
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_widget = ScrolledText(log_frame, height=18, state="disabled")
        self.log_widget.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ttk.Button(log_frame, text="Clear logs", command=self.clear_logs).grid(
            row=1, column=0, sticky="e", padx=4, pady=4
        )
        # Bind after frames exist to ensure visibility toggles run cleanly
        self.provider_var.trace_add("write", lambda *_args: self._on_provider_change())
        self._on_provider_change()

    def _poll_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)
        self.root.after(200, self._poll_logs)

    def _append_log(self, message: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert(tk.END, message + "\n")
        total_lines = int(float(self.log_widget.index("end-1c")))
        if total_lines > self.max_log_lines:
            drop_start = total_lines - self.max_log_lines
            self.log_widget.delete("1.0", f"{drop_start}.0")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state="disabled")

    def clear_logs(self) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.configure(state="disabled")

    def _current_provider(self) -> str:
        display = (self.provider_var.get() or "").strip()
        mapped = self.provider_display_map.get(display)
        if mapped:
            return mapped
        # Fallback: normalize free-typed values
        normalized = display.lower().replace(" ", "")
        return normalized or "lmstudio"

    def _provider_display(self, provider_value: str) -> str:
        return self.provider_value_to_display.get(provider_value.lower(), provider_value)

    def _config_path(self) -> Path:
        override = os.getenv("CC_ADAPTER_CONFIG_DIR", "").strip()
        if override:
            return Path(override) / "gui.json"
        return Path(platformdirs.user_config_dir("cc-adapter")) / "gui.json"

    def _codex_status_text(self) -> str:
        tokens = load_tokens()
        if not tokens:
            return "Not logged in"
        now_ms = int(time.time() * 1000)
        remaining_s = max(0, (tokens.expires_at_ms - now_ms) // 1000)
        mins = remaining_s // 60
        account_id = extract_chatgpt_account_id(tokens.access)
        suffix = f", {mins}m remaining" if remaining_s else ", expired"
        if account_id:
            return f"Logged in{suffix}"
        return f"Logged in{suffix}"

    def _codex_action_text(self) -> str:
        if getattr(self, "codex_login_in_progress", False):
            return "Logging in..."
        return "Logout" if load_tokens() else "Login"

    def _refresh_codex_status(self) -> None:
        self.codex_auth_status_var.set(self._codex_status_text())
        if hasattr(self, "codex_auth_action_var"):
            self.codex_auth_action_var.set(self._codex_action_text())
        if hasattr(self, "codex_auth_button"):
            state = "disabled" if getattr(self, "codex_login_in_progress", False) else "normal"
            try:
                self.codex_auth_button.configure(state=state)
            except Exception:
                pass

    def _load_config_values(self) -> None:
        path = self._config_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            logging.warning("Failed to load config file %s: %s", path, exc)
            return

        def set_if(key: str, var: tk.StringVar):
            if key in data and data[key] is not None:
                var.set(str(data[key]))

        provider_val = data.get("provider")
        if provider_val:
            self.provider_var.set(self._provider_display(str(provider_val)))

        set_if("model", self.model_var)
        set_if("host", self.host_var)
        set_if("port", self.port_var)
        set_if("log_level", self.log_level_var)
        set_if("lmstudio_base", self.lmstudio_base_var)
        set_if("lmstudio_timeout", self.lmstudio_timeout_var)
        set_if("poe_base_url", self.poe_base_var)
        set_if("poe_api_key", self.poe_key_var)
        set_if("openrouter_base", self.openrouter_base_var)
        set_if("openrouter_api_key", self.openrouter_key_var)
        set_if("codex_base_url", self.codex_base_var)
        set_if("http_proxy", self.http_proxy_var)
        set_if("https_proxy", self.https_proxy_var)
        set_if("all_proxy", self.all_proxy_var)
        set_if("no_proxy", self.no_proxy_var)
        set_if("context_window", self.context_window_var)

    def _auto_save(self) -> None:
        try:
            self._save_config_file(silent=True)
        except Exception:
            # Ignore autosave failures; keep UI responsive.
            logging.debug("Autosave skipped due to an error", exc_info=True)

    def _on_provider_change(self) -> None:
        provider = self._current_provider()
        prev_provider = getattr(self, "last_provider", provider)
        display = self._provider_display(provider)
        if self.provider_var.get() != display:
            self.provider_var.set(display)
        if provider == "codex":
            self._refresh_codex_status()
        self._refresh_model_options()
        self._update_context_window_default(force=True)
        self._format_context_window_var()
        self._update_provider_visibility()
        self.last_provider = provider
        self._auto_save()
        self._restart_if_running()

    def _on_model_change(self) -> None:
        self._update_context_window_default(force=True)
        self._format_context_window_var()
        self._auto_save()
        self._restart_if_running()

    def _restart_if_running(self) -> None:
        """Restart the server if it's currently running to apply new settings."""
        if self.server_thread and self.server_thread.is_alive():
            self.stop_server()
            self.start_server()

    def _apply_log_level(self) -> None:
        level_name = self.log_level_var.get().upper()
        level = resolve_log_level(level_name)
        os.environ["LOG_LEVEL"] = level_name
        logging.getLogger().setLevel(level)
        logging.getLogger("cc-adapter").setLevel(level)
        self.log_handler.setLevel(level)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        try:
            self.root.title(f"CC Adapter GUI ({text})")
        except Exception:
            self.root.title("CC Adapter GUI")

    def _format_context_window_var(self) -> None:
        raw = (self.context_window_var.get() or "").replace(",", "").strip()
        if not raw:
            return
        try:
            num = int(raw)
            self.context_window_var.set(f"{num:,}")
        except ValueError:
            # leave as-is on invalid input
            pass

    def _build_settings(self) -> Settings:
        settings = load_settings()
        try:
            port = int(self.port_var.get())
        except ValueError as exc:
            raise ValueError("Port must be an integer") from exc

        try:
            timeout = float(self.lmstudio_timeout_var.get())
        except ValueError as exc:
            raise ValueError("LM Studio timeout must be a number") from exc

        model = self.model_var.get().strip()
        provider = self._current_provider()
        composed_model = ""
        if model:
            composed_model = f"{provider}:{model}" if provider else model

        overrides = {
            "host": self.host_var.get().strip() or settings.host,
            "port": port,
            "model": composed_model,
            "context_window": self._safe_int(self.context_window_var.get().replace(",", "").strip(), default=None),
            "lmstudio_base": self.lmstudio_base_var.get().strip() or settings.lmstudio_base,
            "lmstudio_timeout": timeout,
            "poe_api_key": self.poe_key_var.get().strip(),
            "poe_base_url": self.poe_base_var.get().strip() or settings.poe_base_url,
            "openrouter_key": self.openrouter_key_var.get().strip(),
            "openrouter_base": self.openrouter_base_var.get().strip() or settings.openrouter_base,
            "codex_base_url": self.codex_base_var.get().strip() or settings.codex_base_url,
            "http_proxy": self.http_proxy_var.get().strip(),
            "https_proxy": self.https_proxy_var.get().strip(),
            "all_proxy": self.all_proxy_var.get().strip(),
            "no_proxy": self.no_proxy_var.get().strip(),
        }
        settings = apply_overrides(settings, overrides)
        if provider == "lmstudio" and model:
            # Keep LM Studio default model aligned with the selected model field.
            settings.lmstudio_model = model
        return settings

    def _build_provider_frames(self, wrapper: ttk.LabelFrame) -> None:
        stack = ttk.Frame(wrapper)
        stack.grid(row=0, column=0, sticky="nsew")
        stack.columnconfigure(0, weight=1)
        stack.rowconfigure(0, weight=1)

        lm_frame = ttk.Frame(stack)
        lm_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            lm_frame.columnconfigure(i, weight=1)
        ttk.Label(lm_frame, text="Base URL").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(lm_frame, textvariable=self.lmstudio_base_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(lm_frame, text="Timeout (s)").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(lm_frame, textvariable=self.lmstudio_timeout_var).grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        self.provider_frames["lmstudio"] = lm_frame

        poe_frame = ttk.Frame(stack)
        poe_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            poe_frame.columnconfigure(i, weight=1)
        ttk.Label(poe_frame, text="Base URL").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(poe_frame, textvariable=self.poe_base_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(poe_frame, text="API key").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(poe_frame, textvariable=self.poe_key_var, show="*").grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        self.provider_frames["poe"] = poe_frame

        openrouter_frame = ttk.Frame(stack)
        openrouter_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            openrouter_frame.columnconfigure(i, weight=1)
        ttk.Label(openrouter_frame, text="Base URL").grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        ttk.Entry(openrouter_frame, textvariable=self.openrouter_base_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(openrouter_frame, text="API key").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        ttk.Entry(openrouter_frame, textvariable=self.openrouter_key_var, show="*").grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        self.provider_frames["openrouter"] = openrouter_frame

        codex_frame = ttk.Frame(stack)
        codex_frame.grid(row=0, column=0, sticky="nsew")
        for i in range(2):
            codex_frame.columnconfigure(i, weight=1)
        ttk.Label(codex_frame, text="Base URL").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(codex_frame, textvariable=self.codex_base_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Label(codex_frame, text="OAuth status").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        status_row = ttk.Frame(codex_frame)
        status_row.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        status_row.columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=self.codex_auth_status_var).grid(row=0, column=0, sticky="w")
        self.codex_auth_button = ttk.Button(
            status_row,
            textvariable=self.codex_auth_action_var,
            command=self._toggle_codex_auth,
        )
        self.codex_auth_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.provider_frames["codex"] = codex_frame

        for frame in self.provider_frames.values():
            frame.grid_remove()
        self._update_provider_visibility()

    def _codex_logout(self) -> None:
        delete_tokens()
        self._refresh_codex_status()
        messagebox.showinfo("Codex OAuth", "Logged out (local tokens deleted).")

    def _toggle_codex_auth(self) -> None:
        if self.codex_login_in_progress:
            return
        if load_tokens() is None:
            self._codex_login()
        else:
            self._codex_logout()

    def _codex_login(self, on_success=None) -> None:
        if self.codex_login_in_progress:
            return
        self.codex_login_in_progress = True
        self._refresh_codex_status()

        def finish(status: str, message: str) -> None:
            self.codex_login_in_progress = False
            self._refresh_codex_status()
            if status == "ok":
                messagebox.showinfo("Codex OAuth", message)
                if on_success:
                    try:
                        on_success()
                    except Exception:
                        logging.exception("Codex post-login callback failed")
            else:
                messagebox.showerror("Codex OAuth", message)

        def worker() -> None:
            try:
                verifier, challenge = generate_pkce_pair()
                state = create_state()
                url = build_authorization_url(state, challenge)

                server = None
                code = None
                try:
                    server = start_local_callback_server(state)
                except OSError:
                    server = None

                webbrowser.open(url, new=1, autoraise=True)

                if server:
                    code = wait_for_callback_code(server, timeout_seconds=300)
                    try:
                        server.shutdown()
                    except Exception:
                        pass
                    try:
                        server.server_close()
                    except Exception:
                        pass

                if not code:
                    result: dict = {"raw": None}
                    done = threading.Event()

                    def ask() -> None:
                        result["raw"] = simpledialog.askstring(
                            "Codex OAuth",
                            "Paste the full callback URL or authorization code (code#state supported):",
                        )
                        done.set()

                    self.root.after(0, ask)
                    done.wait(timeout=600)
                    raw = (result.get("raw") or "").strip()
                    code, returned_state = parse_authorization_input(raw)
                    if not code:
                        raise RuntimeError("No authorization code provided.")
                    if returned_state and returned_state != state:
                        raise RuntimeError("State mismatch.")

                tokens = exchange_authorization_code(code=code, code_verifier=verifier)
                save_tokens(tokens)
                self.root.after(0, lambda: finish("ok", "Login successful."))
            except Exception as exc:
                msg = f"Login failed: {exc}"
                self.root.after(0, lambda m=msg: finish("err", m))

        threading.Thread(target=worker, daemon=True).start()

    def _update_provider_visibility(self) -> None:
        selected = self._current_provider()
        if selected not in self.provider_frames:
            selected = "lmstudio"
            self.provider_var.set(selected)
        display = self._provider_display(selected)
        self.provider_wrapper.configure(text=f"{display} settings")
        for name, frame in self.provider_frames.items():
            if name == selected:
                frame.grid()
            else:
                frame.grid_remove()

    def _refresh_model_options(self) -> None:
        provider = self._current_provider()
        options = self.provider_models.get(provider, [])
        if hasattr(self, "model_combo"):
            self.model_combo["values"] = options
        current = self.model_var.get().strip()
        if options:
            if current not in options:
                self.model_var.set(options[0])
                current = options[0]
            if hasattr(self, "model_combo"):
                self.model_combo.set(current)
        else:
            if hasattr(self, "model_combo"):
                self.model_combo.set(current)

    def _resolved_context_default(self) -> str:
        model = self.model_var.get().strip()
        provider = self._current_provider()
        target = model
        if provider and model and ":" not in model:
            target = f"{provider}:{model}"
        default = default_context_window_for(target)
        return str(default) if default > 0 else ""

    def _update_context_window_default(self, force: bool) -> None:
        default = self._resolved_context_default()
        if not default:
            if force:
                self.context_window_var.set("")
            return
        formatted = f"{int(default):,}"
        current_raw = (self.context_window_var.get() or "").replace(",", "").strip()
        if force or not current_raw:
            self.context_window_var.set(formatted)
        self.last_context_default = formatted

    def _save_config_file(self, silent: bool = False) -> None:
        data = {
            "host": self.host_var.get().strip(),
            "port": self._safe_int(self.port_var.get().strip(), default=None),
            "provider": self._current_provider(),
            "model": self.model_var.get().strip(),
            "context_window": self._safe_int(self.context_window_var.get().strip(), default=None),
            "log_level": self.log_level_var.get().strip(),
            "lmstudio_base": self.lmstudio_base_var.get().strip(),
            "lmstudio_timeout": self._safe_float(self.lmstudio_timeout_var.get().strip(), default=None),
            "poe_base_url": self.poe_base_var.get().strip(),
            "poe_api_key": self.poe_key_var.get(),
            "openrouter_base": self.openrouter_base_var.get().strip(),
            "openrouter_api_key": self.openrouter_key_var.get(),
            "codex_base_url": self.codex_base_var.get().strip(),
            "http_proxy": self.http_proxy_var.get().strip(),
            "https_proxy": self.https_proxy_var.get().strip(),
            "all_proxy": self.all_proxy_var.get().strip(),
            "no_proxy": self.no_proxy_var.get().strip(),
        }
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logging.info("Saved GUI config to %s", path)
        if not silent:
            messagebox.showinfo("Saved", f"Saved config to {path}")

    def _safe_int(self, value: str, default: Optional[int]) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return default

    def _safe_float(self, value: str, default: Optional[float]) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return default

    def _control_host(self, host: str) -> str:
        value = (host or "").strip()
        if not value or value == "0.0.0.0":
            return "127.0.0.1"
        return value

    def _try_shutdown_existing_adapter(self, host: str, port: int) -> bool:
        control_host = self._control_host(host)
        base_url = f"http://{control_host}:{port}"
        try:
            resp = requests.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code != 200:
                return False
            data = resp.json() if resp.content else {}
            if not isinstance(data, dict) or data.get("status") != "ok":
                return False
        except Exception:
            return False

        try:
            requests.post(f"{base_url}/shutdown", timeout=1.0)
        except Exception:
            return False

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if port_available(host, port):
                return True
            time.sleep(0.1)
        return port_available(host, port)

    def start_server(self) -> None:
        if self.server_thread and self.server_thread.is_alive():
            return
        try:
            settings = self._build_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        provider = self._current_provider()
        if provider == "codex":
            env_present = bool(
                settings.codex_access_token
                and settings.codex_refresh_token
                and settings.codex_expires_at_ms > 0
            )
            file_present = load_tokens() is not None
            if not (env_present or file_present):
                self._set_status("Codex login required")
                self._codex_login(on_success=self.start_server)
                return

        if not port_available(settings.host, settings.port):
            if self._try_shutdown_existing_adapter(settings.host, settings.port):
                logging.info(
                    "Stopped existing adapter on %s:%s to free the port.",
                    settings.host,
                    settings.port,
                )
            else:
                logging.error(
                    "Port %s:%s is already in use. Stop the existing adapter or choose another port.",
                    settings.host,
                    settings.port,
                )
                self._set_status("Port in use")
                self.start_stop_text.set("Start")
                messagebox.showerror(
                    "Port in use",
                    f"Port {settings.host}:{settings.port} is already in use. Stop the existing adapter or choose another port.",
                )
                return

        if not port_available(settings.host, settings.port):
            logging.error(
                "Port %s:%s is already in use. Stop the existing adapter or choose another port.",
                settings.host,
                settings.port,
            )
            self._set_status("Port in use")
            self.start_stop_text.set("Start")
            messagebox.showerror(
                "Port in use",
                f"Port {settings.host}:{settings.port} is already in use. Stop the existing adapter or choose another port.",
            )
            return

        try:
            server = build_server(settings)
        except OSError as exc:
            logging.exception("Failed to start server")
            messagebox.showerror("Port in use", f"Failed to start server: {exc}")
            self._set_status("Failed to start")
            self.start_stop_text.set("Start")
            return

        self._apply_log_level()
        self.server_instance = server
        self.server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.server_thread.start()
        self._set_status(f"Running on {settings.host}:{settings.port}")
        self.start_stop_text.set("Stop")
        self._auto_save()
        logging.info(
            "Adapter started from GUI on http://%s:%s (model=%s)",
            settings.host,
            settings.port,
            settings.model or "client-provided",
        )

    def stop_server(self) -> None:
        if not self.server_instance:
            return
        self.server_instance.shutdown()
        self.server_instance.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=3)
        self.server_instance = None
        self.server_thread = None
        self._set_status("Server stopped")
        self.start_stop_text.set("Start")
        self._auto_save()
        logging.info("Adapter stopped from GUI")

    def toggle_server(self) -> None:
        if self.server_thread and self.server_thread.is_alive():
            self.stop_server()
        else:
            self.start_server()

    def test_provider(self) -> None:
        provider = self._current_provider()
        model = self.model_var.get().strip()
        if not provider:
            messagebox.showerror("Missing provider", "Select a provider to test.")
            return
        if not model:
            messagebox.showerror("Missing model", "Enter the model name to test.")
            return
        try:
            settings = self._build_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        start = time.monotonic()
        try:
            if provider == "lmstudio":
                detail = self._test_lmstudio(settings, model)
            elif provider == "poe":
                detail = self._test_poe(settings, model)
            elif provider == "openrouter":
                detail = self._test_openrouter(settings, model)
            elif provider == "codex":
                detail = self._test_codex(settings, model)
            else:
                messagebox.showerror("Unsupported provider", self._provider_display(provider))
                return
        except Exception as exc:
            logging.exception("Provider test failed")
            messagebox.showerror("Test failed", f"{provider} check failed: {exc}")
            return

        elapsed_ms = int((time.monotonic() - start) * 1000)
        status_text = detail.split(";")[0] if detail else "ok"
        display_name = self._provider_display(provider)
        messagebox.showinfo("Connection ok", f"{display_name} connected successfully ({status_text}, {elapsed_ms} ms)")
        self._auto_save()
        logging.info("Provider %s test ok: %s (%sms)", provider, status_text, elapsed_ms)

    def _test_lmstudio(self, settings: Settings, model: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 32,
        }
        resp = requests.post(
            settings.lmstudio_base,
            json=payload,
            timeout=settings.lmstudio_timeout,
            proxies=settings.resolved_proxies(),
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0] if isinstance(data.get("choices"), list) else {}
        finish_reason = choice.get("finish_reason") or choice.get("finish_details", {}).get("type")
        return f"HTTP {resp.status_code}"

    def _test_poe(self, settings: Settings, model: str) -> str:
        if not settings.poe_api_key:
            raise ValueError("POE_API_KEY is required for Poe")
        model_name = model.split(":", 1)[1] if model.lower().startswith("poe:") else model
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 32,
        }
        resp = requests.post(
            settings.poe_base_url,
            json=payload,
            headers={"Authorization": f"Bearer {settings.poe_api_key}"},
            timeout=settings.lmstudio_timeout,
            proxies=settings.resolved_proxies(),
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            snippet = (resp.text or "").strip()
            raise requests.HTTPError(f"{exc} | body_snippet={snippet[:400]}") from exc
        resp.json()
        return f"HTTP {resp.status_code}"

    def _test_openrouter(self, settings: Settings, model: str) -> str:
        if not settings.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter")
        target_model = model
        if "/" not in target_model and target_model.lower().startswith("claude"):
            target_model = f"anthropic/{target_model}"
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 32,
        }
        resp = requests.post(
            settings.openrouter_base,
            json=payload,
            headers={"Authorization": f"Bearer {settings.openrouter_key}"},
            timeout=settings.lmstudio_timeout,
            proxies=settings.resolved_proxies(),
        )
        resp.raise_for_status()
        data = resp.json()
        return f"HTTP {resp.status_code}"

    def _test_codex(self, settings: Settings, model: str) -> str:
        from .model_registry import canonicalize_model
        from .providers import codex as codex_provider

        model_name = model.split(":", 1)[1] if model.lower().startswith("codex:") else model
        target_model = canonicalize_model("codex", model_name)

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
            "max_tokens": 16,
        }

        tokens, account_id = codex_provider._resolve_codex_auth(settings)
        model_key = codex_provider._codex_model_key(settings, target_model)
        body = codex_provider._request_body(payload, settings, model_key=model_key)

        resp = requests.post(
            settings.codex_base_url,
            json=body,
            headers=codex_provider._headers(account_id, tokens.access),
            timeout=min(60.0, float(settings.lmstudio_timeout)),
            proxies=settings.resolved_proxies(),
            stream=True,
        )
        try:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                if not line.startswith(b"data:"):
                    continue
                return f"HTTP {resp.status_code}"
            return f"HTTP {resp.status_code}"
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def on_close(self) -> None:
        self._save_config_file(silent=True)
        if self.server_instance:
            self.stop_server()
        self.root.destroy()


def main() -> None:
    app = AdapterGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()

import argparse
import sys
import webbrowser

from .codex_oauth import (
    build_authorization_url,
    create_state,
    delete_tokens,
    exchange_authorization_code,
    extract_chatgpt_account_id,
    generate_pkce_pair,
    parse_authorization_input,
    save_tokens,
    start_local_callback_server,
    wait_for_callback_code,
)


def _prompt_manual_code(expected_state: str) -> str:
    print("Paste the full callback URL or the authorization code (code#state also supported):")
    raw = sys.stdin.readline().strip()
    code, state = parse_authorization_input(raw)
    if not code:
        raise SystemExit("No authorization code provided")
    if state and state != expected_state:
        raise SystemExit("State mismatch")
    return code


def login(no_browser: bool = False) -> int:
    verifier, challenge = generate_pkce_pair()
    state = create_state()
    url = build_authorization_url(state, challenge)

    print("OpenAI Codex OAuth login (ChatGPT subscription)")
    print(f"Auth URL:\n{url}\n")

    server = None
    code = None
    try:
        server = start_local_callback_server(state)
        if not no_browser:
            webbrowser.open(url, new=1, autoraise=True)
        code = wait_for_callback_code(server, timeout_seconds=300)
    except OSError:
        server = None
    finally:
        if server:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass

    if not code:
        if not no_browser:
            webbrowser.open(url, new=1, autoraise=True)
        code = _prompt_manual_code(state)

    tokens = exchange_authorization_code(code=code, code_verifier=verifier)
    token_path = save_tokens(tokens)
    account_id = extract_chatgpt_account_id(tokens.access)

    print(f"Saved OAuth tokens to: {token_path}")
    if account_id:
        print(f"ChatGPT account id: {account_id}")
    else:
        print("Warning: failed to extract ChatGPT account id from access token.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Authenticate cc-adapter for OpenAI Codex (ChatGPT OAuth)")
    parser.add_argument("--no-browser", action="store_true", help="Do not attempt to open a browser automatically")
    parser.add_argument("--logout", action="store_true", help="Delete the stored OAuth token file")
    args = parser.parse_args()

    if args.logout:
        delete_tokens()
        print("Deleted stored OpenAI Codex OAuth tokens.")
        raise SystemExit(0)

    raise SystemExit(login(no_browser=bool(args.no_browser)))


if __name__ == "__main__":
    main()

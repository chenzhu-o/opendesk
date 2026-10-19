"""Streamable HTTP transport for the OpenDesk MCP server.

Run with::

    opendesk-mcp --transport http --host 127.0.0.1 --port 8765
    opendesk mcp --http --port 8765

Clients connect to ``http://<host>:<port>/mcp`` and must send
``Authorization: Bearer <token>``.  The token comes from ``--token``, else
``$OPENDESK_MCP_TOKEN``, else it is generated once and stored (mode 0600) in
``<home>/mcp-token`` and printed on start.  Binding to a non-loopback
address without auth is refused: the server drives a real desktop.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

TOKEN_FILENAME = "mcp-token"
MCP_PATH = "/mcp"

_Scope = dict[str, Any]
_Receive = Callable[[], Awaitable[dict[str, Any]]]
_Send = Callable[[dict[str, Any]], Awaitable[None]]
_ASGIApp = Callable[[_Scope, _Receive, _Send], Awaitable[None]]


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def _default_home() -> Path:
    from opendesk.protocol.auth.identity import DEFAULT_HOME
    return DEFAULT_HOME


def token_path(home: Optional[Path] = None) -> Path:
    return (home or _default_home()) / TOKEN_FILENAME


def load_or_create_token(home: Optional[Path] = None) -> tuple[str, bool]:
    """Return ``(token, created)`` for the stored token, creating it if absent.

    The file is created with mode 0600 so other local users can't read it.
    """
    path = token_path(home)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token, False
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        os.write(fd, (token + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - e.g. Windows
        pass
    return token, True


def resolve_token(
    token: Optional[str], home: Optional[Path] = None,
) -> tuple[str, str]:
    """Pick the bearer token: explicit → ``$OPENDESK_MCP_TOKEN`` → stored file.

    Returns ``(token, source)`` where *source* describes where it came from.
    """
    if token:
        return token, "--token"
    env = os.environ.get("OPENDESK_MCP_TOKEN", "").strip()
    if env:
        return env, "$OPENDESK_MCP_TOKEN"
    value, created = load_or_create_token(home)
    path = token_path(home)
    return value, (f"generated and saved to {path}" if created else f"read from {path}")


def is_loopback(host: str) -> bool:
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# ASGI auth middleware
# ---------------------------------------------------------------------------


class BearerAuthMiddleware:
    """Reject HTTP requests whose ``Authorization`` isn't ``Bearer <token>``.

    Non-HTTP scopes (``lifespan``) pass straight through.
    """

    def __init__(self, app: _ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("BearerAuthMiddleware needs a non-empty token")
        self._app = app
        self._token = token.encode("utf-8")

    async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        if self._authorized(scope):
            await self._app(scope, receive, send)
            return
        body = json.dumps({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32001, "message": "Unauthorized: missing or invalid bearer token"},
        }).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="opendesk"'),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    def _authorized(self, scope: _Scope) -> bool:
        for key, value in scope.get("headers") or []:
            if key.lower() == b"authorization":
                scheme, _, presented = value.partition(b" ")
                if scheme.lower() != b"bearer":
                    return False
                return hmac.compare_digest(presented.strip(), self._token)
        return False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_http_app(
    server: Any,
    *,
    token: Optional[str],
    host: str = "127.0.0.1",
    port: int = 8765,
    json_response: bool = False,
    on_shutdown: Optional[Callable[[], Awaitable[None]]] = None,
) -> Any:
    """Wrap an MCP low-level ``Server`` in a Starlette app serving ``/mcp``.

    *token* ``None`` disables auth (only sensible on loopback).
    *on_shutdown* runs when the app's lifespan ends (e.g. closing peers).
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Route

    if is_loopback(host):
        # DNS-rebinding protection: only accept loopback Host / Origin headers.
        hosts = ["127.0.0.1", "localhost", "[::1]"]
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[h for base in hosts for h in (base, f"{base}:*")],
            allowed_origins=[
                o for base in hosts for o in (f"http://{base}", f"http://{base}:*")
            ],
        )
    else:
        # Reachable under arbitrary Host names; the bearer token is the guard.
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    manager = StreamableHTTPSessionManager(
        app=server, json_response=json_response, security_settings=security,
    )

    class _MCPEndpoint:
        # A class instance (not a function) makes Starlette's Route treat it
        # as a raw ASGI app, so /mcp is served without a /mcp/ redirect.
        async def __call__(self, scope: _Scope, receive: _Receive, send: _Send) -> None:
            await manager.handle_request(scope, receive, send)

    endpoint = _MCPEndpoint()

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            async with manager.run():
                yield
        finally:
            if on_shutdown is not None:
                await on_shutdown()

    routes = [
        Route(MCP_PATH, endpoint=endpoint, methods=["GET", "POST", "DELETE"]),
        Route(MCP_PATH + "/", endpoint=endpoint, methods=["GET", "POST", "DELETE"]),
    ]
    app: Any = Starlette(routes=routes, lifespan=lifespan)
    if token:
        app = BearerAuthMiddleware(app, token)
    return app


def run_http_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: Optional[str] = None,
    no_auth: bool = False,
    home: Optional[Path] = None,
) -> None:
    """Serve the OpenDesk MCP server over streamable HTTP (blocking)."""
    import uvicorn

    from opendesk.integrations.mcp import create_mcp_server
    from opendesk.integrations.mcp_session import MCPSession
    from opendesk.registry import create_registry

    loopback = is_loopback(host)
    if no_auth and not loopback:
        raise ValueError(
            f"refusing to serve on non-loopback address {host!r} without a token; "
            "drop --no-auth (a token is generated for you) or bind to 127.0.0.1"
        )
    if no_auth and token:
        raise ValueError("--token and --no-auth are mutually exclusive")

    bearer: Optional[str] = None
    source = "disabled (--no-auth)"
    if not no_auth:
        bearer, source = resolve_token(token, home)

    session = MCPSession(home=home)
    server = create_mcp_server(create_registry(), session)
    app = build_http_app(
        server, token=bearer, host=host, port=port, on_shutdown=session.aclose,
    )

    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    print(f"OpenDesk MCP (streamable HTTP) on http://{url_host}:{port}{MCP_PATH}", file=sys.stderr)
    if bearer:
        print(f"  Auth: Authorization: Bearer {bearer}", file=sys.stderr)
        print(f"  Token {source}", file=sys.stderr)
    else:
        print(f"  Auth: {source}", file=sys.stderr)
    if not loopback:
        print("  WARNING: listening beyond loopback; anyone with the token controls this desktop.",
              file=sys.stderr)

    uvicorn.run(app, host=host, port=port, log_level="warning")

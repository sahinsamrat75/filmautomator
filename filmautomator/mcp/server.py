"""Filmautomator MCP server.

The canonical interface through which an external AI operates the production
engine. Any MCP-compatible client — Claude Code, or anything else that speaks
the protocol — connects here and gets filmmaking capabilities rather than
Blender internals.

Transports:
  * **stdio** (default) — what local clients like Claude Code use.
  * **HTTP** (optional) — a small JSON-RPC endpoint on localhost for other
    clients. It binds loopback only and is never exposed beyond the machine.

The server is a thin dispatcher: it validates the protocol envelope and hands
the work to the tool registry. All the filmmaking lives in ``toolset``.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO

from .. import __version__
from ..config import AppConfig, load_config
from . import toolset  # noqa: F401 - registers the tools on import
from .protocol import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    SERVER_NAME,
    SUPPORTED_PROTOCOL_VERSIONS,
    MCPError,
    Request,
    encode_message,
    error_response,
    json_text,
    negotiate_protocol_version,
    result_response,
    tool_result,
)
from .tools import ToolContext, get_tool, list_tools

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
Filmautomator is an autonomous film studio running on this machine. You give it
a creative objective in plain language and it plans shots, builds them in
Blender, renders previews, reviews them with a Vision Agent, corrects what looks
wrong, renders at final quality, and assembles the result with FFmpeg.

Typical flow:
  1. create_project(name, objective)         -> project_id
  2. start_production(project_id, duration_s) -> returns immediately
  3. get_production_status()                  -> poll until state is COMPLETED
  4. get_preview(project_id, shot_id)         -> look at what it rendered
  5. get_final_movie(project_id)              -> the deliverable and its path

Production runs in the background and can take minutes, so do not expect
start_production to block. Use get_agent_activity to follow what the agents are
doing, and get_preview to see the same frames the Vision Agent judged.

There is no tool for running shell commands or arbitrary code, by design. If you
need the scene changed, describe the change as an objective and let the Director
translate it.
"""


class MCPServer:
    """Protocol handling and dispatch, independent of transport."""

    def __init__(self, context: ToolContext | None = None,
                 config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.context = context or ToolContext.create(self.config)
        self._initialised = False

    # -- dispatch ----------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one message. Returns None for notifications."""
        try:
            request = Request.parse(message)
        except MCPError as exc:
            return exc.to_dict(message.get("id") if isinstance(message, dict) else None)

        try:
            result = self._dispatch(request)
        except MCPError as exc:
            if request.is_notification:
                log.warning("notification %s failed: %s", request.method, exc.message)
                return None
            return exc.to_dict(request.request_id)
        except Exception as exc:  # noqa: BLE001 - a tool bug is not a crash
            log.error("method %s raised", request.method, exc_info=True)
            if request.is_notification:
                return None
            return error_response(request.request_id, INTERNAL_ERROR,
                                  f"{type(exc).__name__}: {exc}")

        if request.is_notification:
            return None
        return result_response(request.request_id, result)

    def _dispatch(self, request: Request) -> dict[str, Any]:
        method = request.method

        if method == "initialize":
            return self._initialize(request.params)
        if method in ("notifications/initialized", "initialized"):
            self._initialised = True
            return {}
        if method == "notifications/cancelled":
            return {}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": list_tools()}
        if method == "tools/call":
            return self._call_tool(request.params)

        raise MCPError(METHOD_NOT_FOUND, f"unknown method: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = negotiate_protocol_version(
            requested if isinstance(requested, str) else None
        )
        client = params.get("clientInfo") or {}
        log.info("MCP client connected: %s %s (protocol %s)",
                 client.get("name", "unknown"), client.get("version", ""), version)
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": SERVER_NAME,
                "title": "FilmAutomator",
                "version": __version__,
            },
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise MCPError(INVALID_REQUEST, "tools/call requires a tool name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise MCPError(INVALID_REQUEST, "arguments must be an object")

        try:
            found = get_tool(name)
        except MCPError:
            # An unknown tool is reported as a failed tool call rather than a
            # transport error, and the reply names the tools that do exist.
            # A model that mistyped a name can then correct itself instead of
            # the client treating the whole turn as broken.
            return tool_result(
                [json_text({
                    "error": f"unknown tool: {name}",
                    "available_tools": sorted(t["name"] for t in list_tools()),
                })],
                is_error=True,
            )

        log.info("tool call: %s(%s)", name, ", ".join(sorted(arguments)) or "")
        try:
            return found.handler(self.context, arguments)
        except MCPError as exc:
            return tool_result([json_text({"error": exc.message})], is_error=True)
        except Exception as exc:  # noqa: BLE001 - report, never drop the connection
            log.error("tool %s raised", name, exc_info=True)
            return tool_result(
                [json_text({"error": f"{type(exc).__name__}: {exc}"})],
                is_error=True,
            )

    def close(self) -> None:
        self.context.close()


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def serve_stdio(server: MCPServer | None = None, *,
                stdin: TextIO | None = None,
                stdout: TextIO | None = None) -> int:
    """Serve MCP over stdin/stdout.

    Diagnostics go to stderr. stdout carries protocol messages only — a stray
    print here would corrupt the stream and the client would drop the
    connection.
    """
    server = server or MCPServer()
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    try:
        for line in stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError as exc:
                stdout.write(
                    encode_message(
                        error_response(None, -32700, f"invalid JSON: {exc}")
                    ).decode("utf-8")
                )
                stdout.flush()
                continue

            response = server.handle(message)
            if response is not None:
                stdout.write(encode_message(response).decode("utf-8"))
                stdout.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        server.close()
    return 0


# ---------------------------------------------------------------------------
# HTTP transport (localhost only)
# ---------------------------------------------------------------------------


def make_http_handler(server: MCPServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"Filmautomator/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            log.debug("mcp-http: " + fmt, *args)

        def _send(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/"):
                self._send({"status": "ok", "server": SERVER_NAME,
                            "version": __version__,
                            "protocol": LATEST_PROTOCOL_VERSION,
                            "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
                            "tools": len(list_tools())})
            else:
                self._send({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                message = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send(error_response(None, -32700, f"invalid JSON: {exc}"))
                return

            if isinstance(message, list):
                responses = [r for r in (server.handle(m) for m in message)
                             if r is not None]
                self._send(responses if responses else {"jsonrpc": "2.0"})
                return

            response = server.handle(message)
            if response is None:
                self._send({"jsonrpc": "2.0", "id": None, "result": {}})
            else:
                self._send(response)

    return Handler


def serve_http(server: MCPServer | None = None, host: str = "127.0.0.1",
               port: int = 8766) -> ThreadingHTTPServer:
    """Start the localhost HTTP transport and return the running server."""
    server = server or MCPServer()
    httpd = ThreadingHTTPServer((host, port), make_http_handler(server))
    thread = threading.Thread(target=httpd.serve_forever, name="mcp-http",
                              daemon=True)
    thread.start()
    log.info("MCP HTTP transport on http://%s:%d", host, httpd.server_address[1])
    return httpd

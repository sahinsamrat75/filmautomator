"""MCP protocol primitives.

Model Context Protocol is JSON-RPC 2.0 with a defined handshake. This module
holds the wire-level pieces — message framing, error codes, protocol version
negotiation — so the server and the tools can stay concerned with filmmaking
rather than with transport.

Implemented directly rather than via the official SDK on purpose: the SDK pulls
in around thirty transitive dependencies (pydantic, cryptography, starlette,
uvicorn, httpx), and this project's whole deployment story is that it runs on
the standard library plus whatever Blender already ships. The stdio transport is
newline-delimited JSON, and the surface a tools server needs — initialize,
tools/list, tools/call — is small enough to implement exactly and test
directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

JSONRPC_VERSION = "2.0"

#: Protocol revisions this server understands, newest first.
#: 2025-06-18 is current; the older two are accepted so a client that has not
#: been updated still connects.
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

SERVER_NAME = "filmautomator"
SERVER_TITLE = "FilmAutomator"

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def negotiate_protocol_version(requested: str | None) -> str:
    """Pick the protocol version to answer with.

    Per the spec: echo the client's version when it is one we support,
    otherwise offer our latest and let the client decide whether to continue.
    """
    if requested and requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    return LATEST_PROTOCOL_VERSION


class MCPError(Exception):
    """A JSON-RPC error to return to the client."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self, request_id: Any) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


@dataclass
class Request:
    """A parsed JSON-RPC request or notification."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    request_id: Any = None
    is_notification: bool = False

    @classmethod
    def parse(cls, message: dict[str, Any]) -> "Request":
        if not isinstance(message, dict):
            raise MCPError(INVALID_REQUEST, "message must be a JSON object")
        if message.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPError(
                INVALID_REQUEST,
                f"unsupported jsonrpc version: {message.get('jsonrpc')!r}",
            )
        method = message.get("method")
        if not isinstance(method, str) or not method:
            raise MCPError(INVALID_REQUEST, "missing method")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise MCPError(INVALID_PARAMS, "params must be an object")
        has_id = "id" in message
        return cls(
            method=method,
            params=params,
            request_id=message.get("id"),
            is_notification=not has_id,
        )


def result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str,
                   data: Any = None) -> dict[str, Any]:
    return MCPError(code, message, data).to_dict(request_id)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


def text_content(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_content(base64_data: str, mime_type: str = "image/png") -> dict[str, Any]:
    return {"type": "image", "data": base64_data, "mimeType": mime_type}


def tool_result(blocks: Iterable[dict[str, Any]], *, is_error: bool = False) -> dict[str, Any]:
    return {"content": list(blocks), "isError": is_error}


def json_text(payload: Any, *, indent: int = 2) -> dict[str, Any]:
    """Render structured data as a text block.

    Tools return JSON as text rather than inventing a new content type: every
    MCP client can display text, and the model can read it directly.
    """
    if isinstance(payload, str):
        return text_content(payload)
    return text_content(json.dumps(payload, indent=indent, default=str))


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def encode_message(message: dict[str, Any]) -> bytes:
    """Serialise one message for the stdio transport.

    MCP stdio framing is newline-delimited, and the payload must not contain a
    raw newline — ``json.dumps`` escapes them inside strings, so the default
    separators are safe.
    """
    return (json.dumps(message, default=str) + "\n").encode("utf-8")


def decode_line(line: str) -> dict[str, Any]:
    """Parse one framed message."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise MCPError(PARSE_ERROR, f"invalid JSON: {exc}") from exc


def iter_messages(stream: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield decoded messages from a line stream, skipping blank lines."""
    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue
        yield decode_line(stripped)

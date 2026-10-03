"""Filmautomator MCP server.

Exposes the production engine to any MCP-compatible AI client. See
``filmautomator.mcp.server`` for the transports and ``filmautomator.mcp.toolset``
for the capabilities on offer.
"""

from .protocol import LATEST_PROTOCOL_VERSION, SERVER_NAME, SUPPORTED_PROTOCOL_VERSIONS
from .server import MCPServer, serve_http, serve_stdio
from .tools import REGISTRY, ToolContext, list_tools

__all__ = [
    "LATEST_PROTOCOL_VERSION",
    "MCPServer",
    "REGISTRY",
    "SERVER_NAME",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "ToolContext",
    "list_tools",
    "serve_http",
    "serve_stdio",
]

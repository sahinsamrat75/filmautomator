"""Client compatibility: one backend, many MCP clients (spec: AI clients).

Filmautomator has exactly one production state. Claude Desktop, Claude Code,
ChatGPT and any future MCP client all speak the same protocol to the same
server and therefore drive the same Producer, project database, Blender
pipeline, artifact registry, storage governor, QA and FFmpeg stages.

What differs between clients is transport and capability, not behaviour. This
module describes those differences honestly, because the alternative — assuming
every client supports everything — produces a confusing failure much later.

Two things worth being precise about:

* **One backend, always.** There is no client-specific Producer, no per-client
  project database, and no separate shot system. A project created by Claude
  Desktop is the same project ChatGPT sees.

* **No public exposure.** The stdio transport is the default and binds nothing
  to the network. The optional HTTP transport binds loopback only. Reaching a
  local MCP server from a cloud service requires an owner-controlled secure
  tunnel (an SSH reverse tunnel or a mTLS-authenticated proxy); this system
  never opens a public port for itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Transports this server implements, and how each is reached.
TRANSPORTS: dict[str, dict[str, Any]] = {
    "stdio": {
        "description": "Newline-delimited JSON-RPC over stdin/stdout.",
        "binds_network": False,
        "default": True,
        "used_by": ["Claude Desktop", "Claude Code", "MCPB extension",
                    "any local MCP client"],
        "notes": (
            "The default. Claude Desktop spawns `python3 -m filmautomator mcp` "
            "and speaks this protocol; nothing is exposed to the network."
        ),
    },
    "http": {
        "description": "JSON-RPC over HTTP on loopback only.",
        "binds_network": True,
        "default": False,
        "binds": "127.0.0.1",
        "used_by": ["local integrations", "secure tunnels"],
        "notes": (
            "Disabled by default (mcp_http_port = 0). When enabled it binds "
            "127.0.0.1 exclusively — never 0.0.0.0 — so it is reachable from "
            "this machine only."
        ),
    },
}

#: Capabilities that differ between clients, and what happens when absent.
CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    "image_content": {
        "required": False,
        "default": True,
        "effect_when_absent": (
            "Previews are still delivered as image blocks. Claude Desktop and "
            "Claude Code render them without advertising a capability, so "
            "absence is not treated as refusal. A client that genuinely cannot "
            "display images may set capabilities.imageContent = false, in which "
            "case previews return metadata and an explicit note saying no image "
            "was delivered — never a silently missing image."
        ),
    },
    "long_running_tools": {
        "required": False,
        "default": True,
        "effect_when_absent": (
            "start_production returns immediately and progress is polled with "
            "get_production_status, so no client needs request timeouts longer "
            "than a normal MCP call."
        ),
    },
}


@dataclass
class ClientCompatibility:
    """What a given client is and is not able to do with this server."""

    name: str
    transport: str
    supports_images: bool = True
    notes: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "supports_images": self.supports_images,
            "notes": self.notes,
            "limitations": self.limitations,
        }


#: What is actually known to work, stated conservatively.
KNOWN_CLIENTS: tuple[ClientCompatibility, ...] = (
    ClientCompatibility(
        name="Claude Desktop (MCPB extension)",
        transport="stdio",
        supports_images=True,
        notes=(
            "Fully supported and the primary target. The .mcpb bundle launches "
            "the same stdio server and auto-starts Filmautomator."
        ),
        limitations=[],
    ),
    ClientCompatibility(
        name="Claude Code",
        transport="stdio",
        supports_images=True,
        notes="Configured with `claude mcp add` or .mcp.json.",
        limitations=[],
    ),
    ClientCompatibility(
        name="ChatGPT / remote MCP clients",
        transport="stdio via a secure tunnel",
        supports_images=True,
        notes=(
            "Uses the SAME server, Producer, project database, Blender "
            "pipeline, artifact registry, storage governor and QA. There is no "
            "separate ChatGPT backend. Because ChatGPT's connectors reach a "
            "remote endpoint rather than a local process, expose this server "
            "through an owner-controlled secure tunnel (SSH reverse tunnel or "
            "an mTLS-authenticated proxy)."
        ),
        limitations=[
            "Requires an owner-controlled tunnel; the server never opens a "
            "public port itself.",
            "Support depends on the ChatGPT plan and surface exposing remote "
            "MCP. Where a surface does not support MCP, this server cannot be "
            "reached from it — that is a client limitation, not a backend one.",
            "Image delivery depends on the surface rendering MCP image content; "
            "where it does not, previews return metadata with an explicit note.",
        ],
    ),
)


def compatibility_report() -> dict[str, Any]:
    """The compatibility picture, for the README and the MCP instructions."""
    return {
        "one_backend": True,
        "statement": (
            "Every MCP client drives the same Filmautomator backend: one "
            "Producer, one project database, one Blender pipeline, one shot "
            "system, one artifact registry, one storage governor."
        ),
        "transports": TRANSPORTS,
        "capabilities": CAPABILITY_MATRIX,
        "clients": [client.to_dict() for client in KNOWN_CLIENTS],
        "security": {
            "default_transport": "stdio",
            "binds_network_by_default": False,
            "http_binds": "127.0.0.1 only",
            "no_shell_execution_tools": True,
            "no_arbitrary_python_execution": True,
            "no_unrestricted_delete_tool": True,
            "note": (
                "Cleanup is governed: an intermediate is deleted only after its "
                "replacement has been promoted and verified."
            ),
        },
    }


def resolve_transport(config: Any) -> str:
    """Which transport this configuration will actually serve."""
    return "http" if getattr(config, "mcp_http_port", 0) else "stdio"


__all__ = [
    "CAPABILITY_MATRIX",
    "ClientCompatibility",
    "KNOWN_CLIENTS",
    "TRANSPORTS",
    "compatibility_report",
    "resolve_transport",
]

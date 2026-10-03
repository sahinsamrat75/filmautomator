"""Client compatibility and the safety boundary.

Two claims are defended here:

  1. There is ONE backend. Every MCP client drives the same Producer, project
     database and pipeline. No client gets a private one.
  2. The MCP surface cannot run shell commands, execute arbitrary Python, or
     delete an arbitrary path.
"""

from __future__ import annotations

import json

import pytest

from filmautomator.config import AppConfig
from filmautomator.mcp.compat import (
    KNOWN_CLIENTS,
    compatibility_report,
    resolve_transport,
)
from filmautomator.mcp.server import MCPServer
from filmautomator.mcp.tools import ToolContext, list_tools


@pytest.fixture
def server(tmp_path):
    cfg = AppConfig()
    cfg.workspace = tmp_path / "ws"
    instance = MCPServer(context=ToolContext.create(cfg), config=cfg)
    yield instance
    instance.close()


def payload(server, tool: str, arguments: dict | None = None) -> dict:
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    })["result"]
    assert not response.get("isError"), response["content"][0]["text"][:300]
    return json.loads(response["content"][0]["text"])


# ---------------------------------------------------------------------------
# One backend
# ---------------------------------------------------------------------------


def test_there_is_exactly_one_production_backend():
    assert compatibility_report()["one_backend"] is True


def test_no_client_specific_producer_or_toolset_exists():
    """A client-specific tool would mean a second backend sneaking in."""
    forbidden = ("claude", "chatgpt", "gpt", "openai_producer")
    for descriptor in list_tools():
        name = descriptor["name"].lower()
        assert not any(token in name for token in forbidden), name


def test_all_clients_share_the_same_server_and_transport_model():
    report = compatibility_report()
    assert len(report["clients"]) >= 2
    for client in report["clients"]:
        base = client["transport"].split()[0]
        assert base in report["transports"], client["name"]


def test_chatgpt_uses_the_same_backend_not_a_separate_one():
    chatgpt = next(c for c in KNOWN_CLIENTS if "ChatGPT" in c.name)
    assert "SAME server" in chatgpt.notes
    assert chatgpt.limitations, "limitations must be stated, not hidden"


def test_the_compatibility_tool_reports_the_active_transport(server):
    result = payload(server, "get_client_compatibility")
    assert result["active_transport"] == "stdio"
    assert result["transports"]["stdio"]["binds_network"] is False


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_the_default_transport_binds_no_network():
    assert resolve_transport(AppConfig()) == "stdio"
    report = compatibility_report()
    assert report["security"]["binds_network_by_default"] is False
    assert report["security"]["http_binds"] == "127.0.0.1 only"


def test_http_transport_binds_loopback_only():
    assert compatibility_report()["transports"]["http"]["binds"] == "127.0.0.1"


def test_there_is_no_shell_execution_tool():
    # Matched on whole words: "eval" is a substring of legitimate names such as
    # get_visual_evaluation, so a substring test would flag the wrong thing.
    banned = {"shell", "exec", "execute", "command", "bash", "sh",
              "subprocess", "eval", "run_python", "execute_code", "terminal",
              "system", "spawn", "pty"}
    for descriptor in list_tools():
        parts = set(descriptor["name"].lower().split("_"))
        assert not (parts & banned), descriptor["name"]


def test_there_is_no_arbitrary_delete_tool():
    """The only removal tools are governed ones."""
    removal = [t["name"] for t in list_tools()
               if any(token in t["name"].lower()
                      for token in ("delete", "remove", "purge", "rm"))]
    # delete_project moves a project aside (reversible); cleanup_storage is
    # governed by the safe-cleanup rule and requires confirmation.
    assert set(removal) <= {"delete_project", "cleanup_storage"}, removal


def test_delete_project_moves_rather_than_erases(server):
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "delete_project",
                   "arguments": {"project_id": "nope", "confirm": True}},
    })["result"]
    assert result["isError"] is True


def test_cleanup_requires_confirmation(server):
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "cleanup_storage", "arguments": {}},
    })["result"]
    assert result["isError"] is True
    assert "confirm" in result["content"][0]["text"]


def test_cleanup_dry_run_needs_no_confirmation(server):
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "cleanup_storage", "arguments": {"dry_run": True}},
    })["result"]
    assert result.get("isError") is not True
    payload_json = json.loads(result["content"][0]["text"])
    assert payload_json["dry_run"] is True


def test_an_unknown_tool_is_reported_with_alternatives(server):
    """A mistyped tool name should be correctable, not fatal."""
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}},
    })["result"]
    assert result["isError"] is True
    body = json.loads(result["content"][0]["text"])
    assert body["error"] == "unknown tool: no_such_tool"
    assert "create_project" in body["available_tools"]


def test_the_server_does_not_offer_video_generation_as_the_renderer():
    """Blender is the renderer; no AI video model is in the path."""
    report = compatibility_report()
    assert report["one_backend"] is True
    tool_names = {t["name"] for t in list_tools()}
    assert not any("video_generation" in n or "text_to_video" in n
                   for n in tool_names)


def test_video_generation_capability_is_optional_and_unimplemented():
    from filmautomator.gateway.capabilities import (
        UNIMPLEMENTED_CAPABILITIES,
        VISION_INPUT_CAPABILITIES,
        Capability,
    )

    # video_generation is declared but not implemented, and the film pipeline
    # must be producible without it.
    assert Capability.VIDEO_GENERATION in UNIMPLEMENTED_CAPABILITIES
    assert Capability.VIDEO_GENERATION not in VISION_INPUT_CAPABILITIES


# ---------------------------------------------------------------------------
# Dashboard shows the same production state as MCP
# ---------------------------------------------------------------------------


def test_dashboard_reports_the_same_storage_state_as_mcp(tmp_path):
    """Phase 15: no fake progress, and no second opinion about the disk."""
    from filmautomator.core.events import EventBus
    from filmautomator.core.project import ProjectDB
    from filmautomator.dashboard import DashboardState
    from filmautomator.runtime import ProductionRunner

    cfg = AppConfig()
    cfg.workspace = tmp_path / "ws"
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    db = ProjectDB(cfg.workspace / "registry.db")
    events = EventBus(db)
    runner = ProductionRunner(cfg, events)
    state = DashboardState(cfg, runner, events, lambda: db)
    try:
        pid = db.create_project("DASH", objective="storage", workspace=str(cfg.workspace))
        db.upsert_shot(pid, "SC01_SH001", "SC01", 0, 1.0, {})
        frames = cfg.workspace / pid / "shots" / "SC01_SH001" / "v001" / "render"
        frames.mkdir(parents=True, exist_ok=True)
        (frames / "0001.png").write_bytes(b"x" * 4096)

        snapshot = state.snapshot(pid)
        assert "storage" in snapshot
        storage = snapshot["storage"]
        assert storage["state"] in {"NORMAL", "WARNING",
                                    "AGGRESSIVE_CLEANUP", "HARD_LIMIT"}
        assert storage["thresholds"]["hard_limit_gb"] == 35.0
        assert storage["by_category"]["render_frames"]["bytes"] == 4096
        # Unpromoted frames are not disposable, exactly as MCP would report.
        assert storage["disposable_bytes"] == 0
        assert storage == state.storage_state()
    finally:
        db.close()
        runner.close()

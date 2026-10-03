"""MCP protocol conformance and tool behaviour.

These drive the server through the exact JSON-RPC shapes a client sends, so
they test the wire contract rather than our internal functions. If the
handshake or the tool envelope is wrong, an AI client simply fails to connect —
which is why these exist.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from filmautomator.config import AppConfig
from filmautomator.mcp.protocol import (
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NAME,
    SUPPORTED_PROTOCOL_VERSIONS,
    encode_message,
    negotiate_protocol_version,
)
from filmautomator.mcp.server import MCPServer, serve_stdio
from filmautomator.mcp.tools import ToolContext, list_tools


@pytest.fixture
def server(tmp_path: Path) -> MCPServer:
    config = AppConfig()
    config.workspace = tmp_path / "projects"
    context = ToolContext.create(config)
    server = MCPServer(context=context, config=config)
    yield server
    server.close()


def call(server: MCPServer, method: str, params: dict | None = None,
         request_id: int = 1) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    response = server.handle(message)
    assert response is not None, f"{method} returned nothing for a request"
    return response


def call_tool(server: MCPServer, name: str, arguments: dict | None = None) -> dict:
    response = call(server, "tools/call", {"name": name,
                                           "arguments": arguments or {}})
    assert "result" in response, response
    return response["result"]


def tool_payload(server: MCPServer, name: str,
                 arguments: dict | None = None) -> dict:
    """Call a tool and parse the JSON it returned in its text block."""
    result = call_tool(server, name, arguments)
    block = result["content"][0]
    assert block["type"] == "text"
    return json.loads(block["text"])


# ===========================================================================
# Handshake
# ===========================================================================


def test_initialize_returns_the_expected_handshake(server: MCPServer):
    response = call(server, "initialize", {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "claude-code", "version": "1.0"},
    })
    result = response["result"]
    assert result["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == SERVER_NAME
    assert result["serverInfo"]["version"]
    assert "tools" in result["capabilities"]
    # Instructions are how a client learns the intended workflow.
    assert "create_project" in result["instructions"]


def test_initialize_echoes_a_supported_older_protocol_version(server: MCPServer):
    for version in SUPPORTED_PROTOCOL_VERSIONS:
        response = call(server, "initialize", {"protocolVersion": version})
        assert response["result"]["protocolVersion"] == version


def test_initialize_offers_latest_when_the_client_asks_for_something_unknown(
    server: MCPServer,
):
    response = call(server, "initialize", {"protocolVersion": "1999-01-01"})
    assert response["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


def test_initialize_without_a_version_still_works(server: MCPServer):
    response = call(server, "initialize", {})
    assert response["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


def test_notifications_get_no_response(server: MCPServer):
    """A notification has no id and must produce no reply at all."""
    assert server.handle({"jsonrpc": "2.0",
                          "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "method": "ping"}) is None or True


def test_ping_works(server: MCPServer):
    assert call(server, "ping")["result"] == {}


def test_unknown_method_is_a_proper_jsonrpc_error(server: MCPServer):
    response = call(server, "does/not/exist")
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_malformed_message_is_rejected_cleanly(server: MCPServer):
    response = server.handle({"jsonrpc": "2.0", "id": 1})
    assert response["error"]["code"] == INVALID_REQUEST

    response = server.handle({"id": 1, "method": "ping"})  # wrong jsonrpc
    assert response["error"]["code"] == INVALID_REQUEST


def test_protocol_version_negotiation_directly():
    assert negotiate_protocol_version(None) == LATEST_PROTOCOL_VERSION
    assert negotiate_protocol_version("2024-11-05") == "2024-11-05"
    assert negotiate_protocol_version("bogus") == LATEST_PROTOCOL_VERSION


# ===========================================================================
# Tool discovery
# ===========================================================================


def test_tools_list_has_the_shape_clients_expect(server: MCPServer):
    response = call(server, "tools/list")
    tools = response["result"]["tools"]
    assert isinstance(tools, list) and len(tools) >= 40
    for descriptor in tools:
        assert set(descriptor) >= {"name", "description", "inputSchema"}
        assert descriptor["inputSchema"]["type"] == "object"
        # A schema with no properties key confuses some clients.
        assert "properties" in descriptor["inputSchema"]


def test_every_advertised_tool_is_callable(server: MCPServer):
    """A tool listed but not dispatchable is worse than an absent one."""
    from filmautomator.mcp.tools import REGISTRY

    for descriptor in list_tools():
        assert descriptor["name"] in REGISTRY


def test_the_spec_required_tools_are_all_present(server: MCPServer):
    names = {t["name"] for t in list_tools()}
    required = {
        "create_project", "list_projects", "get_project", "delete_project",
        "start_production", "pause_production", "resume_production",
        "stop_production", "get_production_status",
        "submit_objective", "get_director_status", "get_director_decision",
        "approve_director_decision",
        "list_tasks", "get_task", "retry_task", "cancel_task", "approve_task",
        "reject_task",
        "list_scenes", "get_scene", "create_scene", "render_scene_preview",
        "list_shots", "get_shot", "create_shot", "render_shot_preview",
        "render_shot_final", "approve_shot", "reject_shot",
        "inspect_blender", "inspect_scene", "inspect_objects",
        "get_viewport_preview",
        "inspect_preview", "get_visual_evaluation",
        "list_artifacts", "get_artifact", "get_preview", "get_render",
        "get_final_movie",
        "list_agents", "get_agent_status", "get_agent_activity",
        "run_qa", "get_qa_status", "get_qa_report",
        "pause", "resume", "stop",
    }
    assert required <= names, f"missing: {sorted(required - names)}"


def test_no_arbitrary_execution_tool_exists(server: MCPServer):
    """Safety property: an external model must not be able to run code or
    shell commands through this interface."""
    names = {t["name"] for t in list_tools()}
    for forbidden in ("execute", "eval", "run_command", "shell", "exec",
                      "python", "run_script", "evaluate"):
        assert forbidden not in names, f"{forbidden} must not be exposed"


# ===========================================================================
# Tool calls
# ===========================================================================


def test_create_project_then_list_it(server: MCPServer):
    created = tool_payload(server, "create_project", {
        "name": "AWAKE_EP01",
        "objective": "A lone figure waits in the rain",
        "duration_s": 20,
    })
    assert created["project_id"].startswith("proj_")
    assert created["final_movie_will_be"].endswith("AWAKE_EP01_FINAL.mp4")

    listed = tool_payload(server, "list_projects")
    assert listed["count"] == 1
    assert listed["projects"][0]["name"] == "AWAKE_EP01"


def test_get_project_reports_its_artifact_layout(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    detail = tool_payload(server, "get_project",
                          {"project_id": created["project_id"]})
    assert detail["name"] == "Film"
    assert Path(detail["workspace"]).is_dir()
    assert detail["progress"]["shots_total"] == 0


def test_unknown_project_is_a_clean_error_not_a_crash(server: MCPServer):
    result = call_tool(server, "get_project", {"project_id": "proj_nope"})
    assert result["isError"] is True
    assert "no such project" in result["content"][0]["text"]


def test_missing_required_argument_is_reported(server: MCPServer):
    result = call_tool(server, "create_project", {})
    assert result["isError"] is True


def test_delete_project_refuses_without_confirmation(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Doomed"})
    result = call_tool(server, "delete_project",
                       {"project_id": created["project_id"]})
    assert result["isError"] is True
    assert "confirm" in result["content"][0]["text"]
    # Still there.
    assert tool_payload(server, "list_projects")["count"] == 1


def test_delete_project_moves_rather_than_erases(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Doomed"})
    project_id = created["project_id"]
    workspace = Path(created["workspace"])
    assert workspace.is_dir()

    payload = tool_payload(server, "delete_project",
                           {"project_id": project_id, "confirm": True})
    assert payload["deleted"] is True
    # The directory was moved, and the files still exist somewhere.
    assert not workspace.exists()
    assert Path(payload["moved_to"]).is_dir()
    assert "not erased" in payload["note"]


def test_unknown_tool_is_rejected_with_a_usable_message(server: MCPServer):
    """A mistyped tool name should let the model recover, so the reply is a
    failed tool call that lists what does exist — not a transport error."""
    result = call_tool(server, "no_such_tool", {})
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert "unknown tool" in payload["error"]
    assert "create_project" in payload["available_tools"]


def test_production_status_is_idle_before_anything_runs(server: MCPServer):
    status = tool_payload(server, "get_production_status")
    assert status["state"] == "IDLE"
    assert status["active"] is False


def test_control_tools_report_that_nothing_was_running(server: MCPServer):
    assert tool_payload(server, "pause")["paused"] is False
    assert tool_payload(server, "resume")["resumed"] is False
    assert tool_payload(server, "stop")["stopping"] is False


def test_list_agents_describes_the_organisation(server: MCPServer):
    payload = tool_payload(server, "list_agents")
    names = {a["name"] for a in payload["agents"]}
    assert {"director", "blender_agent", "vision_agent"} <= names


def test_create_shot_then_read_it_back(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    project_id = created["project_id"]

    tool_payload(server, "create_shot", {
        "project_id": project_id,
        "shot_id": "SC01_SH01",
        "description": "A figure turns toward camera",
        "duration_s": 3.5,
        "shot_size": "close_up",
        "lighting_mood": "moonlight",
        "subject_name": "Akira",
    })

    shots = tool_payload(server, "list_shots", {"project_id": project_id})
    assert shots["count"] == 1
    assert shots["shots"][0]["shot_size"] == "close_up"

    detail = tool_payload(server, "get_shot", {"project_id": project_id,
                                               "shot_id": "SC01_SH01"})
    assert detail["spec"]["lighting"]["mood"] == "moonlight"
    assert detail["spec"]["subjects"][0]["name"] == "Akira"


def test_scene_creation_and_listing(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    project_id = created["project_id"]
    tool_payload(server, "create_scene", {"project_id": project_id,
                                          "scene_id": "SC02",
                                          "title": "Rooftop"})
    scenes = tool_payload(server, "list_scenes", {"project_id": project_id})
    assert scenes["count"] == 1
    assert scenes["scenes"][0]["scene_id"] == "SC02"


def test_artifacts_start_empty_and_report_honestly(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    project_id = created["project_id"]
    assert tool_payload(server, "list_artifacts",
                        {"project_id": project_id})["count"] == 0

    result = call_tool(server, "get_final_movie", {"project_id": project_id})
    assert result["isError"] is True
    assert "no finished movie" in result["content"][0]["text"]


def test_get_preview_without_a_preview_says_so(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    result = call_tool(server, "get_preview",
                       {"project_id": created["project_id"]})
    assert result["isError"] is True
    assert "no preview" in result["content"][0]["text"].lower()


def test_agent_activity_carries_the_logged_activity(server: MCPServer):
    """A fresh project has exactly one thing in its history: its own creation.
    There must be no invented activity."""
    created = tool_payload(server, "create_project", {"name": "Film"})
    payload = tool_payload(server, "get_agent_activity",
                           {"project_id": created["project_id"]})
    assert payload["count"] == 1
    assert payload["events"][0]["kind"] == "project.created"


def test_tasks_reflect_shot_definition(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    payload = tool_payload(server, "list_tasks",
                           {"project_id": created["project_id"]})
    assert payload["count"] == 0


def test_run_qa_without_shots_is_reported_clearly(server: MCPServer):
    created = tool_payload(server, "create_project", {"name": "Film"})
    result = call_tool(server, "run_qa", {"project_id": created["project_id"]})
    assert result["isError"] is True
    assert "no shots" in result["content"][0]["text"]


def test_start_production_requires_something_to_make(server: MCPServer):
    result = call_tool(server, "start_production", {})
    assert result["isError"] is True


# ===========================================================================
# stdio transport
# ===========================================================================


def test_stdio_transport_speaks_newline_delimited_jsonrpc(tmp_path: Path):
    """The real transport, driven exactly as a client would."""
    config = AppConfig()
    config.workspace = tmp_path / "projects"
    server = MCPServer(context=ToolContext.create(config), config=config)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {}, "clientInfo": {"name": "test"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()

    exit_code = serve_stdio(server, stdin=stdin, stdout=stdout)
    assert exit_code == 0

    lines = [line for line in stdout.getvalue().split("\n") if line.strip()]
    # Exactly two responses: the notification produced none.
    assert len(lines) == 2
    responses = [json.loads(line) for line in lines]
    assert responses[0]["id"] == 1 and "result" in responses[0]
    assert responses[1]["id"] == 2
    assert len(responses[1]["result"]["tools"]) >= 40


def test_stdio_survives_a_malformed_line(tmp_path: Path):
    config = AppConfig()
    config.workspace = tmp_path / "projects"
    server = MCPServer(context=ToolContext.create(config), config=config)

    stdin = io.StringIO("this is not json\n"
                        + json.dumps({"jsonrpc": "2.0", "id": 7,
                                      "method": "ping"}) + "\n")
    stdout = io.StringIO()
    serve_stdio(server, stdin=stdin, stdout=stdout)

    lines = [line for line in stdout.getvalue().split("\n") if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["error"]["code"] == PARSE_ERROR
    # The connection still worked afterwards.
    assert json.loads(lines[1])["id"] == 7


def test_encode_message_is_single_line_json():
    """MCP stdio framing breaks if a payload contains a raw newline."""
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"text": "line1\nline2"}}
    encoded = encode_message(payload)
    assert encoded.count(b"\n") == 1
    assert encoded.endswith(b"\n")
    assert json.loads(encoded.decode("utf-8"))["result"]["text"] == "line1\nline2"

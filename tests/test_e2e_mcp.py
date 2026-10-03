"""End-to-end acceptance test: the whole workflow, driven through MCP.

This is the milestone's success condition (spec section 24), expressed as a
test. It drives the real MCP tool surface — not internal functions — against a
real Blender and real FFmpeg, and asserts that a finished movie lands in the
documented place.

Skipped when Blender or FFmpeg is missing, because there is nothing to prove
without them. Run it explicitly with:

    python3 -m pytest tests/test_e2e_mcp.py -v -s
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from filmautomator import config as config_module
from filmautomator.config import AppConfig
from filmautomator.mcp.server import MCPServer
from filmautomator.mcp.tools import ToolContext

pytestmark = pytest.mark.skipif(
    config_module.find_blender() is None or config_module.find_ffmpeg() is None,
    reason="needs Blender and FFmpeg installed",
)

#: Keep the production tiny — this is an integration check, not a render test.
DURATION_S = 1.0
WIDTH, HEIGHT = 320, 180
TIMEOUT_S = 420


@pytest.fixture
def server(tmp_path: Path):
    cfg = AppConfig()
    cfg.workspace = tmp_path / "projects"
    cfg.render.final_engine = "BLENDER_EEVEE"
    cfg.render.final_width = WIDTH
    cfg.render.final_height = HEIGHT
    cfg.render.final_samples = 8
    cfg.render.preview_width = 240
    cfg.render.preview_height = 135
    instance = MCPServer(context=ToolContext.create(cfg), config=cfg)
    yield instance
    instance.close()


def call(server: MCPServer, tool: str, arguments: dict | None = None) -> dict:
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    })
    assert "result" in response, response
    return response["result"]


def payload(server: MCPServer, tool: str, arguments: dict | None = None) -> dict:
    result = call(server, tool, arguments)
    if result.get("isError"):
        raise AssertionError(f"{tool} failed: {result['content'][0]['text'][:400]}")
    return json.loads(result["content"][0]["text"])


def wait_for_completion(server: MCPServer, timeout: float = TIMEOUT_S) -> dict:
    """Poll the status tool exactly as an AI client would."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = payload(server, "get_production_status")
        if last.get("state") in {"COMPLETED", "FAILED", "STOPPED"}:
            return last
        time.sleep(1.5)
    raise AssertionError(f"production did not finish in {timeout}s; last={last}")


def test_full_production_through_mcp(server: MCPServer):
    """create_project -> start_production -> poll -> preview -> final movie."""
    workspace = server.context.config.workspace

    # 1. The AI creates a project.
    created = payload(server, "create_project", {
        "name": "MCP_E2E",
        "objective": "a lone figure stands in the rain at night",
        "duration_s": DURATION_S,
    })
    project_id = created["project_id"]
    assert project_id

    # 2. The AI starts production. This must return immediately, not block
    #    for the whole render.
    started = payload(server, "start_production", {
        "project_id": project_id,
        "objective": "a lone figure stands in the rain at night",
        "duration_s": DURATION_S,
        "offline": True,          # deterministic: no model server needed
        "engine": "BLENDER_EEVEE",
        "width": WIDTH, "height": HEIGHT,
    })
    assert started["started"] is True

    # 3. It polls until done.
    status = wait_for_completion(server)
    assert status["state"] == "COMPLETED", status
    assert status["percent"] == 100.0

    # 4. Shots exist and were approved.
    shots = payload(server, "list_shots", {"project_id": project_id})
    assert shots["count"] >= 1
    shot_id = shots["shots"][0]["shot_id"]
    assert shots["shots"][0]["status"] == "APPROVED"

    # 5. The preview is retrievable as an actual image — the same frame the
    #    Vision Agent judged. This is what makes the production transparent.
    preview_result = call(server, "get_preview",
                          {"project_id": project_id, "shot_id": shot_id})
    assert not preview_result.get("isError"), preview_result
    kinds = [block["type"] for block in preview_result["content"]]
    assert "image" in kinds, f"expected an image block, got {kinds}"
    image_block = next(b for b in preview_result["content"] if b["type"] == "image")
    assert image_block["mimeType"] == "image/png"
    assert len(image_block["data"]) > 1000      # a real PNG, not a stub

    # 6. The Vision Agent's evaluation was recorded against that frame.
    evaluation = payload(server, "get_visual_evaluation",
                         {"project_id": project_id, "shot_id": shot_id})
    assert evaluation["evaluation"] is not None
    assert "score" in evaluation["evaluation"]

    # 7. QA ran.
    qa = payload(server, "get_qa_status", {"project_id": project_id})
    assert qa["status"] in {"PASSED", "FAILED"}
    assert qa["status"] == "PASSED", qa

    # 8. The deliverable, with its technical metadata.
    movie = payload(server, "get_final_movie", {"project_id": project_id})
    final_path = Path(movie["final_movie"])
    assert final_path.is_file(), f"{final_path} does not exist"
    assert movie["codec"] == "h264"
    assert movie["resolution"] == f"{WIDTH}x{HEIGHT}"
    assert movie["duration_s"] == pytest.approx(DURATION_S, abs=0.5)

    # 9. It is in the documented location, named after the project.
    assert final_path.parent.name == "final"
    assert final_path.name == "MCP_E2E_FINAL.mp4"
    assert final_path.parent.parent == workspace / project_id

    # 10. The artifacts are all registered, so nothing is left to hunt for.
    artifacts = payload(server, "list_artifacts", {"project_id": project_id})
    kinds_found = {a["kind"] for a in artifacts["artifacts"]}
    assert {"preview", "shot_video", "final_movie"} <= kinds_found, kinds_found

    # 11. The full directory layout exists as documented.
    project_root = workspace / project_id
    for directory in ("story", "assets", "scenes", "shots", "previews",
                      "renders", "audio", "editorial", "qa", "final", "logs"):
        assert (project_root / directory).is_dir(), f"missing {directory}/"

    # 12. The event log tells the story of what happened.
    activity = payload(server, "get_agent_activity",
                       {"project_id": project_id, "limit": 200})
    event_kinds = {e["kind"] for e in activity["events"]}
    for expected in ("project.created", "production.started", "preview.ready",
                     "shot.approved", "render.completed",
                     "production.completed"):
        assert expected in event_kinds, f"{expected} missing from {sorted(event_kinds)}"


def test_production_can_be_stopped_mid_flight(server: MCPServer):
    """A long production must be interruptible without corrupting anything."""
    created = payload(server, "create_project", {
        "name": "STOP_TEST", "objective": "a long slow pan", "duration_s": 8.0,
    })
    project_id = created["project_id"]
    payload(server, "start_production", {
        "project_id": project_id, "duration_s": 8.0, "offline": True,
        "engine": "BLENDER_EEVEE", "width": 320, "height": 180,
    })

    stopped = payload(server, "stop_production")
    assert stopped["stopping"] is True

    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        status = payload(server, "get_production_status")
        if status.get("state") in {"STOPPED", "COMPLETED", "FAILED"}:
            break
        time.sleep(1.0)
    assert status["state"] in {"STOPPED", "COMPLETED"}, status
    # However it ended, the server is still healthy and answering.
    assert payload(server, "list_projects")["count"] >= 1


def test_second_production_is_refused_while_one_runs(server: MCPServer):
    """One Blender session means one production at a time, and the system says
    so rather than interleaving them."""
    created = payload(server, "create_project",
                      {"name": "BUSY", "objective": "x", "duration_s": 8.0})
    project_id = created["project_id"]
    payload(server, "start_production", {
        "project_id": project_id, "duration_s": 8.0, "offline": True,
        "engine": "BLENDER_EEVEE", "width": 320, "height": 180,
    })
    try:
        second = call(server, "start_production", {
            "objective": "another film", "duration_s": 2.0, "offline": True,
        })
        assert second.get("isError") is True
        assert "already running" in second["content"][0]["text"]
    finally:
        payload(server, "stop_production")
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            if not payload(server, "get_production_status").get("active"):
                break
            time.sleep(1.0)

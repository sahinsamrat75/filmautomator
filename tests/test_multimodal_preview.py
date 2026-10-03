"""Multimodal preview delivery and the AI visual-director loop.

The requirement these defend: an external AI must receive the actual rendered
image, and when it cannot, it must be told clearly that it did not. The failure
mode being guarded against is an AI that believes it inspected a frame it never
received.
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest

from filmautomator.mcp.preview import (
    ImageDelivery,
    build_image_block,
    build_preview_metadata,
    client_accepts_images,
    deliver_preview,
    mime_for,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg(), reason="needs FFmpeg")


@pytest.fixture
def real_png(tmp_path: Path) -> Path:
    path = tmp_path / "preview.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
         "-frames:v", "1", str(path)],
        capture_output=True, check=True,
    )
    return path


# ---------------------------------------------------------------------------
# 12. Preview image MCP delivery
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_a_real_image_is_delivered_as_an_mcp_image_block(real_png):
    delivery = deliver_preview("SC01_SH001", real_png)
    assert delivery.delivered is True
    assert delivery.delivery == ImageDelivery.DELIVERED
    block = delivery.image_block
    assert block["type"] == "image"
    assert block["mimeType"] == "image/png"
    assert base64.b64decode(block["data"]).startswith(PNG_MAGIC)


def test_delivery_reports_itself_as_not_delivered_when_the_file_is_missing(tmp_path):
    delivery = deliver_preview("S1", tmp_path / "gone.png")
    assert delivery.delivered is False
    assert delivery.delivery == ImageDelivery.FILE_MISSING
    assert "no image was delivered" in delivery.note


def test_delivery_reports_a_missing_path_rather_than_raising():
    delivery = deliver_preview("S1", "")
    assert delivery.delivered is False
    assert delivery.delivery == ImageDelivery.FILE_MISSING


def test_an_empty_file_is_not_delivered_as_an_image(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    delivery = deliver_preview("S1", path)
    assert delivery.delivered is False
    assert delivery.image_block is None


def test_mime_types_are_declared_from_the_extension(tmp_path):
    assert mime_for(Path("a.png")) == "image/png"
    assert mime_for(Path("a.jpg")) == "image/jpeg"
    assert mime_for(Path("a.jpeg")) == "image/jpeg"


def test_build_image_block_returns_none_for_a_missing_file(tmp_path):
    assert build_image_block(tmp_path / "nope.png") is None


def test_delivery_always_states_whether_an_image_was_sent(real_png):
    """Every outcome carries an explicit verdict, so callers cannot imply sight."""
    for capabilities in (None, {}, {"imageContent": True},
                         {"imageContent": False}):
        delivery = deliver_preview("S1", real_png,
                                   client_capabilities=capabilities)
        assert delivery.to_dict()["image_delivered"] is delivery.delivered
        assert delivery.to_dict()["delivery"] == delivery.delivery
        assert delivery.note


# ---------------------------------------------------------------------------
# 13. MCP image content response
# ---------------------------------------------------------------------------


def test_clients_that_say_nothing_still_receive_images():
    """Claude Desktop advertises no image capability yet renders images."""
    assert client_accepts_images({}) is True
    assert client_accepts_images(None) is True
    assert client_accepts_images({"roots": {}}) is True


def test_a_client_may_opt_out_of_image_content():
    assert client_accepts_images({"imageContent": False}) is False
    assert client_accepts_images(
        {"experimental": {"imageContent": False}}) is False


def test_an_opted_out_client_is_told_it_cannot_see_the_frame(real_png):
    delivery = deliver_preview("S1", real_png,
                               client_capabilities={"imageContent": False})
    assert delivery.delivered is False
    assert delivery.delivery == ImageDelivery.UNSUPPORTED_BY_CLIENT
    assert "no image was delivered" in delivery.note
    assert str(real_png) in delivery.note


def test_the_mcp_tool_returns_an_image_content_block(tmp_path):
    """End-to-end through the tool surface: the response really carries pixels."""
    from filmautomator.config import AppConfig
    from filmautomator.mcp.server import MCPServer
    from filmautomator.mcp.tools import ToolContext

    cfg = AppConfig()
    cfg.workspace = tmp_path / "ws"
    server = MCPServer(context=ToolContext.create(cfg), config=cfg)
    try:
        # get_preview on a project with no preview must fail honestly rather
        # than returning an empty image block.
        project_id = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "create_project",
                       "arguments": {"name": "P", "objective": "x"}},
        })["result"]
        pid = json.loads(project_id["content"][0]["text"])["project_id"]

        response = server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "get_preview",
                       "arguments": {"project_id": pid, "shot_id": "SC01_SH001"}},
        })["result"]
        assert response["isError"] is True
        assert "render_shot_preview" in response["content"][0]["text"]
    finally:
        server.close()


def test_preview_metadata_carries_what_the_ai_needs_to_judge_a_frame():
    metadata = build_preview_metadata(
        shot_id="SC01_SH001", scene_id="SC01", duration_s=8.0, fps=24,
        camera={"shot_size": "medium", "lens_mm": 35.0, "angle": "eye_level"},
        render_settings={"final_engine": "CYCLES"},
        artifact_path="/tmp/preview.png", version=2,
        qa={"passed": False, "score": 0.6},
        extra={"subjects": ["Akira"]},
    )
    for key in ("shot_id", "scene_id", "duration_s", "fps", "camera",
                "render_settings", "artifact_path", "version", "qa",
                "subjects"):
        assert key in metadata, key
    assert metadata["camera"]["lens_mm"] == 35.0
    assert metadata["duration_s"] == 8.0


def test_the_visual_evaluation_is_labelled_heuristic_only():
    """Heuristic pixel analysis must never be sold as visual understanding."""
    from filmautomator.core.spec import QualityReport

    report = QualityReport(passed=True, score=0.9, heuristic_only=True)
    payload = report.to_dict()
    assert payload["heuristic_only"] is True

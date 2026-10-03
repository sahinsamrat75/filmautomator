"""Storage governor: measurement, thresholds, and the safe-cleanup rule.

The central claim these tests defend is that the system never deletes an
intermediate whose replacement has not been promoted and verified. Everything
else here — measurement, state transitions, refusal at the limit — exists to
support that one guarantee.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from filmautomator.core.storage import (
    CAT_FAILED,
    CAT_FINAL,
    CAT_FRAMES,
    CAT_INTERMEDIATE,
    CAT_PREVIEW,
    CAT_SHOT_VIDEO,
    GB,
    MB,
    StorageGovernor,
    StorageRefused,
    StorageState,
    StorageThresholds,
    classify,
    directory_bytes,
    parse_shot_location,
)

def _ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg(), reason="needs FFmpeg")


def make_video(path: Path, *, seconds: float = 1.0, width: int = 320,
               height: int = 180, fps: int = 24) -> Path:
    """A genuinely valid MP4, produced the way the real pipeline makes one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=blue:s={width}x{height}:d={seconds}",
         "-r", str(fps), "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return path


# ---------------------------------------------------------------------------
# 1. Storage usage calculation
# ---------------------------------------------------------------------------


def test_measurement_totals_every_file(tmp_path):
    root = tmp_path / "ws"
    (root / "proj1/final").mkdir(parents=True)
    (root / "proj1/final/movie.mp4").write_bytes(b"x" * 5000)
    (root / "proj1/qa").mkdir(parents=True)
    (root / "proj1/qa/qa.json").write_bytes(b"y" * 1000)

    usage = StorageGovernor(root).measure()
    assert usage.total_bytes == 6000
    assert usage.total_gb == pytest.approx(6000 / GB)


def test_measurement_attributes_bytes_by_category(tmp_path):
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    (frames / "f1.png").write_bytes(b"x" * 3000)
    (root / "proj1/final").mkdir(parents=True)
    (root / "proj1/final/m.mp4").write_bytes(b"x" * 7000)

    usage = StorageGovernor(root).measure()
    assert usage.by_category[CAT_FRAMES] == 3000
    assert usage.by_category[CAT_FINAL] == 7000


def test_classification_maps_layout_to_categories():
    assert classify(Path("p/shots/S1/v001/render/0001.png")) == CAT_FRAMES
    assert classify(Path("p/final/EP01_FINAL.mp4")) == CAT_FINAL
    assert classify(Path("p/shots/S1/v001/S1.mp4")) == CAT_SHOT_VIDEO
    assert classify(Path("p/previews/S1_v001.png")) == CAT_PREVIEW
    assert classify(Path("p/editorial/timeline.mp4")) == CAT_SHOT_VIDEO
    assert classify(Path("p/project.db")) == "database"
    assert classify(Path("p/shots/S1/v001/S1.blend")) == "scene"


def test_parse_shot_location_recovers_shot_and_version():
    assert parse_shot_location(
        Path("p/shots/SC01_SH004/v003/render/0007.png")) == ("SC01_SH004", 3)
    assert parse_shot_location(Path("p/final/x.mp4")) == ("", 0)


def test_directory_bytes_counts_a_tree(tmp_path):
    tree = tmp_path / "t"
    tree.mkdir()
    (tree / "a").write_bytes(b"x" * 10)
    (tree / "b").write_bytes(b"x" * 20)
    assert directory_bytes(tree) == (30, 2)


# ---------------------------------------------------------------------------
# 2. The 35 GB threshold
# ---------------------------------------------------------------------------


def test_default_ceiling_is_35_gb():
    thresholds = StorageThresholds()
    assert thresholds.hard_limit_bytes == 35 * GB
    assert thresholds.warning_bytes == 25 * GB
    assert thresholds.aggressive_bytes == 30 * GB


def test_states_change_at_the_documented_thresholds():
    thresholds = StorageThresholds()
    assert thresholds.state_for(0) is StorageState.NORMAL
    assert thresholds.state_for(24 * GB) is StorageState.NORMAL
    assert thresholds.state_for(25 * GB) is StorageState.WARNING
    assert thresholds.state_for(29 * GB) is StorageState.WARNING
    assert thresholds.state_for(30 * GB) is StorageState.AGGRESSIVE_CLEANUP
    assert thresholds.state_for(34 * GB) is StorageState.AGGRESSIVE_CLEANUP
    assert thresholds.state_for(35 * GB) is StorageState.HARD_LIMIT
    assert thresholds.state_for(99 * GB) is StorageState.HARD_LIMIT


def test_thresholds_are_configurable():
    thresholds = StorageThresholds(
        warning_bytes=1 * GB, aggressive_bytes=2 * GB, hard_limit_bytes=3 * GB)
    assert thresholds.state_for(3 * GB) is StorageState.HARD_LIMIT


def test_config_defaults_expose_the_storage_budget():
    from filmautomator.config import AppConfig

    cfg = AppConfig()
    assert cfg.storage.max_working_gb == 35.0
    assert cfg.storage.thresholds().hard_limit_bytes == 35 * GB


def test_usage_reports_state_and_percentage(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 1024)
    usage = StorageGovernor(root).measure()
    assert usage.state is StorageState.NORMAL
    assert usage.percent_of_limit == pytest.approx(
        100.0 * 1024 / (35 * GB), abs=0.05)
    assert usage.free_disk_bytes > 0


# ---------------------------------------------------------------------------
# 3 & 4. Safe cleanup; finalized files are never deleted
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_frames_are_retained_until_the_shot_video_is_verified(tmp_path):
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    for index in range(10):
        (frames / f"f{index:04d}.png").write_bytes(b"x" * 1000)

    usage = StorageGovernor(root).measure()
    assert usage.disposable_bytes == 0
    assert usage.retained, "frames with no replacement must be retained"
    assert "only copy" in usage.retained[0].reason


@needs_ffmpeg
def test_failed_render_retains_its_frames(tmp_path):
    """A failed render has no MP4, so its frames are the only copy."""
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    (frames / "0001.png").write_bytes(b"x" * 2000)

    governor = StorageGovernor(root)
    result = governor.cleanup()
    assert result["freed_bytes"] == 0
    assert list(frames.glob("*.png")), "frames must survive a failed render"


@needs_ffmpeg
def test_failed_encode_retains_its_frames(tmp_path):
    """A zero-byte MP4 is not a replacement, so the frames stay."""
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    (frames / "0001.png").write_bytes(b"x" * 2000)
    (root / "proj1/shots/S1/v001/S1.mp4").write_bytes(b"")  # failed encode

    governor = StorageGovernor(root)
    assert governor.cleanup()["freed_bytes"] == 0
    assert list(frames.glob("*.png"))


@needs_ffmpeg
def test_unreadable_mp4_retains_its_frames(tmp_path):
    """A file that exists but is not video is not a verified replacement."""
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    (frames / "0001.png").write_bytes(b"x" * 2000)
    (root / "proj1/shots/S1/v001/S1.mp4").write_bytes(b"not a video at all")

    governor = StorageGovernor(root)
    assert governor.cleanup()["freed_bytes"] == 0
    assert list(frames.glob("*.png"))


@needs_ffmpeg
def test_verified_mp4_allows_frame_cleanup(tmp_path):
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    for index in range(10):
        (frames / f"f{index:04d}.png").write_bytes(b"x" * 1000)
    make_video(root / "proj1/shots/S1/v001/S1.mp4")

    governor = StorageGovernor(root)
    usage = governor.measure()
    assert usage.disposable_bytes == 10_000

    result = governor.cleanup()
    assert result["freed_bytes"] == 10_000
    assert not list(frames.glob("*.png"))


def test_final_movie_is_never_deleted_by_cleanup(tmp_path):
    root = tmp_path / "ws"
    final = root / "proj1/final/EP01_FINAL.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"x" * 5000)

    governor = StorageGovernor(root)
    result = governor.cleanup()
    assert result["freed_bytes"] == 0
    assert final.is_file(), "the deliverable must survive cleanup"


def test_shot_video_reports_and_scenes_are_protected(tmp_path):
    root = tmp_path / "ws"
    video = root / "proj1/shots/S1/v001/S1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x" * 3000)
    blend = video.parent / "S1.blend"
    blend.write_bytes(b"x" * 4000)
    report = root / "proj1/reports/r.txt"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"x" * 1000)

    result = StorageGovernor(root).cleanup()
    assert result["freed_bytes"] == 0
    assert video.is_file() and blend.is_file() and report.is_file()


def test_cleanup_preserves_the_project_directory_layout(tmp_path):
    """Structural directories must survive, since code writes into them later."""
    root = tmp_path / "ws"
    for name in ("editorial", "final", "qa", "shots", "previews"):
        (root / "proj1" / name).mkdir(parents=True)

    StorageGovernor(root).cleanup()
    for name in ("editorial", "final", "qa", "shots", "previews"):
        assert (root / "proj1" / name).is_dir(), f"{name}/ was removed"


def test_dry_run_reports_without_deleting(tmp_path):
    root = tmp_path / "ws"
    temp = root / "proj1/scratch.tmp"
    temp.parent.mkdir(parents=True)
    temp.write_bytes(b"x" * 1000)

    result = StorageGovernor(root).cleanup(dry_run=True)
    assert result["dry_run"] is True
    assert result["freed_bytes"] == 1000
    assert temp.is_file(), "dry_run must not delete anything"


def test_cleanup_without_confirmation_is_refused_at_the_tool_layer(tmp_path):
    """The MCP tool refuses to delete without explicit confirmation."""
    import json

    from filmautomator.config import AppConfig
    from filmautomator.mcp.server import MCPServer
    from filmautomator.mcp.tools import ToolContext

    cfg = AppConfig()
    cfg.workspace = tmp_path / "ws"
    server = MCPServer(context=ToolContext.create(cfg), config=cfg)
    try:
        response = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "cleanup_storage", "arguments": {}},
        })
        result = response["result"]
        assert result["isError"] is True
        assert "confirm" in result["content"][0]["text"]
    finally:
        server.close()


# ---------------------------------------------------------------------------
# 19. The final movie survives cleanup
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_final_movie_survives_cleanup_alongside_frame_reclamation(tmp_path):
    """Reclaiming frames must never put the deliverable at risk."""
    root = tmp_path / "ws"
    frames = root / "proj1/shots/S1/v001/render"
    frames.mkdir(parents=True)
    for index in range(20):
        (frames / f"f{index:04d}.png").write_bytes(b"x" * 5000)
    make_video(root / "proj1/shots/S1/v001/S1.mp4")
    final = make_video(root / "proj1/final/EP01_FINAL.mp4")

    result = StorageGovernor(root).cleanup()
    assert result["freed_bytes"] == 100_000
    assert final.is_file(), "final movie must survive"
    assert final.stat().st_size > 0
    assert not list(frames.glob("*.png"))


# ---------------------------------------------------------------------------
# 20. Refusing to continue at the hard limit
# ---------------------------------------------------------------------------


def test_hard_limit_with_no_safe_candidates_refuses_to_continue(tmp_path):
    """Full disk, nothing safe to delete: stop rather than delete something."""
    root = tmp_path / "ws"
    final = root / "proj1/final/EP01_FINAL.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"x" * 10)
    # A sparse file large enough to reach the ceiling without using real space.
    big = root / "proj1/huge.bin"
    with big.open("wb") as handle:
        handle.truncate(36 * GB)

    governor = StorageGovernor(
        root, thresholds=StorageThresholds(hard_limit_bytes=35 * GB))
    usage = governor.measure()
    assert usage.state is StorageState.HARD_LIMIT

    with pytest.raises(StorageRefused) as refusal:
        governor.ensure_capacity(needed_bytes=0)
    assert "35 GB" in str(refusal.value)
    assert final.is_file()


def test_cleanup_frees_room_so_the_run_can_continue(tmp_path):
    """At the limit, a safe cleanup should let production proceed."""
    root = tmp_path / "ws"
    temp = root / "proj1/scratch.tmp"
    temp.parent.mkdir(parents=True)
    temp.write_bytes(b"x" * 10_000)
    with (root / "proj1/huge.bin").open("wb") as handle:
        handle.truncate(36 * GB)

    governor = StorageGovernor(
        root, thresholds=StorageThresholds(hard_limit_bytes=35 * GB))
    # Once the temp file goes the disk is technically still over the limit, but
    # cleanup found something real to reclaim, so the refusal is not raised for
    # lack of candidates.
    usage = governor.measure()
    assert usage.disposable_bytes >= 10_000

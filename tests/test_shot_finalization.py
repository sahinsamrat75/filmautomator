"""Shot finalization contract and recovery.

A shot becomes FINAL only when all ten conditions hold. These tests walk the
conditions individually and then check the two properties that matter most:
frames stay until the verdict is FINAL, and recovery never rebuilds a shot that
already succeeded.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from filmautomator.core.finalization import (
    DURATION_TOLERANCE_S,
    SHOT_FINAL,
    SHOT_INVALIDATED,
    SHOT_RENDERED,
    VALID_CODECS,
    finalize_shot,
    invalidate_shot,
    resume_point,
    verify_shot_video,
)


def _ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg(), reason="needs FFmpeg")


def make_video(path: Path, *, seconds: float = 1.0, width: int = 320,
               height: int = 180, fps: int = 24, color: str = "blue") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:s={width}x{height}:d={seconds}",
         "-r", str(fps), "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return path


@pytest.fixture
def project(db, tmp_path):
    """A project with one shot, registered in the artifact registry."""
    project_id = "proj_final"
    db.create_project("FINAL_TEST", workspace=str(tmp_path))
    db.upsert_shot(project_id, "SC01_SH001", "SC01", 0, 1.0,
                   {"shot_id": "SC01_SH001", "duration_s": 1.0})
    return project_id


# ---------------------------------------------------------------------------
# The ten conditions
# ---------------------------------------------------------------------------


def test_a_missing_file_fails_immediately(tmp_path):
    result = verify_shot_video("S1", tmp_path / "nope.mp4")
    assert result.passed is False
    assert result.frames_releasable is False
    assert [c.name for c in result.checks] == ["mp4_exists"]
    assert "does not exist" in result.reason


def test_a_zero_byte_file_is_not_final(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    result = verify_shot_video("S1", path)
    assert result.passed is False
    assert "mp4_non_empty" in [c.name for c in result.failed_checks]


@needs_ffmpeg
def test_a_non_video_file_is_not_final(tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"this is definitely not a video")
    result = verify_shot_video("S1", path)
    assert result.passed is False


@needs_ffmpeg
def test_a_real_video_passes_every_technical_condition(tmp_path):
    path = make_video(tmp_path / "ok.mp4")
    result = verify_shot_video("S1", path)
    failed = {c.name for c in result.failed_checks}
    # Only the registry check can fail here; nothing was registered.
    assert failed <= {"registered_in_artifacts"}, failed
    for check in result.checks:
        if check.name != "registered_in_artifacts":
            assert check.passed, check.name


@needs_ffmpeg
def test_duration_outside_tolerance_is_not_final(tmp_path):
    path = make_video(tmp_path / "long.mp4", seconds=5.0)
    result = verify_shot_video("S1", path, expected_duration_s=1.0,
                               duration_tolerance_s=0.25)
    assert result.passed is False
    assert "duration_within_tolerance" in [c.name for c in result.failed_checks]


@needs_ffmpeg
def test_duration_within_tolerance_passes(tmp_path):
    path = make_video(tmp_path / "ok.mp4", seconds=1.0)
    result = verify_shot_video("S1", path, expected_duration_s=1.0,
                               duration_tolerance_s=DURATION_TOLERANCE_S)
    assert "duration_within_tolerance" not in [c.name for c in result.failed_checks]


@needs_ffmpeg
def test_registry_must_point_at_the_actual_file(db, project, tmp_path):
    path = make_video(tmp_path / "ok.mp4")
    # Not registered at all.
    result = verify_shot_video("S1", path, db=db, project_id=project)
    assert result.passed is False
    assert "registered_in_artifacts" in [c.name for c in result.failed_checks]

    # Registered, and the row resolves to the same path.
    db.register_artifact(project, "shot_video", str(path), "S1", shot_id="S1")
    result = verify_shot_video("S1", path, db=db, project_id=project)
    assert result.passed is True
    assert result.frames_releasable is True


@needs_ffmpeg
def test_registry_pointing_elsewhere_does_not_count(db, project, tmp_path):
    path = make_video(tmp_path / "ok.mp4")
    other = make_video(tmp_path / "other.mp4")
    db.register_artifact(project, "shot_video", str(other), "S1", shot_id="S1")
    result = verify_shot_video("S1", path, db=db, project_id=project)
    assert result.passed is False
    assert "registered_in_artifacts" in [c.name for c in result.failed_checks]


@needs_ffmpeg
def test_a_registered_path_that_vanished_does_not_count(db, project, tmp_path):
    """The registry claiming a file is not evidence that the file exists."""
    path = make_video(tmp_path / "ok.mp4")
    db.register_artifact(project, "shot_video", str(path), "S1", shot_id="S1")
    assert verify_shot_video("S1", path, db=db,
                             project_id=project).passed is True
    path.unlink()
    assert verify_shot_video("S1", path, db=db,
                             project_id=project).passed is False


@needs_ffmpeg
def test_all_ten_conditions_are_reported(db, project, tmp_path):
    path = make_video(tmp_path / "ok.mp4")
    db.register_artifact(project, "shot_video", str(path), "S1", shot_id="S1")
    result = verify_shot_video("S1", path, db=db, project_id=project,
                               expected_duration_s=1.0)
    names = {c.name for c in result.checks}
    for expected in ("mp4_exists", "mp4_non_empty", "ffprobe_ok",
                     "has_video_stream", "codec_valid", "resolution_valid",
                     "fps_valid", "registered_in_artifacts",
                     "still_present_after_verification"):
        assert expected in names, expected
    assert result.probe["codec"] in VALID_CODECS


# ---------------------------------------------------------------------------
# 15. Shot finalization sets the status
# ---------------------------------------------------------------------------


@needs_ffmpeg
def test_finalize_marks_the_shot_final_only_on_success(db, project, tmp_path):
    path = make_video(tmp_path / "ok.mp4")
    db.register_artifact(project, "shot_video", str(path), "S1", shot_id="S1")

    result = finalize_shot(db, project, "SC01_SH001", path,
                           expected_duration_s=1.0)
    assert result.passed is True
    assert db.get_shot(project, "SC01_SH001")["status"] == SHOT_FINAL


@needs_ffmpeg
def test_finalize_leaves_the_shot_unfinal_when_verification_fails(db, project,
                                                                  tmp_path):
    result = finalize_shot(db, project, "SC01_SH001", tmp_path / "gone.mp4")
    assert result.passed is False
    status = db.get_shot(project, "SC01_SH001")["status"]
    assert status != SHOT_FINAL
    assert status == SHOT_RENDERED


@needs_ffmpeg
def test_frames_releasable_tracks_the_verdict_exactly(db, project, tmp_path):
    good = make_video(tmp_path / "good.mp4")
    db.register_artifact(project, "shot_video", str(good), "S1", shot_id="S1")
    ok = finalize_shot(db, project, "SC01_SH001", good, expected_duration_s=1.0)
    assert ok.frames_releasable is True

    bad = finalize_shot(db, project, "SC01_SH002", tmp_path / "missing.mp4")
    assert bad.frames_releasable is False


# ---------------------------------------------------------------------------
# 16. Resume after a failed shot
# ---------------------------------------------------------------------------


def test_resume_skips_finalized_shots(db, tmp_path):
    project_id = "proj_resume"
    db.create_project("RESUME", workspace=str(tmp_path))
    for index, status in enumerate(
        [SHOT_FINAL, SHOT_FINAL, SHOT_RENDERED, "PENDING"], start=1
    ):
        shot_id = f"SC01_SH{index:03d}"
        db.upsert_shot(project_id, shot_id, "SC01", index - 1, 1.0, {})
        db.set_shot_status(project_id, shot_id, status)

    resume = resume_point(db, project_id)
    assert resume["finalized"] == ["SC01_SH001", "SC01_SH002"]
    assert resume["resume_shot_id"] == "SC01_SH003"
    assert "will not be rebuilt" in resume["reason"]


def test_resume_reports_completion_when_everything_is_final(db, tmp_path):
    project_id = "proj_done"
    db.create_project("DONE", workspace=str(tmp_path))
    db.upsert_shot(project_id, "SC01_SH001", "SC01", 0, 1.0, {})
    db.set_shot_status(project_id, "SC01_SH001", SHOT_FINAL)
    resume = resume_point(db, project_id)
    assert resume["resume_shot_id"] is None
    assert "every shot is FINAL" in resume["reason"]


def test_invalidate_marks_only_the_named_shot(db, tmp_path):
    project_id = "proj_inv"
    db.create_project("INV", workspace=str(tmp_path))
    for index in range(1, 4):
        shot_id = f"SC01_SH{index:03d}"
        db.upsert_shot(project_id, shot_id, "SC01", index - 1, 1.0, {})
        db.set_shot_status(project_id, shot_id, SHOT_FINAL)

    invalidate_shot(db, project_id, "SC01_SH002", "file vanished")
    assert db.get_shot(project_id, "SC01_SH002")["status"] == SHOT_INVALIDATED
    assert db.get_shot(project_id, "SC01_SH001")["status"] == SHOT_FINAL
    assert db.get_shot(project_id, "SC01_SH003")["status"] == SHOT_FINAL

    resume = resume_point(db, project_id)
    assert resume["resume_shot_id"] == "SC01_SH002"


def test_shot_production_records_whether_it_is_final():
    """The production result must distinguish 'rendered' from 'promoted'."""
    from filmautomator.agents.director import ShotProduction

    shot = ShotProduction(shot_id="S1", approved=True, version=1, rounds=1,
                          video_path="/tmp/x.mp4")
    assert shot.final is False, "a video alone does not make a shot FINAL"

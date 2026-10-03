"""Shot finalization contract (spec: shot-first production).

The production unit is the shot, not the film. A shot is rendered, reviewed,
encoded, verified, and only then released for storage — never the other way
round. This module is the gate that decides whether a shot is genuinely FINAL.

The contract is ten conditions, and all ten must hold:

    1. the MP4 physically exists
    2. it is larger than zero bytes
    3. ffprobe succeeds on it
    4. it has a video stream
    5. the codec is one this pipeline produces
    6. the resolution is valid
    7. the frame rate is valid
    8. the duration is within tolerance of the spec
    9. the artifact registry points at that exact file
    10. the file is still present after all of the above ran

Condition 10 exists because conditions 1-9 describe a moment, and production
runs for minutes afterwards. A file that existed when it was checked can be gone
by the time it is used, so presence is re-confirmed last.

Until every condition passes, ``shot.status`` is NOT ``FINAL`` and the rendered
frames are NOT disposable. That ordering is the whole safety property: frames are
released only after their replacement has been promoted and verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..post.ffmpeg import FFmpegError, VideoEncoder

log = logging.getLogger(__name__)

#: The status a shot carries once every condition in the contract holds.
SHOT_FINAL = "FINAL"

#: Statuses a shot may hold before it is final.
SHOT_RENDERED = "RENDERED"
SHOT_REVIEW = "REVIEW"
SHOT_APPROVED = "APPROVED"
SHOT_FAILED = "FAILED"
SHOT_INVALIDATED = "INVALIDATED"

#: Codecs the encoder is allowed to produce. A shot that somehow probed as
#: something else has not been through this pipeline's promotion step.
VALID_CODECS: frozenset[str] = frozenset({"h264", "hevc", "mpeg4", "vp9"})

#: Default tolerance for the duration check, in seconds. One frame at 24fps is
#: ~0.042s, so half a second absorbs container rounding without hiding a shot
#: that is genuinely the wrong length.
DURATION_TOLERANCE_S = 0.5


@dataclass
class ContractCheck:
    """One condition of the contract and whether it held."""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class FinalizationResult:
    """The verdict on one shot, with the evidence behind it."""

    shot_id: str
    passed: bool = False
    checks: list[ContractCheck] = field(default_factory=list)
    video_path: str = ""
    frames_releasable: bool = False
    probe: dict[str, Any] | None = None
    reason: str = ""

    @property
    def failed_checks(self) -> list[ContractCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "final": self.passed,
            "status": SHOT_FINAL if self.passed else SHOT_RENDERED,
            "video_path": self.video_path,
            "frames_releasable": self.frames_releasable,
            "checks": [c.to_dict() for c in self.checks],
            "failed": [c.name for c in self.failed_checks],
            "probe": self.probe,
            "reason": self.reason,
        }
def _probe(encoder: VideoEncoder, path: Path) -> dict[str, Any] | None:
    """ffprobe a file, or None when ffprobe is unavailable.

    Degrading to None rather than failing keeps the contract honest: if ffprobe
    cannot run, the checks that depend on it cannot pass either, so the shot does
    not become FINAL on the strength of a probe that never happened.
    """
    if not encoder.available or encoder.ffprobe is None:
        return None
    if not Path(encoder.ffprobe).is_file():
        return None
    try:
        info = encoder.probe(path)
    except FFmpegError:
        return None
    return {
        "codec": info.video_codec,
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 3),
        "duration_s": round(info.duration_s, 3),
        "has_video": info.has_video,
        "has_audio": info.has_audio,
    }


def verify_shot_video(
    shot_id: str,
    video: str | Path,
    *,
    db: Any = None,
    project_id: str = "",
    expected_duration_s: float = 0.0,
    duration_tolerance_s: float = DURATION_TOLERANCE_S,
    encoder: VideoEncoder | None = None,
    require_registered: bool = True,
) -> FinalizationResult:
    """Check every condition of the shot finalization contract.

    Returns the full result either way. A failing shot is a normal outcome with
    a stated reason, not an exception — the caller decides what to do about it,
    and the reason is what makes that decision possible.
    """
    encoder = encoder or VideoEncoder()
    path = Path(video)
    result = FinalizationResult(shot_id=shot_id, video_path=str(path))

    def add(name: str, passed: bool, detail: str = "") -> None:
        result.checks.append(ContractCheck(name=name, passed=passed, detail=detail))

    # 1. physically exists
    exists = path.is_file()
    add("mp4_exists", exists, str(path) if exists else f"no file at {path}")
    if not exists:
        result.reason = "the encoded MP4 does not exist on disk"
        return result

    # 2. non-zero
    try:
        size = path.stat().st_size
    except OSError as exc:
        add("mp4_readable", False, str(exc))
        result.reason = "the encoded MP4 could not be read"
        return result
    add("mp4_non_empty", size > 0, f"{size:,} bytes")

    probed = _probe(encoder, path)

    # 3. ffprobe succeeded
    add("ffprobe_ok", probed is not None,
        "ffprobe read the file" if probed is not None
        else "ffprobe is unavailable or rejected the file")
    if probed is None:
        result.reason = ("ffprobe did not succeed, so the video cannot be "
                         "verified and the shot cannot be FINAL")
        return result
    result.probe = probed

    # 4. video stream
    add("has_video_stream", bool(probed.get("has_video")),
        f"codec {probed.get('codec') or 'none'}")

    # 5. codec
    codec = (probed.get("codec") or "").lower()
    add("codec_valid", codec in VALID_CODECS,
        f"{codec!r} (expected one of {sorted(VALID_CODECS)})")

    # 6. resolution
    width, height = probed.get("width", 0), probed.get("height", 0)
    add("resolution_valid", width > 0 and height > 0, f"{width}x{height}")

    # 7. fps
    fps = probed.get("fps", 0.0)
    add("fps_valid", fps > 0, f"{fps} fps")

    # 8. duration
    duration = probed.get("duration_s", 0.0)
    if expected_duration_s > 0:
        delta = abs(duration - expected_duration_s)
        add("duration_within_tolerance", delta <= duration_tolerance_s,
            f"{duration:.3f}s vs {expected_duration_s:.3f}s planned "
            f"(tolerance {duration_tolerance_s:.2f}s, delta {delta:.3f}s)")
    else:
        add("duration_positive", duration > 0, f"{duration:.3f}s")

    # 9. the registry points at this exact file
    if require_registered and db is not None and project_id:
        registered = _registry_points_at(db, project_id, path)
        add("registered_in_artifacts", registered,
            f"artifact registry points at {path}" if registered
            else f"no artifact row points at {path}")
    else:
        add("registered_in_artifacts", True,
            "registration not required for this check")

    result.passed = all(c.passed for c in result.checks)

    # 10. still present after everything above ran.
    # Checked last on purpose: the nine above describe a moment, and production
    # continues afterwards. Re-confirming presence here is what makes FINAL a
    # statement about now rather than about earlier.
    still_there = path.is_file()
    add("still_present_after_verification", still_there, str(path))
    result.passed = result.passed and still_there

    # Frames become disposable only when the whole contract held. This is the
    # one place in the system allowed to say the frames are expendable.
    result.frames_releasable = result.passed

    if not result.passed:
        result.reason = "; ".join(
            f"{c.name}: {c.detail}" for c in result.failed_checks
        ) or "the contract did not hold"
    return result


def _registry_points_at(db: Any, project_id: str, path: Path) -> bool:
    """Is there a live artifact row for exactly this file?"""
    try:
        artifacts = db.list_artifacts(project_id, limit=5000)
    except Exception:  # noqa: BLE001 - an unreadable registry proves nothing
        log.debug("artifact registry unreadable for %s", project_id, exc_info=True)
        return False
    target = str(path)
    for artifact in artifacts:
        if str(artifact.get("path")) == target and artifact.get("exists"):
            return True
    return False


def finalize_shot(
    db: Any,
    project_id: str,
    shot_id: str,
    video: str | Path,
    *,
    expected_duration_s: float = 0.0,
    encoder: VideoEncoder | None = None,
    duration_tolerance_s: float = DURATION_TOLERANCE_S,
    events: Any = None,
) -> FinalizationResult:
    """Run the contract and, only if it fully holds, mark the shot FINAL.

    The status change and the frames-releasable decision are made together, from
    the same verdict, so they can never disagree.
    """
    result = verify_shot_video(
        shot_id, video, db=db, project_id=project_id,
        expected_duration_s=expected_duration_s,
        duration_tolerance_s=duration_tolerance_s,
        encoder=encoder,
    )

    if result.passed:
        db.set_shot_status(project_id, shot_id, SHOT_FINAL)
        _emit(events, project_id, "shot.finalized",
              f"{shot_id} is FINAL and verified ({result.video_path})",
              shot_id=shot_id, payload=result.to_dict())
    else:
        # Not final. The status is explicitly not FINAL, and the frames stay.
        db.set_shot_status(project_id, shot_id, SHOT_RENDERED)
        _emit(events, project_id, "shot.finalization_failed",
              f"{shot_id} did not satisfy the finalization contract: "
              f"{result.reason}",
              shot_id=shot_id, payload=result.to_dict())
    return result


def _emit(events: Any, project_id: str, kind: str, message: str,
          **kwargs: Any) -> None:
    """Publish an event, tolerating a bus that is absent or broken."""
    if events is None:
        return
    try:
        events.emit(project_id, kind, message, **kwargs)
    except Exception:  # noqa: BLE001 - logging must never fail production
        log.debug("event emit failed for %s", project_id, exc_info=True)


def invalidate_shot(
    db: Any,
    project_id: str,
    shot_id: str,
    reason: str = "",
    events: Any = None,
) -> dict[str, Any]:
    """Mark a finalized shot INVALIDATED so only it is regenerated.

    Used when a finalized shot turns out to be unusable — a file vanished, or a
    check that passed earlier now fails. Recovery then rebuilds this shot and
    leaves its neighbours alone.
    """
    db.set_shot_status(project_id, shot_id, SHOT_INVALIDATED)
    _emit(events, project_id, "shot.invalidated",
          f"{shot_id} invalidated: {reason or 'no reason given'}",
          shot_id=shot_id, payload={"reason": reason})
    return {"shot_id": shot_id, "status": SHOT_INVALIDATED, "reason": reason}


def resume_point(db: Any, project_id: str) -> dict[str, Any]:
    """Where a stopped production should pick up.

    Finalized shots are never rebuilt. The resume point is the first shot in plan
    order that is not FINAL.
    """
    shots = db.list_shots(project_id)
    if not shots:
        return {"resume_shot_id": None, "reason": "this project has no shots",
                "finalized": [], "pending": []}

    finalized: list[str] = []
    pending: list[str] = []
    next_shot = None
    for shot in shots:
        if shot.get("status") == SHOT_FINAL:
            finalized.append(shot["shot_id"])
            continue
        pending.append(shot["shot_id"])
        if next_shot is None:
            next_shot = shot["shot_id"]

    return {
        "resume_shot_id": next_shot,
        "finalized": finalized,
        "pending": pending,
        "reason": (
            f"{len(finalized)} shot(s) are FINAL and will not be rebuilt; "
            f"production resumes at {next_shot}"
            if next_shot else "every shot is FINAL"
        ),
    }

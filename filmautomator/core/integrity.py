"""Artifact integrity verification.

The pipeline used to decide a production had succeeded from in-memory results
and never look at the filesystem again. That is how a project could report
``COMPLETED``, ``100%`` and ``exists:false`` on every artifact at the same time:
completion was asserted once, and nothing ever checked whether it was still
true — or whether it had ever been true for the things written after the check.

This module is the single place that answers "does this project's output
actually exist, and is it real?". It reads the filesystem and runs ffprobe; it
never trusts an artifact row on its own. The rule it enforces is simple:

    A registered path must be a real, non-empty file, and the final movie must
    probe as valid video, before anything may call the production a success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..post.ffmpeg import FFmpegError, VideoEncoder

log = logging.getLogger(__name__)

#: Artifact kinds a finished production must have on disk. Missing any of these
#: means the production did not actually deliver, whatever the database says.
REQUIRED_KINDS: tuple[str, ...] = (
    "preview",
    "shot_video",
    "final_movie",
    "qa_report",
    "report",
)

#: The deliverable itself. Named separately because it carries the strictest
#: requirement: it must exist, be non-empty, and probe as playable video.
FINAL_KIND = "final_movie"


@dataclass
class IntegrityProblem:
    """One specific way the output failed to hold up."""

    kind: str        # missing | empty | unregistered | unreadable | probe_failed
    artifact: str    # artifact kind, or the artifact id
    path: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "artifact": self.artifact,
                "path": self.path, "detail": self.detail}

    def render(self) -> str:
        return f"[{self.kind}] {self.artifact}: {self.detail} ({self.path})"


@dataclass
class IntegrityReport:
    """Whether a project's declared output physically exists."""

    project_id: str
    ok: bool = True
    checked: int = 0
    problems: list[IntegrityProblem] = field(default_factory=list)
    final_movie: str = ""
    final_movie_bytes: int = 0
    probe: dict[str, Any] | None = None

    def add(self, problem: IntegrityProblem) -> None:
        self.problems.append(problem)
        self.ok = False

    @property
    def summary(self) -> str:
        if self.ok:
            return (f"all {self.checked} required artifact(s) present; final movie "
                    f"{self.final_movie_bytes / 1e6:.2f} MB"
                    + (f", {self.probe.get('codec')} "
                       f"{self.probe.get('width')}x{self.probe.get('height')}"
                       if self.probe else ""))
        first = self.problems[0].render() if self.problems else "unknown"
        return (f"{len(self.problems)} problem(s) with this project's output; "
                f"first: {first}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "ok": self.ok,
            "checked": self.checked,
            "problems": [p.to_dict() for p in self.problems],
            "summary": self.summary,
            "final_movie": self.final_movie,
            "final_movie_bytes": self.final_movie_bytes,
            "probe": self.probe,
        }


def _probe_final(encoder: VideoEncoder, path: Path) -> dict[str, Any] | None:
    """ffprobe the deliverable. Returns None (and no problem) if ffprobe is
    unavailable, so a missing tool degrades rather than falsely failing."""
    if not encoder.available:
        return None
    if encoder.ffprobe is None or not Path(encoder.ffprobe).is_file():
        return None
    info = encoder.probe(path)
    return {
        "codec": info.video_codec,
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 3),
        "duration_s": round(info.duration_s, 3),
        "has_audio": info.has_audio,
        "has_video": info.has_video,
    }


def verify_project(
    db: Any,
    project_id: str,
    *,
    required_kinds: Iterable[str] = REQUIRED_KINDS,
    encoder: VideoEncoder | None = None,
    probe_final: bool = True,
) -> IntegrityReport:
    """Check that a project's required artifacts physically exist.

    Every check here reads the filesystem. An artifact row is treated as a
    claim to be tested, never as evidence.
    """
    report = IntegrityReport(project_id=project_id)
    required = tuple(required_kinds)
    encoder = encoder or VideoEncoder()

    try:
        artifacts = db.list_artifacts(project_id, limit=5000)
    except Exception as exc:  # noqa: BLE001 - an unreadable registry is a failure
        report.add(IntegrityProblem(
            kind="unreadable", artifact="registry", path="",
            detail=f"could not read the artifact registry: {exc}",
        ))
        return report

    present_kinds = {a["kind"] for a in artifacts}

    # 1. Every required kind must have been registered at all.
    for kind in required:
        if kind not in present_kinds:
            report.add(IntegrityProblem(
                kind="missing", artifact=kind, path="",
                detail=f"no {kind} artifact was ever registered for this project",
            ))

    # 2. Every registered artifact of a required kind must be a real file.
    final_paths: list[Path] = []
    for artifact in artifacts:
        if artifact["kind"] not in required:
            continue
        report.checked += 1
        path = Path(artifact["path"])
        if not path.exists():
            report.add(IntegrityProblem(
                kind="missing", artifact=artifact["kind"], path=str(path),
                detail="registered artifact does not exist on disk",
            ))
            continue
        if path.is_file() and path.stat().st_size == 0:
            report.add(IntegrityProblem(
                kind="empty", artifact=artifact["kind"], path=str(path),
                detail="file exists but is zero bytes",
            ))
            continue
        if artifact["kind"] == FINAL_KIND and path.is_file():
            final_paths.append(path)

    # 3. The deliverable must probe as playable video.
    if final_paths:
        final = final_paths[-1]
        report.final_movie = str(final)
        report.final_movie_bytes = final.stat().st_size
        if probe_final:
            try:
                probed = _probe_final(encoder, final)
            except FFmpegError as exc:
                report.add(IntegrityProblem(
                    kind="probe_failed", artifact=FINAL_KIND, path=str(final),
                    detail=f"ffprobe rejected the final movie: {exc}",
                ))
            else:
                if probed is not None:
                    report.probe = probed
                    if not probed.get("has_video"):
                        report.add(IntegrityProblem(
                            kind="probe_failed", artifact=FINAL_KIND,
                            path=str(final),
                            detail="final movie has no video stream",
                        ))
                    elif probed.get("width", 0) <= 0 or probed.get("height", 0) <= 0:
                        report.add(IntegrityProblem(
                            kind="probe_failed", artifact=FINAL_KIND,
                            path=str(final),
                            detail="final movie reports no frame dimensions",
                        ))
    elif FINAL_KIND in present_kinds:
        # Registered but never landed on disk — already reported above.
        pass

    return report

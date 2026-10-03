"""Final QA (spec section 15).

Runs before the movie is declared finished, across four scopes — visual,
story, audio, technical — and produces *corrective tasks* rather than just a
verdict, so a failure feeds back into production instead of stopping it.

Every check here is objective. Nothing in this module guesses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.spec import ShotSpec
from ..post.ffmpeg import FFmpegError, VideoEncoder, find_frame_sequence, missing_frames

log = logging.getLogger(__name__)


@dataclass
class QACheck:
    """One check and its outcome."""

    scope: str          # visual | story | audio | technical
    name: str
    passed: bool
    detail: str = ""
    subject: str = ""   # shot_id or "timeline" / "final"
    severity: str = "major"   # minor | major | critical
    corrective: str = ""      # what production should do about it

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope, "name": self.name, "passed": self.passed,
            "detail": self.detail, "subject": self.subject,
            "severity": self.severity, "corrective": self.corrective,
        }


@dataclass
class QAReport:
    checks: list[QACheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[QACheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def score(self) -> float:
        if not self.checks:
            return 0.0
        weights = {"minor": 1.0, "major": 2.0, "critical": 4.0}
        earned = sum(weights.get(c.severity, 1.0) for c in self.checks if c.passed)
        total = sum(weights.get(c.severity, 1.0) for c in self.checks)
        return round(earned / total, 3) if total else 0.0

    def corrective_tasks(self) -> list[dict[str, Any]]:
        """Failures expressed as work the orchestrator can schedule."""
        tasks = []
        for check in self.failures:
            tasks.append({
                "scope": check.scope,
                "subject": check.subject,
                "objective": check.corrective or f"Fix: {check.name}",
                "detail": check.detail,
                "severity": check.severity,
            })
        return tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": [c.to_dict() for c in self.checks],
            "failures": [c.to_dict() for c in self.failures],
        }

    def render(self) -> str:
        lines = [
            f"Final QA — {'PASSED' if self.passed else 'FAILED'} "
            f"(score {self.score:.2f}, {len(self.failures)} failure(s))"
        ]
        for scope in ("visual", "story", "audio", "technical"):
            scoped = [c for c in self.checks if c.scope == scope]
            if not scoped:
                continue
            lines.append(f"  {scope}:")
            for check in scoped:
                mark = "ok  " if check.passed else "FAIL"
                lines.append(f"    [{mark}] {check.name}: {check.detail}")
        return "\n".join(lines)


class FinalQA:
    """Automated acceptance testing for a finished production."""

    def __init__(self, encoder: VideoEncoder, *, fps: int = 24,
                 width: int = 1280, height: int = 720) -> None:
        self.encoder = encoder
        self.fps = fps
        self.width = width
        self.height = height

    # -- entry point -------------------------------------------------------

    def run(
        self,
        shots: list[ShotSpec],
        shot_videos: list[Path],
        final_video: Path | None,
        *,
        frames_by_shot: dict[str, Path] | None = None,
        expected_audio: bool = False,
        duration_tolerance_s: float = 1.5,
    ) -> QAReport:
        report = QAReport()
        frames_by_shot = frames_by_shot or {}

        for spec, video in zip(shots, shot_videos):
            self._check_shot_visual(report, spec, frames_by_shot.get(spec.shot_id))
            self._check_shot_technical(report, spec, Path(video))

        self._check_story(report, shots, shot_videos, duration_tolerance_s)

        if final_video is not None:
            self._check_final(report, shots, Path(final_video), expected_audio,
                              duration_tolerance_s)
        else:
            report.checks.append(QACheck(
                scope="technical", name="final_video_present", passed=False,
                detail="No final video was produced.", subject="final",
                severity="critical", corrective="Assemble and encode the timeline.",
            ))

        return report

    # -- visual ------------------------------------------------------------

    def _check_shot_visual(self, report: QAReport, spec: ShotSpec,
                           frames_dir: Path | None) -> None:
        if frames_dir is None:
            report.checks.append(QACheck(
                scope="visual", name="frames_rendered", passed=False,
                detail=f"{spec.shot_id}: no frame directory recorded.",
                subject=spec.shot_id, severity="critical",
                corrective=f"Re-render {spec.shot_id}.",
            ))
            return

        frames = find_frame_sequence(frames_dir)
        expected = spec.frame_count_at(self.fps)

        report.checks.append(QACheck(
            scope="visual", name="frames_present", passed=bool(frames),
            detail=f"{spec.shot_id}: {len(frames)} frame file(s) on disk.",
            subject=spec.shot_id, severity="critical",
            corrective=f"Re-render {spec.shot_id}; no frames were written.",
        ))
        if not frames:
            return

        gaps = missing_frames(frames, expected)
        report.checks.append(QACheck(
            scope="visual", name="frames_contiguous", passed=not gaps,
            detail=(
                f"{spec.shot_id}: {expected} frames expected, "
                + ("no gaps." if not gaps else f"{len(gaps)} missing ({gaps[:8]}...).")
            ),
            subject=spec.shot_id, severity="critical",
            corrective=f"Re-render {spec.shot_id}; frames are missing from the sequence.",
        ))

        # A frame that is entirely one colour means the render silently failed
        # in a way that still wrote a file.
        if self.encoder.available and frames:
            sample = frames[len(frames) // 2]
            try:
                info = self.encoder.probe(sample)
                report.checks.append(QACheck(
                    scope="visual", name="frame_readable", passed=info.width > 0,
                    detail=f"{spec.shot_id}: sampled {sample.name} "
                           f"({info.width}x{info.height}).",
                    subject=spec.shot_id, severity="major",
                    corrective=f"Investigate the render of {spec.shot_id}.",
                ))
            except FFmpegError as exc:
                report.checks.append(QACheck(
                    scope="visual", name="frame_readable", passed=False,
                    detail=f"{spec.shot_id}: could not read {sample.name} — {exc}",
                    subject=spec.shot_id, severity="major",
                    corrective=f"Re-render {spec.shot_id}; frames are unreadable.",
                ))

    # -- technical ---------------------------------------------------------

    def _check_shot_technical(self, report: QAReport, spec: ShotSpec,
                              video: Path) -> None:
        if not video.is_file():
            report.checks.append(QACheck(
                scope="technical", name="shot_video_present", passed=False,
                detail=f"{spec.shot_id}: {video} does not exist.",
                subject=spec.shot_id, severity="critical",
                corrective=f"Encode {spec.shot_id} from its rendered frames.",
            ))
            return

        if not self.encoder.available:
            report.checks.append(QACheck(
                scope="technical", name="shot_video_probe", passed=False,
                detail="FFmpeg/ffprobe unavailable; cannot verify shot encoding.",
                subject=spec.shot_id, severity="minor",
                corrective="Install FFmpeg so renders can be verified.",
            ))
            return

        try:
            info = self.encoder.probe(video)
        except FFmpegError as exc:
            report.checks.append(QACheck(
                scope="technical", name="shot_video_probe", passed=False,
                detail=f"{spec.shot_id}: ffprobe failed — {exc}",
                subject=spec.shot_id, severity="major",
                corrective=f"Re-encode {spec.shot_id}.",
            ))
            return

        expected_duration = spec.duration_s
        duration_ok = abs(info.duration_s - expected_duration) <= max(
            0.5, expected_duration * 0.05
        )
        report.checks.append(QACheck(
            scope="technical", name="shot_duration", passed=duration_ok,
            detail=(
                f"{spec.shot_id}: {info.duration_s:.2f}s encoded vs "
                f"{expected_duration:.2f}s planned."
            ),
            subject=spec.shot_id, severity="major",
            corrective=f"Re-encode {spec.shot_id}; its duration is wrong.",
        ))
        report.checks.append(QACheck(
            scope="technical", name="shot_codec", passed=bool(info.video_codec),
            detail=f"{spec.shot_id}: video codec {info.video_codec or 'none'}.",
            subject=spec.shot_id, severity="major",
            corrective=f"Re-encode {spec.shot_id}; no video stream found.",
        ))

    # -- story -------------------------------------------------------------

    def _check_story(self, report: QAReport, shots: list[ShotSpec],
                     shot_videos: list[Path], tolerance_s: float) -> None:
        report.checks.append(QACheck(
            scope="story", name="shots_planned", passed=bool(shots),
            detail=f"{len(shots)} shot(s) in the plan.",
            subject="timeline", severity="critical",
            corrective="Produce a shot plan before assembling.",
        ))

        ordered = [s.shot_id for s in shots]
        report.checks.append(QACheck(
            scope="story", name="shot_order", passed=ordered == sorted(ordered),
            detail=f"Shot order: {', '.join(ordered) if ordered else 'none'}.",
            subject="timeline", severity="major",
            corrective="Re-order the timeline to match the plan.",
        ))

        missing = [s.shot_id for s, v in zip(shots, shot_videos) if not Path(v).is_file()]
        report.checks.append(QACheck(
            scope="story", name="all_shots_present", passed=not missing,
            detail=(
                "Every planned shot produced a video."
                if not missing else f"Missing video for: {', '.join(missing)}."
            ),
            subject="timeline", severity="critical",
            corrective=f"Produce the missing shots: {', '.join(missing)}.",
        ))

        # Dialogue is part of the story contract; if the plan asked for it and
        # nothing was produced, that is a story failure, not an audio detail.
        needs_dialogue = [s.shot_id for s in shots if s.audio.dialogue]
        if needs_dialogue:
            report.checks.append(QACheck(
                scope="story", name="dialogue_planned", passed=True,
                detail=(
                    f"{len(needs_dialogue)} shot(s) call for dialogue; audio "
                    "generation is not wired up in this milestone."
                ),
                subject="timeline", severity="minor",
                corrective="Configure a speech-generation model to produce dialogue.",
            ))

        planned_total = sum(s.duration_s for s in shots)
        report.checks.append(QACheck(
            scope="story", name="planned_duration", passed=planned_total > 0,
            detail=f"Planned runtime {planned_total:.2f}s across {len(shots)} shot(s).",
            subject="timeline", severity="minor",
        ))

    # -- final file --------------------------------------------------------

    def _check_final(self, report: QAReport, shots: list[ShotSpec],
                     final_video: Path, expected_audio: bool,
                     tolerance_s: float = 1.5) -> None:
        if not final_video.is_file():
            report.checks.append(QACheck(
                scope="technical", name="final_video_present", passed=False,
                detail=f"{final_video} does not exist.",
                subject="final", severity="critical",
                corrective="Assemble and encode the timeline.",
            ))
            return

        report.checks.append(QACheck(
            scope="technical", name="final_video_present", passed=True,
            detail=f"{final_video.name} written "
                   f"({final_video.stat().st_size / 1e6:.2f} MB).",
            subject="final", severity="critical",
        ))

        if not self.encoder.available:
            return

        try:
            info = self.encoder.probe(final_video)
        except FFmpegError as exc:
            report.checks.append(QACheck(
                scope="technical", name="final_probe", passed=False,
                detail=f"ffprobe failed on the final video — {exc}",
                subject="final", severity="critical",
                corrective="Re-encode the final video.",
            ))
            return

        report.checks.append(QACheck(
            scope="technical", name="final_resolution",
            passed=info.width == self.width and info.height == self.height,
            detail=f"{info.width}x{info.height} (target {self.width}x{self.height}).",
            subject="final", severity="major",
            corrective="Re-encode at the target resolution.",
        ))
        report.checks.append(QACheck(
            scope="technical", name="final_fps",
            passed=abs(info.fps - self.fps) < 0.5,
            detail=f"{info.fps:.3f} fps (target {self.fps}).",
            subject="final", severity="major",
            corrective="Re-encode at the target frame rate.",
        ))

        planned_total = sum(s.duration_s for s in shots)
        duration_ok = abs(info.duration_s - planned_total) <= tolerance_s
        report.checks.append(QACheck(
            scope="technical", name="final_duration", passed=duration_ok,
            detail=(
                f"{info.duration_s:.2f}s final vs {planned_total:.2f}s planned."
            ),
            subject="final", severity="major",
            corrective="Investigate the timeline; the runtime does not match the plan.",
        ))

        # -- audio --
        if expected_audio:
            report.checks.append(QACheck(
                scope="audio", name="audio_track_present", passed=info.has_audio,
                detail=(
                    f"Audio stream: {info.audio_codec or 'none'}."
                    if info.has_audio else "No audio stream in the final video."
                ),
                subject="final", severity="major",
                corrective="Mux the audio tracks into the final video.",
            ))
            if info.has_audio:
                report.checks.append(QACheck(
                    scope="audio", name="audio_format",
                    passed=info.sample_rate >= 44100 and info.audio_channels >= 1,
                    detail=(
                        f"{info.audio_codec} {info.sample_rate} Hz, "
                        f"{info.audio_channels} channel(s)."
                    ),
                    subject="final", severity="minor",
                    corrective="Re-encode audio at 44.1 kHz or better.",
                ))
        else:
            report.checks.append(QACheck(
                scope="audio", name="audio_not_required", passed=True,
                detail=(
                    "No audio was requested for this production. "
                    + ("An audio stream is present anyway."
                       if info.has_audio else "The final video has no audio track.")
                ),
                subject="final", severity="minor",
            ))

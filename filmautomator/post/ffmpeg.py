"""FFmpeg post-production layer (spec section 13).

FFmpeg is treated as a core utility, not a helper: shot encoding, timeline
assembly, audio muxing and the technical half of final QA all live here. Only
the standard library is used — FFmpeg itself is the dependency, and adding a
Python wrapper around it would be one more thing to break.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import find_ffmpeg, find_ffprobe

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """FFmpeg exited non-zero. Carries the tail of its stderr."""

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


@dataclass
class MediaInfo:
    """What ffprobe says about a file — the technical QA evidence."""

    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    sample_rate: int = 0
    bit_rate: int = 0
    has_audio: bool = False
    has_video: bool = False
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path, "duration_s": self.duration_s,
            "width": self.width, "height": self.height, "fps": self.fps,
            "video_codec": self.video_codec, "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "sample_rate": self.sample_rate, "bit_rate": self.bit_rate,
            "has_audio": self.has_audio, "has_video": self.has_video,
        }


def _parse_fraction(value: str) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


class VideoEncoder:
    """Thin, explicit wrapper over the FFmpeg invocations this system needs."""

    def __init__(self, ffmpeg: Path | None = None, ffprobe: Path | None = None) -> None:
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self.ffprobe = ffprobe or find_ffprobe()

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.ffmpeg is not None and Path(self.ffmpeg).is_file()

    def require(self) -> Path:
        if not self.available:
            raise FFmpegError(
                "FFmpeg not found. Install it with `brew install ffmpeg`, or set "
                "FA_FFMPEG_PATH to the ffmpeg binary."
            )
        return Path(self.ffmpeg)  # type: ignore[arg-type]

    def describe(self) -> str:
        if not self.available:
            return "FFmpeg: NOT FOUND (brew install ffmpeg)"
        try:
            out = subprocess.run(
                [str(self.ffmpeg), "-version"],
                capture_output=True, text=True, timeout=30,
            )
            first = (out.stdout or "").splitlines()
            version = first[0] if first else "unknown"
        except Exception as exc:  # noqa: BLE001
            version = f"could not query ({exc})"
        probe_note = "" if self.ffprobe else "  (ffprobe missing — QA probing disabled)"
        return f"FFmpeg: {self.ffmpeg}\n  {version}{probe_note}"

    # -- execution ---------------------------------------------------------

    def _run(self, args: list[str], *, timeout: float = 3600.0) -> subprocess.CompletedProcess:
        cmd = [str(self.require()), "-hide_banner", "-nostdin", "-y", *args]
        log.debug("ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            tail = "\n".join((result.stderr or "").strip().splitlines()[-25:])
            raise FFmpegError(
                f"ffmpeg exited {result.returncode} for: {' '.join(args[:6])}...\n{tail}",
                stderr=result.stderr or "",
            )
        return result

    # -- encoding ----------------------------------------------------------

    def encode_frames(
        self,
        frames_dir: str | Path,
        output: str | Path,
        *,
        fps: int = 24,
        pattern: str = "frame_%04d.png",
        crf: int = 18,
        preset: str = "medium",
        pixel_format: str = "yuv420p",
    ) -> Path:
        """Turn a rendered PNG sequence into a shot video."""
        frames_dir = Path(frames_dir)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not frames_dir.is_dir():
            raise FFmpegError(f"frame directory does not exist: {frames_dir}")
        sequence = frames_dir / pattern
        if not list(frames_dir.glob(pattern.replace("%04d", "*"))):
            raise FFmpegError(f"no frames matching {pattern} in {frames_dir}")

        self._run([
            "-framerate", str(fps),
            "-start_number", "1",
            "-i", str(sequence),
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", pixel_format,
            "-movflags", "+faststart",
            str(output),
        ])
        return output

    def concat(self, videos: list[str | Path], output: str | Path) -> Path:
        """Join shot videos end to end into a timeline.

        Uses the concat demuxer, which is a stream copy — fast and lossless,
        and correct here because every shot is encoded with identical settings.
        """
        if not videos:
            raise FFmpegError("no videos to concatenate")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        listing = output.parent / f".{output.stem}_concat.txt"
        listing.write_text(
            "\n".join(f"file '{Path(v).resolve()}'" for v in videos),
            encoding="utf-8",
        )
        try:
            self._run([
                "-f", "concat", "-safe", "0",
                "-i", str(listing),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output),
            ])
        finally:
            listing.unlink(missing_ok=True)
        return output

    def mux_audio(
        self,
        video: str | Path,
        output: str | Path,
        *,
        audio: str | Path | None = None,
        audio_tracks: list[tuple[str | Path, float]] | None = None,
        normalize: bool = True,
    ) -> Path:
        """Attach audio to a finished picture cut (spec sections 13 and 14).

        ``audio_tracks`` places each file at a time offset in seconds, which is
        how dialogue, music and SFX get synchronised to the timeline. Tracks are
        mixed together, then optionally loudness-normalised to -16 LUFS, the
        usual target for web delivery.
        """
        video = Path(video)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        tracks: list[tuple[str | Path, float]] = list(audio_tracks or [])
        if audio is not None and not tracks:
            tracks = [(audio, 0.0)]

        if not tracks:
            # Nothing to add — remux so the caller always gets a file out.
            self._run(["-i", str(video), "-c", "copy", str(output)])
            return output

        args: list[str] = ["-i", str(video)]
        parts: list[str] = []
        labels: list[str] = []

        for index, (path, offset) in enumerate(tracks, start=1):
            args += ["-i", str(path)]
            delay_ms = max(0, int(round(offset * 1000)))
            # `all=1` applies the delay to every channel of the input.
            stage = f"adelay={delay_ms}:all=1" if delay_ms else "anull"
            parts.append(f"[{index}:a]{stage}[a{index}]")
            labels.append(f"[a{index}]")

        # amix with a single input is a valid pass-through, so this branch
        # handles both the one-track and many-track cases identically.
        parts.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:"
            "dropout_transition=0[mixed]"
        )
        final_label = "[mixed]"
        if normalize:
            parts.append("[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]")
            final_label = "[out]"

        args += [
            "-filter_complex", ";".join(parts),
            "-map", "0:v:0", "-map", final_label,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(output),
        ]
        self._run(args)
        return output

    def burn_subtitles(self, video: str | Path, subtitles: str | Path,
                       output: str | Path) -> Path:
        """Burn an SRT into the picture. Requires an ffmpeg build with libass."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # The subtitles filter needs the colon and backslash escaped in paths.
        escaped = str(Path(subtitles).resolve()).replace("\\", "\\\\").replace(":", "\\:")
        self._run(["-i", str(video), "-vf", f"subtitles='{escaped}'",
                   "-c:a", "copy", str(output)])
        return output

    def extract_frame(self, video: str | Path, timestamp_s: float,
                      output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._run(["-ss", str(timestamp_s), "-i", str(video),
                   "-frames:v", "1", str(output)])
        return output

    # -- probing (the technical half of QA) -------------------------------

    def probe(self, path: str | Path) -> MediaInfo:
        """Inspect a media file. Raises when ffprobe is unavailable."""
        if self.ffprobe is None or not Path(self.ffprobe).is_file():
            raise FFmpegError(
                "ffprobe not found; it normally ships with ffmpeg. "
                "Install with `brew install ffmpeg`."
            )
        target = Path(path)
        if not target.is_file():
            raise FFmpegError(f"file does not exist: {target}")

        result = subprocess.run(
            [str(self.ffprobe), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(target)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise FFmpegError(
                f"ffprobe failed on {target}: {result.stderr.strip()[:400]}"
            )
        payload = json.loads(result.stdout or "{}")
        return self._to_media_info(target, payload)

    @staticmethod
    def _to_media_info(path: Path, payload: dict) -> MediaInfo:
        info = MediaInfo(path=str(path))
        info.raw = payload

        fmt = payload.get("format") or {}
        info.duration_s = float(fmt.get("duration") or 0.0)
        info.bit_rate = int(fmt.get("bit_rate") or 0)

        for stream in payload.get("streams") or []:
            kind = stream.get("codec_type")
            if kind == "video" and not info.has_video:
                info.has_video = True
                info.video_codec = stream.get("codec_name", "")
                info.width = int(stream.get("width") or 0)
                info.height = int(stream.get("height") or 0)
                # Prefer the average rate; fall back to the declared base.
                info.fps = (
                    _parse_fraction(stream.get("avg_frame_rate", ""))
                    or _parse_fraction(stream.get("r_frame_rate", ""))
                )
                if not info.duration_s and stream.get("duration"):
                    info.duration_s = float(stream["duration"])
            elif kind == "audio" and not info.has_audio:
                info.has_audio = True
                info.audio_codec = stream.get("codec_name", "")
                info.audio_channels = int(stream.get("channels") or 0)
                info.sample_rate = int(stream.get("sample_rate") or 0)
        return info

    # -- environment -------------------------------------------------------

    @staticmethod
    def diagnose() -> str:
        ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe()
        lines = []
        if ffmpeg is None:
            lines.append("FFmpeg: NOT FOUND\n  brew install ffmpeg")
        else:
            lines.append(f"FFmpeg: {ffmpeg}")
        lines.append(f"FFprobe: {ffprobe}" if ffprobe else
                     "FFprobe: NOT FOUND (ships with ffmpeg)")

        # Surface whether libx264 and libass are present, since the pipeline
        # depends on the first and subtitles depend on the second.
        if ffmpeg is not None:
            try:
                out = subprocess.run(
                    [str(ffmpeg), "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=60,
                )
                encoders = out.stdout or ""
                lines.append(
                    "  libx264: "
                    + ("yes" if "libx264" in encoders else "NO — shot encoding will fail")
                )
                filters = subprocess.run(
                    [str(ffmpeg), "-hide_banner", "-filters"],
                    capture_output=True, text=True, timeout=60,
                ).stdout or ""
                lines.append(
                    "  subtitles filter: "
                    + ("yes" if re.search(r"\bsubtitles\b", filters) else "no")
                )
            except Exception as exc:  # noqa: BLE001
                lines.append(f"  could not query encoder support ({exc})")
        return "\n".join(lines)


def find_frame_sequence(directory: str | Path) -> list[Path]:
    """Every rendered frame in a directory, in frame order."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    frames = [
        p for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr"} and p.is_file()
    ]
    return sorted(frames, key=lambda p: (
        int(m.group()) if (m := re.search(r"(\d+)", p.stem)) else 0
    ))


def check_frames_contiguous(frames: list[Path]) -> list[int]:
    """Frame numbers present in a sequence, used to detect dropped frames."""
    numbers = []
    for path in frames:
        match = re.search(r"(\d+)", path.stem)
        if match:
            numbers.append(int(match.group()))
    return sorted(numbers)


def missing_frames(frames: list[Path], expected: int) -> list[int]:
    """Which of the 1..expected frames are absent (spec section 15)."""
    present = set(check_frames_contiguous(frames))
    return [n for n in range(1, expected + 1) if n not in present]


__all__ = [
    "FFmpegError",
    "MediaInfo",
    "VideoEncoder",
    "check_frames_contiguous",
    "find_frame_sequence",
    "missing_frames",
]

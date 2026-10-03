"""Frame-sequence helpers and FFmpeg wrapper behaviour that does not need
FFmpeg installed to verify.
"""

from __future__ import annotations

from pathlib import Path

from filmautomator.post.ffmpeg import (
    FFmpegError,
    VideoEncoder,
    check_frames_contiguous,
    find_frame_sequence,
    missing_frames,
)


def _touch_frames(directory: Path, numbers: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        (directory / f"frame_{n:04d}.png").write_bytes(b"x")


def test_find_frame_sequence_sorts_numerically_not_lexically(tmp_path: Path):
    _touch_frames(tmp_path, [1, 2, 10, 11, 100])
    names = [p.name for p in find_frame_sequence(tmp_path)]
    # Lexical order would put frame_1000 before frame_0002.
    assert names == ["frame_0001.png", "frame_0002.png", "frame_0010.png",
                     "frame_0011.png", "frame_0100.png"]


def test_find_frame_sequence_ignores_non_image_files(tmp_path: Path):
    _touch_frames(tmp_path, [1, 2])
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "spec.json").write_text("{}")
    assert len(find_frame_sequence(tmp_path)) == 2


def test_find_frame_sequence_on_missing_directory(tmp_path: Path):
    assert find_frame_sequence(tmp_path / "nope") == []


def test_missing_frames_detects_a_gap(tmp_path: Path):
    _touch_frames(tmp_path, [1, 2, 3, 5, 6])
    frames = find_frame_sequence(tmp_path)
    assert check_frames_contiguous(frames) == [1, 2, 3, 5, 6]
    assert missing_frames(frames, 6) == [4]


def test_missing_frames_when_nothing_rendered(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert missing_frames(find_frame_sequence(empty), 3) == [1, 2, 3]


def test_complete_sequence_has_no_gaps(tmp_path: Path):
    _touch_frames(tmp_path, list(range(1, 49)))
    assert missing_frames(find_frame_sequence(tmp_path), 48) == []


# -- encoder behaviour without FFmpeg -------------------------------------


def test_encoder_reports_unavailable_rather_than_crashing():
    encoder = VideoEncoder(ffmpeg=Path("/nonexistent/ffmpeg"),
                           ffprobe=Path("/nonexistent/ffprobe"))
    assert encoder.available is False
    assert "NOT FOUND" in encoder.describe()


def test_encode_frames_raises_a_clear_error_when_ffmpeg_missing(tmp_path: Path):
    encoder = VideoEncoder(ffmpeg=Path("/nonexistent/ffmpeg"))
    _touch_frames(tmp_path, [1, 2])
    try:
        encoder.encode_frames(tmp_path, tmp_path / "out.mp4")
    except FFmpegError as exc:
        assert "FFmpeg not found" in str(exc)
        return
    raise AssertionError("expected FFmpegError")


def test_encode_frames_rejects_a_directory_with_no_frames(tmp_path: Path):
    encoder = VideoEncoder(ffmpeg=Path("/bin/echo"))  # exists, so require() passes
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        encoder.encode_frames(empty, tmp_path / "out.mp4")
    except FFmpegError as exc:
        assert "no frames matching" in str(exc)
        return
    raise AssertionError("expected FFmpegError for an empty frame directory")


def test_probe_raises_when_ffprobe_missing(tmp_path: Path):
    target = tmp_path / "x.mp4"
    target.write_bytes(b"not really a video")
    encoder = VideoEncoder(ffmpeg=Path("/bin/echo"), ffprobe=Path("/nonexistent/ffprobe"))
    try:
        encoder.probe(target)
    except FFmpegError as exc:
        assert "ffprobe not found" in str(exc)
        return
    raise AssertionError("expected FFmpegError")


def test_diagnose_reports_missing_ffmpeg(monkeypatch):
    monkeypatch.setenv("FA_FFMPEG_PATH", "/nonexistent/ffmpeg")
    monkeypatch.setenv("FA_FFPROBE_PATH", "/nonexistent/ffprobe")
    text = VideoEncoder.diagnose()
    assert "NOT FOUND" in text


# -- muxing argument construction -----------------------------------------


def test_mux_with_no_tracks_still_produces_a_file(tmp_path: Path, monkeypatch):
    """A silent production must still deliver a final video."""
    calls: list[list[str]] = []

    encoder = VideoEncoder(ffmpeg=Path("/bin/echo"))
    monkeypatch.setattr(encoder, "_run", lambda args, **kw: calls.append(args))

    video = tmp_path / "timeline.mp4"
    video.write_bytes(b"x")
    encoder.mux_audio(video, tmp_path / "final.mp4", audio_tracks=[])

    assert len(calls) == 1
    assert "-c" in calls[0] and "copy" in calls[0]


def test_mux_with_tracks_builds_a_mix_filter(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    encoder = VideoEncoder(ffmpeg=Path("/bin/echo"))
    monkeypatch.setattr(encoder, "_run", lambda args, **kw: calls.append(args))

    video = tmp_path / "timeline.mp4"
    video.write_bytes(b"x")
    dialogue = tmp_path / "line.wav"
    dialogue.write_bytes(b"x")
    music = tmp_path / "score.wav"
    music.write_bytes(b"x")

    encoder.mux_audio(
        video, tmp_path / "final.mp4",
        audio_tracks=[(dialogue, 1.5), (music, 0.0)],
        normalize=False,
    )

    args = calls[0]
    filter_complex = args[args.index("-filter_complex") + 1]
    # The 1.5s line must actually be delayed by 1500 ms.
    assert "adelay=1500" in filter_complex
    assert "amix=inputs=2" in filter_complex
    assert filter_complex.count("-i") == 0


def test_mux_normalisation_adds_loudnorm(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []
    encoder = VideoEncoder(ffmpeg=Path("/bin/echo"))
    monkeypatch.setattr(encoder, "_run", lambda args, **kw: calls.append(args))

    video = tmp_path / "t.mp4"
    video.write_bytes(b"x")
    track = tmp_path / "a.wav"
    track.write_bytes(b"x")

    encoder.mux_audio(video, tmp_path / "f.mp4", audio_tracks=[(track, 0.0)],
                      normalize=True)
    filter_complex = calls[0][calls[0].index("-filter_complex") + 1]
    assert "loudnorm=I=-16" in filter_complex

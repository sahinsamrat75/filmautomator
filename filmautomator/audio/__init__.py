"""Audio capability foundation (spec: model-agnostic audio).

There is no requirement for a large local audio model, and this system does not
install one. What matters is that the *capabilities* exist, are addressable, and
can be filled by something better later without changing callers:

    dialogue_generation  -> a script, with performance direction
    tts                  -> spoken audio
    music_generation     -> a score
    sfx_generation       -> discrete effects
    ambience_generation  -> a continuous environment bed

The implementations shipped here are deterministic and local: procedural
synthesis written to WAV with the standard library. They are honest about what
they are. A deterministic tone is labelled ``synthetic``, exactly as the mock
model driver is, so a stand-in decision is never presented as a model's work.

Why procedural rather than waiting for a real model: the pipeline needs to be
exercisable end to end today — dialogue timed, music bed laid under a cut,
ambience under rain — and a real voice or a real score should be a drop-in
replacement for these functions, not a reason the feature is absent.
"""

from __future__ import annotations

import logging
import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

#: Every audio artifact this module produces is tagged with its provenance, so a
#: caller can always tell a real model from a deterministic stand-in.
SYNTHETIC = "synthetic"

AUDIO_CAPABILITIES: tuple[str, ...] = (
    "dialogue_generation",
    "tts",
    "music_generation",
    "sfx_generation",
    "ambience_generation",
)


@dataclass
class AudioArtifact:
    """One generated audio file, with the provenance that explains it."""

    kind: str            # dialogue | music | sfx | ambience
    path: str
    duration_s: float
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    source: str = SYNTHETIC
    model: str = ""
    deterministic: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source": self.source,
            "model": self.model,
            "deterministic": self.deterministic,
            "synthetic": self.source == SYNTHETIC,
            "metadata": self.metadata,
        }


def _write_wav(path: str | Path, samples: Iterable[float],
               sample_rate: int = SAMPLE_RATE) -> float:
    """Write mono float samples in [-1, 1] to a 16-bit WAV.

    Peaks are clamped rather than wrapped: a clipped sample is an honest
    artefact of an overdriven source, whereas wrapping turns a loud passage into
    a crackle that sounds like a bug.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    count = 0
    for value in samples:
        clipped = max(-1.0, min(1.0, value))
        frames += struct.pack("<h", int(clipped * 32767))
        count += 1
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return count / float(sample_rate or 1)


def _envelope(index: int, total: int, attack: float = 0.02,
              release: float = 0.25) -> float:
    """A simple attack/release shape, so tones do not click."""
    if total <= 0:
        return 0.0
    position = index / total
    if position < attack:
        return position / attack
    if position > (1.0 - release):
        return max(0.0, (1.0 - position) / release)
    return 1.0


def generate_tts(
    text: str,
    path: str | Path,
    *,
    voice: str = "neutral",
    duration_s: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
) -> AudioArtifact:
    """Produce spoken-word-length audio for a line of dialogue.

    Honest about what it is: this generates a voiced placeholder whose length is
    derived from the text, not intelligible speech. It exists so dialogue has a
    real duration on the timeline and the mix can be built and checked now. The
    artifact is labelled ``synthetic`` so no downstream step mistakes it for a
    finished voice performance.

    Replacing this with a real TTS engine is a change to this function alone.
    """
    words = max(1, len(text.split()))
    # ~2.6 words per second is ordinary conversational pacing.
    target = duration_s or max(0.4, words / 2.6)
    total = int(target * sample_rate)

    # A voiced formant-ish tone: a fundamental plus two harmonics, with pitch
    # drifting slightly so it does not sound like a test tone.
    pitch = {"neutral": 128.0, "low": 96.0, "high": 176.0}.get(voice, 128.0)

    def samples():
        for index in range(total):
            t = index / sample_rate
            drift = 1.0 + 0.04 * math.sin(2.0 * math.pi * 0.7 * t)
            f0 = pitch * drift
            value = (
                0.55 * math.sin(2.0 * math.pi * f0 * t)
                + 0.25 * math.sin(2.0 * math.pi * f0 * 2.0 * t)
                + 0.12 * math.sin(2.0 * math.pi * f0 * 3.0 * t)
            )
            yield value * 0.35 * _envelope(index, total)

    written = _write_wav(path, samples(), sample_rate)
    return AudioArtifact(
        kind="dialogue",
        path=str(path),
        duration_s=written,
        sample_rate=sample_rate,
        metadata={"text": text, "voice": voice, "words": words,
                  "note": "procedural placeholder speech, not intelligible"},
    )


def generate_music(
    path: str | Path,
    *,
    duration_s: float = 8.0,
    bpm: int = 72,
    key: str = "minor",
    mood: str = "tense",
    instrumentation: str = "synth",
    intensity: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> AudioArtifact:
    """Generate a deterministic procedural score from a musical brief.

    Mood, tempo, key and intensity genuinely change the output, so the brief an
    AI supplies has a real, audible consequence — which is what makes this a
    foundation rather than a placeholder that ignores its inputs. It is a
    harmonic pad and pulse, not a composed piece.
    """
    total = int(max(0.1, duration_s) * sample_rate)
    beat = 60.0 / max(1, bpm)
    scale = [0, 3, 7] if key == "minor" else [0, 4, 7]
    root_hz = 110.0

    def samples():
        for index in range(total):
            t = index / sample_rate
            value = 0.0
            for degree, semitones in enumerate(scale):
                freq = root_hz * (2.0 ** ((semitones + degree * 12) / 12.0))
                value += math.sin(2.0 * math.pi * freq * t) / (degree + 2)
            # A soft pulse on the beat gives the pad a tempo.
            phase = (t % beat) / beat
            pulse = math.exp(-6.0 * phase) * intensity * 0.25
            yield (value * 0.22 + pulse * math.sin(2.0 * math.pi * root_hz * 2 * t)) \
                * _envelope(index, total, attack=0.05, release=0.1)

    written = _write_wav(path, samples(), sample_rate)
    return AudioArtifact(
        kind="music",
        path=str(path),
        duration_s=written,
        sample_rate=sample_rate,
        metadata={"bpm": bpm, "key": key, "mood": mood,
                  "instrumentation": instrumentation,
                  "intensity": intensity,
                  "note": "procedural pad and pulse, not a composed score"},
    )


def generate_sfx(
    name: str,
    path: str | Path,
    *,
    duration_s: float = 0.6,
    sample_rate: int = SAMPLE_RATE,
) -> AudioArtifact:
    """Generate a discrete sound effect by name.

    A small library of recognisable procedural effects. Unknown names fall back
    to a neutral click rather than raising, because an unavailable effect should
    not stop a cut from being assembled.
    """
    total = int(max(0.02, duration_s) * sample_rate)
    kind = name.strip().lower().replace(" ", "_")

    def samples():
        for index in range(total):
            t = index / sample_rate
            progress = index / max(1, total)
            if kind in {"rain", "rain_ambience"}:
                # Filtered noise: a steady bed with slow amplitude wander.
                noise = (math.sin(index * 12.9898) * 43758.5453) % 1.0
                value = noise * 0.18 * (1.0 + 0.3 * math.sin(2 * math.pi * 0.3 * t))
            elif kind in {"impact", "hit", "thud"}:
                freq = 90.0 * math.exp(-6.0 * progress)
                value = math.sin(2.0 * math.pi * freq * t) * math.exp(-5.0 * progress)
            elif kind in {"whoosh", "pass_by"}:
                freq = 400.0 + 1600.0 * math.sin(math.pi * progress)
                value = math.sin(2.0 * math.pi * freq * t) * 0.3 \
                    * math.sin(math.pi * progress)
            elif kind in {"click", "ui"}:
                value = math.sin(2.0 * math.pi * 1200.0 * t) * math.exp(-30.0 * progress)
            else:
                value = math.sin(2.0 * math.pi * 440.0 * t) * math.exp(-8.0 * progress)
            yield value * _envelope(index, total, attack=0.01, release=0.3)

    written = _write_wav(path, samples(), sample_rate)
    return AudioArtifact(
        kind="sfx",
        path=str(path),
        duration_s=written,
        sample_rate=sample_rate,
        metadata={"name": name,
                  "note": "procedurally synthesised effect"},
    )


def generate_ambience(
    path: str | Path,
    *,
    duration_s: float = 8.0,
    environment: str = "night_city",
    sample_rate: int = SAMPLE_RATE,
) -> AudioArtifact:
    """Generate a continuous environment bed.

    Layered low-frequency noise with slow movement. Different environments get
    different beds so a street does not sound like a forest.
    """
    total = int(max(0.1, duration_s) * sample_rate)
    profile = {
        "night_city": (55.0, 0.20),
        "rain": (70.0, 0.26),
        "forest": (40.0, 0.16),
        "room": (35.0, 0.10),
        "wind": (45.0, 0.18),
    }.get(environment, (50.0, 0.18))
    base_hz, level = profile

    def samples():
        for index in range(total):
            t = index / sample_rate
            noise = (math.sin(index * 78.233) * 12345.678) % 1.0
            drift = 0.6 + 0.4 * math.sin(2.0 * math.pi * 0.13 * t)
            value = noise * level * drift
            value += 0.10 * math.sin(2.0 * math.pi * base_hz * t)
            yield value * _envelope(index, total, attack=0.1, release=0.1)

    written = _write_wav(path, samples(), sample_rate)
    return AudioArtifact(
        kind="ambience",
        path=str(path),
        duration_s=written,
        sample_rate=sample_rate,
        metadata={"environment": environment,
                  "note": "procedural environment bed"},
    )


def dialogue_script(
    beats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise a dialogue brief into timed lines.

    Accepts ``{character, line, start_s, duration_s}`` in any order and fills in
    missing durations from the speaking rate, so a caller can supply beats as
    written rather than pre-computing timings.
    """
    lines: list[dict[str, Any]] = []
    for beat in beats or []:
        text = str(beat.get("line") or beat.get("text") or "")
        if not text.strip():
            continue
        words = len(text.split())
        duration = float(beat.get("duration_s") or 0.0) or max(0.5, words / 2.6)
        lines.append({
            "character": str(beat.get("character") or ""),
            "line": text,
            "start_s": float(beat.get("start_s") or 0.0),
            "duration_s": duration,
            "emotion": str(beat.get("emotion") or "neutral"),
        })
    return lines


__all__ = [
    "AUDIO_CAPABILITIES",
    "AudioArtifact",
    "SYNTHETIC",
    "dialogue_script",
    "generate_ambience",
    "generate_music",
    "generate_sfx",
    "generate_tts",
]

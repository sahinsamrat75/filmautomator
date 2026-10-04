"""Structured production specs.

Spec section 8: agents communicate through structured data, not arbitrary
prose. These dataclasses are the contract — the Director emits a
:class:`ShotSpec`, the Blender Agent consumes it, the Vision Agent judges the
render against it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Cinematography vocabulary
# --------------------------------------------------------------------------

SHOT_SIZES = (
    "extreme_wide", "wide", "full", "medium_wide", "medium",
    "medium_close", "close_up", "extreme_close_up",
)

CAMERA_ANGLES = (
    "eye_level", "low", "high", "birds_eye", "worms_eye", "dutch", "over_shoulder",
)

CAMERA_MOVEMENTS = (
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down",
    "dolly_in", "dolly_out", "truck_left", "truck_right", "crane_up", "handheld",
)

LIGHTING_MOODS = (
    "natural", "golden_hour", "blue_hour", "overcast", "harsh_noon",
    "moonlight", "candlelit", "neon", "firelight", "high_key", "low_key", "silhouette",
)


@dataclass
class CameraSpec:
    """Where the camera is and what it is doing."""

    shot_size: str = "medium"
    angle: str = "eye_level"
    movement: str = "static"
    lens_mm: float = 50.0
    #: World-space position of the camera.
    location: tuple[float, float, float] = (0.0, -6.0, 1.6)
    #: World-space point the camera aims at.
    look_at: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Shallow depth of field when set.
    dof_distance: float | None = None
    fstop: float = 2.8
    #: Whether the corresponding field was explicitly requested by the
    #: director, rather than left to the framing solver. Presence — not
    #: value — decides this: a shot that asks for camera position
    #: [0, -6, 1.6] (which happens to equal the default) must still get
    #: exactly that position, never a shot-size solve that moves it 40 m.
    location_explicit: bool = False
    look_at_explicit: bool = False
    lens_explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LightingSpec:
    """The lighting design for a shot."""

    mood: str = "natural"
    #: Overall exposure intent, in stops relative to a neutral render.
    exposure_compensation: float = 0.0
    key_energy: float = 1200.0
    key_color: tuple[float, float, float] = (1.0, 0.96, 0.9)
    #: Key light position relative to the subject, world space.
    key_location: tuple[float, float, float] = (3.0, -4.0, 5.0)
    fill_energy: float = 300.0
    fill_color: tuple[float, float, float] = (0.75, 0.82, 1.0)
    fill_location: tuple[float, float, float] = (-4.0, -3.0, 3.0)
    rim_energy: float = 800.0
    rim_color: tuple[float, float, float] = (0.85, 0.9, 1.0)
    rim_location: tuple[float, float, float] = (-2.0, 5.0, 4.0)
    #: Ambient world colour and strength.
    world_color: tuple[float, float, float] = (0.05, 0.06, 0.09)
    world_strength: float = 1.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubjectSpec:
    """Something that must appear in the shot, and how it should look."""

    name: str
    kind: str = "character"  # character | prop | vehicle | creature | effect
    #: Canonical asset to instance, if the Asset Registry has one.
    asset_id: str = ""
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Approximate height in metres — the Vision Agent uses this to judge
    #: whether the subject reads at the intended size in frame.
    height_m: float = 1.8
    appearance: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioRequirement:
    """What the shot needs on the soundtrack (spec section 14)."""

    dialogue: list[dict[str, Any]] = field(default_factory=list)
    ambience: str = ""
    sfx: list[str] = field(default_factory=list)
    music_cue: str = ""

    def is_empty(self) -> bool:
        return not (self.dialogue or self.ambience or self.sfx or self.music_cue)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShotSpec:
    """One shot: the atomic unit of production (spec section 12)."""

    shot_id: str
    scene_id: str = ""
    ordinal: int = 0
    duration_s: float = 4.0
    description: str = ""
    camera: CameraSpec = field(default_factory=CameraSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    subjects: list[SubjectSpec] = field(default_factory=list)
    environment: str = ""
    props: list[str] = field(default_factory=list)
    vfx: list[str] = field(default_factory=list)
    audio: AudioRequirement = field(default_factory=AudioRequirement)
    #: Style constraints the Continuity Agent enforces across shots.
    style: str = ""
    continuity: list[str] = field(default_factory=list)
    #: What "good" means for this shot, used verbatim by the Vision Agent.
    quality_criteria: list[str] = field(default_factory=list)
    #: The owner-facing creative intent this shot serves.
    dramatic_purpose: str = ""

    @property
    def frame_count(self) -> int:
        return 0  # resolved at render time once fps is known

    def frame_count_at(self, fps: int) -> int:
        return max(1, int(round(self.duration_s * fps)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "ordinal": self.ordinal,
            "duration_s": self.duration_s,
            "description": self.description,
            "camera": self.camera.to_dict(),
            "lighting": self.lighting.to_dict(),
            "subjects": [s.to_dict() for s in self.subjects],
            "environment": self.environment,
            "props": list(self.props),
            "vfx": list(self.vfx),
            "audio": self.audio.to_dict(),
            "style": self.style,
            "continuity": list(self.continuity),
            "quality_criteria": list(self.quality_criteria),
            "dramatic_purpose": self.dramatic_purpose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShotSpec":
        camera_data = data.get("camera") or {}
        camera = CameraSpec(**{k: _coerce_tuple(v) for k, v in
                               camera_data.items()
                               if k in CameraSpec.__dataclass_fields__})
        # Explicitness is carried in the flags when the spec has been through
        # a store/round-trip. For a freshly assembled spec (from a tool call)
        # the *presence* of the key decides: a director who passed
        # "location": [...] asked for that position even when it equals the
        # dataclass default.
        if "location_explicit" in camera_data:
            camera.location_explicit = bool(camera_data["location_explicit"])
        else:
            camera.location_explicit = camera_data.get("location") is not None
        if "look_at_explicit" in camera_data:
            camera.look_at_explicit = bool(camera_data["look_at_explicit"])
        else:
            camera.look_at_explicit = camera_data.get("look_at") is not None
        if "lens_explicit" in camera_data:
            camera.lens_explicit = bool(camera_data["lens_explicit"])
        else:
            camera.lens_explicit = camera_data.get("lens_mm") is not None
        return cls(
            shot_id=data["shot_id"],
            scene_id=data.get("scene_id", ""),
            ordinal=int(data.get("ordinal", 0)),
            duration_s=float(data.get("duration_s", 4.0)),
            description=data.get("description", ""),
            camera=camera,
            lighting=LightingSpec(**{k: _coerce_tuple(v) for k, v in
                                     (data.get("lighting") or {}).items()
                                     if k in LightingSpec.__dataclass_fields__}),
            subjects=[SubjectSpec(**{k: _coerce_tuple(v) for k, v in s.items()
                                     if k in SubjectSpec.__dataclass_fields__})
                      for s in data.get("subjects") or []],
            environment=data.get("environment", ""),
            props=list(data.get("props") or []),
            vfx=list(data.get("vfx") or []),
            audio=AudioRequirement(**{k: v for k, v in
                                      (data.get("audio") or {}).items()
                                      if k in AudioRequirement.__dataclass_fields__}),
            style=data.get("style", ""),
            continuity=list(data.get("continuity") or []),
            quality_criteria=list(data.get("quality_criteria") or []),
            dramatic_purpose=data.get("dramatic_purpose", ""),
        )


def _coerce_tuple(value: Any) -> Any:
    """JSON has no tuples; convert lists back for vector-ish fields."""
    if isinstance(value, list) and value and all(
        isinstance(v, (int, float)) for v in value
    ):
        return tuple(value)
    return value


@dataclass
class SceneSpec:
    """A scene groups shots that share a location and time."""

    scene_id: str
    ordinal: int = 0
    title: str = ""
    synopsis: str = ""
    location: str = ""
    time_of_day: str = ""
    shots: list[ShotSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "ordinal": self.ordinal,
            "title": self.title,
            "synopsis": self.synopsis,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "shots": [s.to_dict() for s in self.shots],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSpec":
        return cls(
            scene_id=data["scene_id"],
            ordinal=int(data.get("ordinal", 0)),
            title=data.get("title", ""),
            synopsis=data.get("synopsis", ""),
            location=data.get("location", ""),
            time_of_day=data.get("time_of_day", ""),
            shots=[ShotSpec.from_dict(s) for s in data.get("shots") or []],
        )


@dataclass
class QualityFinding:
    """One problem the Vision Agent found, actionable by the Blender Agent."""

    category: str        # framing | lighting | composition | continuity | technical
    severity: str        # minor | major | critical
    description: str
    #: Concrete correction the Blender Agent can apply, in its own vocabulary.
    suggested_fix: str = ""
    #: Optional parameter delta, e.g. {"camera.lens_mm": 85, "lighting.key_energy": 2000}
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityFinding":
        return cls(
            category=data.get("category", "technical"),
            severity=data.get("severity", "minor"),
            description=data.get("description", ""),
            suggested_fix=data.get("suggested_fix", ""),
            parameters=data.get("parameters") or {},
        )


@dataclass
class QualityReport:
    """The Vision Agent's judgement of a rendered frame."""

    passed: bool
    score: float = 0.0
    findings: list[QualityFinding] = field(default_factory=list)
    summary: str = ""
    #: True when produced by image analysis rather than a vision model, so the
    #: owner knows the judgement was mechanical.
    heuristic_only: bool = False

    @property
    def blocking_findings(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity in {"major", "critical"}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "heuristic_only": self.heuristic_only,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityReport":
        return cls(
            passed=bool(data.get("passed")),
            score=float(data.get("score", 0.0)),
            findings=[QualityFinding.from_dict(f) for f in data.get("findings") or []],
            summary=data.get("summary", ""),
            heuristic_only=bool(data.get("heuristic_only")),
        )

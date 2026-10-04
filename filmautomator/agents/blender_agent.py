"""Blender Agent — turns a :class:`ShotSpec` into a real Blender scene.

This agent owns the Blender session. No other agent drives it.

Canonical assets are real Blender geometry: environments compile from their
records into meshes, lights and materials, and characters compile into
segmented humanoids with armatures, named parts and canonical materials. A
shot instantiates the canonical records its spec references, so continuity is
structural rather than a naming convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..blender.assets import (
    build_character,
    build_environment,
    pose_character,
)
from ..blender.session import BlenderSession
from ..config import RenderConfig
from ..core.events import EventKind
from ..core.spec import LightingSpec, QualityFinding, ShotSpec, SubjectSpec
from ..core.task import Task
from ..core.workspace import Workspace
from .base import Agent, AgentContext

#: How much of the frame height the subject should fill, per shot size.
#: Used to solve camera distance from subject height.
_FILL_RATIO = {
    "extreme_wide": 0.06,
    "wide": 0.16,
    "full": 0.55,
    "medium_wide": 0.66,
    "medium": 0.80,
    "medium_close": 1.10,
    "close_up": 1.90,
    "extreme_close_up": 3.50,
}

#: Lenses conventionally paired with each shot size.
_DEFAULT_LENS = {
    "extreme_wide": 24.0,
    "wide": 28.0,
    "full": 35.0,
    "medium_wide": 40.0,
    "medium": 50.0,
    "medium_close": 65.0,
    "close_up": 85.0,
    "extreme_close_up": 100.0,
}

#: Vertical offset of the camera from the look-at point, per angle.
_ANGLE_HEIGHT = {
    "eye_level": 0.0,
    "low": -0.7,
    "high": 1.5,
    "birds_eye": 4.0,
    "worms_eye": -1.4,
    "dutch": 0.1,
    "over_shoulder": 0.25,
}

#: Camera roll in radians, per angle.
_ANGLE_ROLL = {"dutch": math.radians(15.0)}

#: Where the camera aims, as a fraction of subject height. A wide shot should
#: centre the whole body; a close-up should centre the face. Aiming at one
#: fixed height (head height) crops the legs out of every medium shot.
_LOOK_AT_HEIGHT = {
    "extreme_wide": 0.50,
    "wide": 0.50,
    "full": 0.52,
    "medium_wide": 0.55,
    "medium": 0.58,
    "medium_close": 0.72,
    "close_up": 0.88,
    "extreme_close_up": 0.92,
}


@dataclass
class ShotBuildResult:
    """What the build produced for a version."""

    shot_id: str
    version: int
    blend_path: str
    object_count: int
    camera: str
    notes: list[str]
    #: Every object the build created, so verification can check presence.
    objects: list[str] = field(default_factory=list)
    #: Per-part roles, e.g. {"environment": [...], "characters": {...}}.
    manifest: dict[str, Any] = field(default_factory=dict)
    #: The camera state read back from Blender after the build.
    camera_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewRender:
    """A preview frame plus the background plate that goes with it."""

    image_path: Path
    background_path: Path | None
    engine: str = ""
    plate_note: str = ""


def subject_object_names(spec: ShotSpec,
                         characters: dict[str, dict[str, Any]] | None = None
                         ) -> list[str]:
    """The scene objects that represent this shot's subjects.

    Resolved from the Blender manifest when available: canonical characters
    build a dozen named parts (``CHR_<name>_*``), so the subject is the union
    of its parts plus its root. Falls back to the legacy two-object names for
    specs built before canonical characters existed. The root and armature are
    included so hiding the subject for a background plate removes every
    render-visible part — a hidden empty parent does not hide its children.
    """
    names: list[str] = []
    for subject in spec.subjects:
        built: dict[str, Any] | None = None
        if characters:
            built = characters.get(subject.name) or {}
        parts = [str(p) for p in (built or {}).get("parts") or []]
        if parts:
            names.extend(parts)
            names.append(str(built.get("root") or f"CHR_{subject.name}_root"))
            armature = built.get("armature")
            if armature:
                names.append(str(armature))
        else:
            names.extend([
                f"{subject.name}_body", f"{subject.name}_head",
                f"CHR_{subject.name}_root",
            ])
    return names


class BlenderAgent(Agent):
    name = "blender_agent"
    role = "Constructs and revises Blender scenes from structured shot specs."

    def __init__(self, context: AgentContext, session: BlenderSession,
                 render: RenderConfig) -> None:
        super().__init__(context)
        self.session = session
        self.render = render
        #: Per-build evidence, consumed by build_shot and verify_build.
        self._last_environment: dict[str, Any] = {}
        self._last_characters: dict[str, Any] = {}
        self._camera_verification: dict[str, Any] = {}

    # -- construction ------------------------------------------------------

    def build_shot(self, task: Task, spec: ShotSpec, version: int,
                   *, fps: int | None = None) -> ShotBuildResult:
        """Build the whole scene for one shot version from scratch.

        Always starts from an empty scene so a rebuild is reproducible from the
        spec alone rather than depending on whatever was left in the file.
        """
        ws = self.ctx.workspace
        fps = fps or self.render.fps
        ws.prepare_shot_version(spec.shot_id, version)
        notes: list[str] = []

        self.announce(f"building {spec.shot_id} v{version:03d}: {spec.description or 'shot'}")
        self.session.reset_scene(empty=True)

        self._setup_scene(fps, spec)
        self._last_environment = {}
        self._last_characters = {}
        self._camera_verification = {}
        self._build_environment(spec, notes)
        self._build_subjects(spec, notes)
        self._build_lighting(spec)
        self._build_camera(spec)

        blend_path = ws.shot_blend(spec.shot_id, version)
        self.session.save_blend(blend_path)
        ws.write_json(
            Path("shots") / spec.shot_id / f"v{version:03d}" / "spec.json",
            spec.to_dict(),
        )

        state = self.session.scene_state()
        scopes = []
        if self._last_environment.get("scope"):
            scopes.append(self._last_environment["scope"])
        for built in self._last_characters.values():
            if built.get("scope"):
                scopes.append(built["scope"])
        manifest = self.session.object_manifest(scopes)
        verification = self.verify_build(spec, state, manifest)
        if not verification["ok"]:
            self.log.warning(
                "build verification failed for %s: %s",
                spec.shot_id, verification["problems"],
            )
            notes.append(
                "build verification: " + "; ".join(verification["problems"][:4])
            )
        return ShotBuildResult(
            shot_id=spec.shot_id,
            version=version,
            blend_path=str(blend_path),
            object_count=state["object_count"],
            camera=state.get("active_camera") or "",
            notes=notes,
            objects=[o["name"] for o in state.get("objects", [])],
            manifest={
                "environment": self._last_environment,
                "characters": self._last_characters,
                "camera": self._camera_verification,
                "verification": verification,
            },
            camera_state=self._camera_verification,
        )

    def verify_build(self, spec: ShotSpec, state: dict[str, Any],
                     manifest: dict[str, Any]) -> dict[str, Any]:
        """Verify the built scene against the spec, from Blender's own state.

        Blender is authoritative for scene content: the database can claim
        anything, but only bpy.data proves the objects exist, are visible, and
        carry materials. Every problem is a concrete missing piece, never a
        vague verdict.
        """
        problems: list[str] = []
        objects = {o["name"]: o for o in state.get("objects", [])}
        groups = manifest.get("groups", {})

        # -- camera must exist and be active --------------------------------
        active = state.get("active_camera")
        if not active:
            problems.append("no active camera in the built scene")
        elif active not in objects:
            problems.append(f"active camera {active!r} has no scene object")

        # -- environment parts must exist and be render-visible -------------
        if spec.environment:
            env_scope = self._last_environment.get("scope", "")
            env_objects = self._group_members(groups, env_scope)
            if not env_objects:
                problems.append(
                    f"environment {spec.environment!r}: no Blender objects built"
                )
            for obj in env_objects:
                if obj.get("hide_render"):
                    problems.append(
                        f"environment object {obj['name']!r} is hidden"
                    )
                if obj.get("type") == "MESH" and not obj.get("vertex_count"):
                    problems.append(
                        f"environment object {obj['name']!r} has no mesh"
                    )
                if obj.get("type") == "MESH" and not any(obj.get("materials") or []):
                    problems.append(
                        f"environment object {obj['name']!r} has no material"
                    )

        # -- character parts must exist and be render-visible ---------------
        for subject in spec.subjects:
            built = self._last_characters.get(subject.name) or {}
            parts = self._group_members(groups, built.get("scope", ""))
            if not parts:
                problems.append(
                    f"subject {subject.name!r}: no Blender objects built"
                )
                continue
            meshes = [p for p in parts if p.get("type") == "MESH"]
            if len(meshes) < 5:
                problems.append(
                    f"subject {subject.name!r}: only {len(meshes)} mesh parts, "
                    "expected a segmented character"
                )
            for obj in parts:
                if obj.get("hide_render"):
                    problems.append(
                        f"character object {obj['name']!r} is render-hidden"
                    )
            if not any(p.get("type") == "ARMATURE" for p in parts):
                problems.append(
                    f"subject {subject.name!r}: no armature was built"
                )

        # -- materials for canonical colours --------------------------------
        material_names = {m.get("name") for m in state.get("materials", [])}
        for subject in spec.subjects:
            built = self._last_characters.get(subject.name) or {}
            scope = built.get("scope", "")
            expected = f"{scope}_MAT_coat" if scope else ""
            if expected and expected not in material_names:
                problems.append(
                    f"subject {subject.name!r}: canonical coat material "
                    f"{expected!r} missing"
                )

        return {"ok": not problems, "problems": problems,
                "object_count": len(objects),
                "groups": sorted(groups)}

    @staticmethod
    def _group_members(groups: dict[str, Any], scope: str) -> list[dict[str, Any]]:
        """Every manifest entry belonging to ``scope``."""
        if not scope:
            return []
        if scope in groups:
            return list(groups[scope])
        found: list[dict[str, Any]] = []
        for group_name, members in groups.items():
            if group_name.startswith(scope):
                found.extend(members)
        return found

    @staticmethod
    def _safe_id(value: str) -> str:
        return "".join(c if (c.isalnum() or c == "_") else "_" for c in value)

    # -- scene pieces ------------------------------------------------------

    def _setup_scene(self, fps: int, spec: ShotSpec) -> None:
        frames = spec.frame_count_at(fps)
        self.session.set_scene_timing(frame_start=1, frame_end=frames, fps=fps)
        self.session.set_render_settings(
            engine=self.render.preview_engine,
            resolution_x=self.render.preview_width,
            resolution_y=self.render.preview_height,
            resolution_percentage=100,
            fps=fps,
            film_transparent=False,
            world_color=spec.lighting.world_color,
            world_strength=spec.lighting.world_strength,
            # AgX is Blender 4.x's default filmic response. The control server
            # reports back what it actually applied if this build disagrees,
            # rather than failing the whole call.
            view_transform="AgX",
        )

    def _build_environment(self, spec: ShotSpec, notes: list[str]) -> None:
        """Compile the canonical environment record into real Blender objects.

        When the shot references a defined environment (``spec.environment``),
        the stored record — ground, backdrop, fountain, lantern, archway,
        layout, materials, lighting baseline — drives the build. Otherwise a
        neutral stage is built so framing and exposure can still be judged.
        """
        record = self._environment_record(spec.environment)
        if record is None:
            scope = "ENV_NeutralStage"
            self.session.create_primitive(
                "plane", name=f"{scope}_ground", location=(0.0, 0.0, 0.0),
                size=200.0,
            )
            self.session.create_material(
                "MAT_Ground", base_color=(0.16, 0.15, 0.14), roughness=0.92,
                metallic=0.0,
            )
            self.session.assign_material("MAT_Ground", [f"{scope}_ground"])
            self.session.create_primitive(
                "plane", name=f"{scope}_backdrop",
                location=(0.0, 14.0, 10.0),
                rotation=(math.radians(90.0), 0.0, 0.0),
                size=60.0,
            )
            self.session.create_material(
                "MAT_Backdrop", base_color=(0.10, 0.11, 0.14), roughness=1.0,
            )
            self.session.assign_material("MAT_Backdrop", [f"{scope}_backdrop"])
            self._last_environment = {
                "environment_id": "", "scope": scope,
                "objects": [f"{scope}_ground", f"{scope}_backdrop"],
                "materials": ["MAT_Ground", "MAT_Backdrop"], "lights": [],
                "notes": ["neutral proxy stage"],
            }
            if spec.environment:
                notes.append(
                    f"no canonical record for environment {spec.environment!r}; "
                    "built a neutral stage instead"
                )
            else:
                notes.append("no environment specified; used a neutral proxy stage")
            return

        built = build_environment(
            self.session, record["environment_id"], record["spec"],
        )
        notes.append(
            f"environment {record['environment_id']}: "
            f"{len(built.objects)} Blender objects "
            f"({', '.join(built.objects[:6])}"
            f"{', …' if len(built.objects) > 6 else ''})"
        )
        notes.extend(built.notes)
        self._last_environment = built.to_dict()

    def _environment_record(self, environment_id: str) -> dict[str, Any] | None:
        """Load the canonical environment record, if one was defined."""
        if not environment_id:
            return None
        db = self.ctx.db
        if db is None:
            return None
        try:
            assets = db.find_assets(self.ctx.project_id, "environment")
        except Exception:  # noqa: BLE001 - a lost registry is not a lost scene
            return None
        for asset in assets:
            metadata = asset.get("metadata") or {}
            if metadata.get("environment_id") == environment_id:
                trimmed = {k: v for k, v in metadata.items()
                           if k != "environment_id"}
                return {"environment_id": environment_id, "spec": trimmed}
            if asset.get("name") == environment_id:
                return {"environment_id": environment_id, "spec": dict(metadata)}
        return None

    def _build_subjects(self, spec: ShotSpec, notes: list[str]) -> None:
        if not spec.subjects:
            notes.append("shot has no subjects; only the stage was built")
            return

        self._last_characters = {}
        for index, subject in enumerate(spec.subjects):
            built = self._build_subject_canonical(subject, index, notes)
            self._last_characters[subject.name] = built.to_dict()

    def _character_record(self, name: str) -> dict[str, Any] | None:
        """Load the canonical character record, if one was defined."""
        db = self.ctx.db
        if db is None:
            return None
        try:
            return db.get_character(self.ctx.project_id, name)
        except Exception:  # noqa: BLE001
            return None

    def _build_subject_canonical(
        self, subject: SubjectSpec, index: int, notes: list[str],
    ):
        """Build a subject from its canonical character record.

        The canonical definition — proportions, clothing, colors, materials —
        drives the geometry. The subject's shot-level location, scale and action
        place and pose it. A subject with no canonical record falls back to a
        visible proxy block so an undefined name is still present rather
        than absent.
        """
        record = self._character_record(subject.name)
        canonical = dict(record.get("canonical") or {}) if record else {}
        # The shot's height wins when the record is silent; the record wins
        # when the shot restates nothing. Either way the number is explicit.
        if subject.height_m and not canonical.get("height_m"):
            canonical["height_m"] = float(subject.height_m)
        colors = dict(canonical.get("colors") or {})
        if subject.appearance and "appearance_note" not in colors:
            colors["appearance_note"] = subject.appearance
        canonical["colors"] = colors
        if subject.action and "action_note" not in canonical:
            canonical = dict(canonical)
            canonical["action_note"] = subject.action

        character_id = ((record or {}).get("character_id", "")
                        or subject.asset_id or "")
        built = build_character(
            self.session, character_id, subject.name, canonical,
            location=tuple(subject.location),
        )

        # Shot-level scale on the root handle; the parts inherit through
        # parenting rather than being rebuilt per shot.
        sx, sy, sz = subject.scale
        if (sx, sy, sz) != (1.0, 1.0, 1.0):
            self.session.set_transform(built.root, scale=(sx, sy, sz))

        notes.append(
            f"subject {subject.name!r}: canonical character with "
            f"{len(built.parts)} parts and {len(built.bones)} bones "
            f"(record={'defined' if record else 'proxy fallback'})"
        )

        # The shot's action selects a deterministic pose.
        pose = self._pose_for_action(subject.action)
        try:
            pose_character(self.session, built.armature, pose, frame=None)
            notes.append(f"{subject.name!r} posed: {pose}")
        except Exception as exc:  # noqa: BLE001 - a pose must not fail a build
            notes.append(f"{subject.name!r} pose {pose!r} skipped: {exc}")
        return built

    @staticmethod
    def _pose_for_action(action: str) -> str:
        """Map a shot action onto the deterministic pose vocabulary."""
        text = (action or "").lower()
        if any(word in text for word in ("walk", "stride", "approach", "cross")):
            return "walk"
        if any(word in text for word in ("enter", "arrive", "step in")):
            return "enter"
        if any(word in text for word in ("reach", "touch", "take", "grasp",
                                         "lift")):
            return "reach"
        if any(word in text for word in ("lean", "rest", "slump")):
            return "lean"
        if any(word in text for word in ("react", "startle", "turn", "look",
                                         "surprise", "fear", "gasp")):
            return "react"
        return "idle"

    def _build_lighting(self, spec: ShotSpec) -> None:
        light = spec.lighting
        target = self._lighting_target(spec)

        self.session.create_light(
            name="LGT_Key", light_type="AREA", energy=light.key_energy,
            color=light.key_color, location=light.key_location,
            look_at=target, size=3.0,
        )
        self.session.create_light(
            name="LGT_Fill", light_type="AREA", energy=light.fill_energy,
            color=light.fill_color, location=light.fill_location,
            look_at=target, size=5.0,
        )
        self.session.create_light(
            name="LGT_Rim", light_type="AREA", energy=light.rim_energy,
            color=light.rim_color, location=light.rim_location,
            look_at=target, size=2.0,
        )

    @staticmethod
    def _lighting_target(spec: ShotSpec) -> tuple[float, float, float]:
        if spec.subjects:
            primary = spec.subjects[0]
            return (
                primary.location[0],
                primary.location[1],
                primary.location[2] + primary.height_m * 0.55,
            )
        return (0.0, 0.0, 1.0)

    def _build_camera(self, spec: ShotSpec) -> None:
        location, look_at, lens = self._solve_camera(spec)
        created = self.session.create_camera(
            name="CAM_Main",
            lens_mm=lens,
            location=location,
            look_at=look_at,
            dof_distance=spec.camera.dof_distance,
            fstop=spec.camera.fstop,
            make_active=True,
        )
        # Read the camera back and compare against the *applied* placement, not
        # the raw spec values. When the spec left the camera to be solved, the
        # solved position is the intent; comparing against the untouched
        # dataclass default would report a drift that never happened. What this
        # catches is Blender silently keeping its own default instead of using
        # what was asked for.
        self._camera_verification = self.verify_camera(
            spec, created, intended_location=location, intended_lens=lens,
            tolerance_m=0.05, tolerance_lens=0.5,
        )
        if not self._camera_verification["ok"]:
            self.log.warning(
                "camera verification failed for %s: %s",
                spec.shot_id, self._camera_verification["problems"],
            )
        # Movement becomes real keyframes on the camera, including an explicit
        # static hold so STATIC is a deliberate state rather than an accident.
        # Start from the *applied* location — the solved position when the spec
        # left the camera to be framed, the explicit position when it asked for
        # one — never the dataclass default, which would yank the camera away
        # from the solved framing.
        keyframes = self._keyframe_camera(
            spec, created.get("name", "CAM_Main"), start=location,
        )
        if keyframes:
            self._camera_verification["keyframes"] = keyframes

    def verify_camera(self, spec: ShotSpec, created: dict[str, Any],
                      intended_location: tuple[float, float, float] | None = None,
                      intended_lens: float | None = None,
                      tolerance_m: float = 0.05,
                      tolerance_lens: float = 0.5) -> dict[str, Any]:
        """Compare the intended camera against what Blender actually reports.

        Returns ``ok`` plus the per-field deltas. Anything outside tolerance is
        a problem, not a rounding wobble — silently substituting a default
        camera is a build failure, and it is reported as one.
        """
        want_location = tuple(
            float(v) for v in (intended_location or spec.camera.location)
        )
        want_lens = float(
            intended_lens if intended_lens is not None
            else (spec.camera.lens_mm or 0.0)
        )
        got_location = tuple(created.get("location") or (0.0, 0.0, 0.0))
        got_lens = float(created.get("lens_mm") or 0.0)
        got_rotation = tuple(created.get("rotation_euler") or (0.0, 0.0, 0.0))

        problems: list[str] = []
        location_delta = math.dist(want_location, got_location)
        if location_delta > tolerance_m:
            problems.append(
                f"location drifted {location_delta:.3f} m "
                f"(want {want_location}, got {got_location})"
            )
        if want_lens and abs(want_lens - got_lens) > tolerance_lens:
            problems.append(
                f"lens is {got_lens:.1f} mm, spec asked {want_lens:.1f} mm"
            )
        return {
            "ok": not problems,
            "problems": problems,
            "requested": {"location": want_location,
                          "look_at": tuple(float(v) for v in spec.camera.look_at),
                          "lens_mm": want_lens, "movement": spec.camera.movement},
            "actual": {"location": got_location, "rotation_euler": got_rotation,
                       "lens_mm": got_lens},
            "location_delta_m": round(location_delta, 4),
        }

    def _keyframe_camera(self, spec: ShotSpec, camera_name: str,
                         start: tuple[float, float, float] | None = None
                         ) -> list[dict]:
        """Keyframe camera movement; an explicit hold when STATIC.

        Dolly/pan/tilt/crane/handheld become start/end keyframes on location so
        the final render actually moves. STATIC keys a single hold frame so the
        camera is provably still rather than accidentally unkeyed.

        ``start`` is the camera position already applied to Blender (solved or
        explicit). It is never the spec's raw default, which would displace the
        camera from its solved framing.
        """
        movement = (spec.camera.movement or "static").lower()
        frames = max(1, spec.frame_count_at(self.render.fps))
        keyframes: list[dict] = []

        def key(frame: int, location: tuple) -> None:
            self.session.set_transform(camera_name, location=list(location))
            self.session.call("set_keyframe", name=camera_name,
                              location=list(location), frame=frame)
            keyframes.append({"frame": frame, "location": list(location)})

        start_location = tuple(
            float(v) for v in (start if start is not None else spec.camera.location)
        )
        end = self._movement_end(spec, start_location)
        if movement == "static" or end is None:
            key(1, start_location)
            return keyframes
        key(1, start_location)
        key(frames, end)
        return keyframes

    @staticmethod
    def _movement_end(spec: ShotSpec,
                      start: tuple[float, float, float]) -> tuple | None:
        """Where the camera should be on the last frame of a move."""
        movement = (spec.camera.movement or "static").lower()
        look_at = tuple(float(v) for v in spec.camera.look_at)
        sx, sy, sz = start
        if movement == "dolly_in":
            return (sx, sy * 0.75, sz)
        if movement == "dolly_out":
            return (sx, sy * 1.25, sz)
        if movement == "truck_left":
            return (sx - 1.5, sy, sz)
        if movement == "truck_right":
            return (sx + 1.5, sy, sz)
        if movement == "pan_left":
            return start
        if movement == "pan_right":
            return start
        if movement == "tilt_up":
            return (sx, sy, sz + 0.8)
        if movement == "tilt_down":
            return (sx, sy, sz - 0.8)
        if movement == "crane_up":
            return (sx, sy * 1.1, sz + 1.5)
        if movement == "handheld":
            return (sx + 0.12, sy - 0.1, sz + 0.08)
        if movement == "static":
            return None
        return None

    def _solve_camera(
        self, spec: ShotSpec
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """Derive camera placement from shot size, angle and the subject.

        The spec's explicit location and lens always win. Only when the spec
        carries defaults (location untouched, lens unset) is the framing
        equation solved from shot size and angle. An AI that asked for a camera
        at (0, -7, 2) with a 35mm lens gets exactly that — never a silently
        substituted solve.
        """
        camera = spec.camera
        subject = spec.subjects[0] if spec.subjects else None

        # An explicitly placed camera is a directorial decision, not a default
        # to be solved away. The dataclass default is (0, -6, 1.6); anything
        # else was asked for on purpose.
        explicit_location = tuple(camera.location) != (0.0, -6.0, 1.6)
        explicit_look = tuple(camera.look_at) != (0.0, 0.0, 1.0)
        explicit_lens = bool(camera.lens_mm)

        if subject is not None:
            look_height = _LOOK_AT_HEIGHT.get(camera.shot_size, 0.58)
            solved_look = (
                subject.location[0],
                subject.location[1],
                subject.location[2] + subject.height_m * look_height,
            )
            height = max(0.2, subject.height_m)
        else:
            solved_look = (0.0, 0.0, 1.0)
            height = 1.8
        look_at = tuple(camera.look_at) if explicit_look else solved_look

        if explicit_location:
            location = tuple(float(v) for v in camera.location)
        else:
            fill = _FILL_RATIO.get(camera.shot_size, 0.80)
            framing_height = height / fill
            distance = (framing_height * float(camera.lens_mm or 50.0))
            aspect = self.render.preview_width / max(1, self.render.preview_height)
            sensor_height = 36.0 / aspect if aspect >= 1.0 else 36.0
            distance = max(0.35, distance / max(1e-6, sensor_height))
            dz = _ANGLE_HEIGHT.get(camera.angle, 0.0) * (height / 1.8)
            dx = -0.55 * height if camera.angle == "over_shoulder" else 0.0
            if camera.angle == "birds_eye":
                dy = -distance * 0.35
            else:
                dy = -distance
            location = (look_at[0] + dx, look_at[1] + dy, look_at[2] + dz)

        lens = float(camera.lens_mm or _DEFAULT_LENS.get(camera.shot_size, 50.0))
        return location, look_at, lens

    # -- rendering ---------------------------------------------------------

    def render_preview(self, task: Task, spec: ShotSpec, version: int,
                       *, frame: int | None = None) -> PreviewRender:
        """Two cheap frames: the shot, and the same shot with subjects hidden.

        The second is what makes the Vision Agent's coverage measurement
        trustworthy — see ``op_analyze_image``. It costs one extra preview
        render, which at preview resolution is a fraction of a second.
        """
        ws = self.ctx.workspace
        path = ws.shot_preview(spec.shot_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.emit(EventKind.PREVIEW_STARTED,
                  f"Rendering preview for {spec.shot_id} (v{version:03d})",
                  task=task, shot_id=spec.shot_id, scene_id=spec.scene_id)

        settings = self.session.set_render_settings(
            engine=self.render.preview_engine,
            resolution_x=self.render.preview_width,
            resolution_y=self.render.preview_height,
            samples=self.render.preview_samples,
            cycles_device=self.render.cycles_device,
        )
        result = self.session.render_still(path, frame=frame)
        self.log.info("preview rendered: %s (%d bytes)", path, result["bytes"])
        self.ctx.db.record_render(
            self.ctx.project_id, shot_id=spec.shot_id, version_no=version,
            kind="preview", path=str(path), engine=result.get("engine", ""),
            metadata=result,
        )

        plate_path: Path | None = None
        note = str(settings.get("engine_note") or "")
        if note:
            self.log.warning("preview render settings: %s", note)

        names = subject_object_names(spec, self._last_characters)
        if names:
            plate_path = ws.shot_preview_plate(spec.shot_id, version)
            try:
                plate = self.session.call(
                    "render_still", filepath=str(plate_path),
                    frame=frame, hide_names=names,
                )
                if plate.get("hidden"):
                    self.log.info(
                        "background plate rendered: %s (hid %s)",
                        plate_path, ", ".join(plate["hidden"]),
                    )
                else:
                    # Nothing was actually hidden, so the "plate" is identical
                    # to the subject render and differencing it would report
                    # zero coverage. Better to have no plate than a misleading
                    # one.
                    self.log.warning(
                        "background plate requested but no subject objects were "
                        "found to hide; falling back to chroma measurement"
                    )
                    plate_path = None
            except Exception as exc:  # noqa: BLE001 - plate is an optimisation
                self.log.warning("background plate render failed (%s); "
                                 "falling back to chroma measurement", exc)
                plate_path = None

        # The dashboard and the Vision Agent must look at the same pixels, so
        # the preview is registered and mirrored the moment it exists. The
        # artifact points at the *mirrored* copy under previews/ — the durable
        # visual record the dashboard serves — not the in-shot working copy,
        # which the storage governor releases with its frames once the shot is
        # promoted.
        mirrored = ws.mirror_preview(spec.shot_id, version)
        artifact_path = str(mirrored or path)
        artifact_id = self.ctx.db.register_artifact(
            self.ctx.project_id, "preview", artifact_path,
            f"{spec.shot_id} v{version:03d} preview",
            shot_id=spec.shot_id, scene_id=spec.scene_id,
            metadata={"version": version, "working_copy": str(path), **result},
        )
        self.emit(EventKind.PREVIEW_READY,
                  f"Preview ready for {spec.shot_id} (v{version:03d})",
                  task=task, shot_id=spec.shot_id, scene_id=spec.scene_id,
                  payload={"path": artifact_path, "artifact_id": artifact_id,
                           "version": version})

        return PreviewRender(
            image_path=path,
            background_path=plate_path,
            engine=str(result.get("engine", "")),
            plate_note=note,
        )

    def render_final_sequence(self, task: Task, spec: ShotSpec, version: int,
                              *, fps: int | None = None) -> Path:
        """Full-quality frame sequence — the input to the FFmpeg encode."""
        ws = self.ctx.workspace
        fps = fps or self.render.fps
        out_dir = ws.shot_render_dir(spec.shot_id, version)

        frames = spec.frame_count_at(fps)
        self.emit(EventKind.RENDER_STARTED,
                  f"Final render {spec.shot_id} v{version:03d} — {frames} frames",
                  task=task, shot_id=spec.shot_id, scene_id=spec.scene_id,
                  payload={"frames": frames, "engine": self.render.final_engine,
                           "width": self.render.final_width,
                           "height": self.render.final_height})
        self.announce(
            f"final render {spec.shot_id} v{version:03d} — {frames} frames, "
            f"{self.render.final_width}x{self.render.final_height}, "
            f"{self.render.final_engine}"
        )

        self.session.set_scene_timing(frame_start=1, frame_end=frames, fps=fps)
        self.session.set_render_settings(
            engine=self.render.final_engine,
            resolution_x=self.render.final_width,
            resolution_y=self.render.final_height,
            resolution_percentage=100,
            fps=fps,
            samples=self.render.final_samples,
            cycles_device=self.render.cycles_device,
            denoise=True,
            image_format="PNG",
        )
        result = self.session.render_animation(out_dir, frame_start=1, frame_end=frames)
        self.ctx.db.record_render(
            self.ctx.project_id, shot_id=spec.shot_id, version_no=version,
            kind="final_frames", path=str(out_dir),
            engine=self.render.final_engine, metadata=result,
        )
        artifact_id = self.ctx.db.register_artifact(
            self.ctx.project_id, "render", str(out_dir),
            f"{spec.shot_id} v{version:03d} frames",
            shot_id=spec.shot_id, scene_id=spec.scene_id,
            metadata={"version": version, **result},
        )
        self.emit(EventKind.RENDER_COMPLETED,
                  f"Rendered {result.get('frame_count', 0)} frames for {spec.shot_id}",
                  task=task, shot_id=spec.shot_id, scene_id=spec.scene_id,
                  payload={"directory": str(out_dir), "artifact_id": artifact_id,
                           "frame_count": result.get("frame_count", 0)})
        return out_dir

    # -- revision ----------------------------------------------------------

    def apply_corrections(self, spec: ShotSpec,
                          findings: list[QualityFinding]) -> tuple[ShotSpec, list[str]]:
        """Fold Vision Agent findings back into the spec.

        Each finding may carry explicit ``parameters`` (dotted paths into the
        spec). Those are applied directly and deterministically — no model call
        needed. Findings without parameters fall back to category-level
        heuristics. Every change is recorded so the revision is auditable.
        """
        import copy

        revised = copy.deepcopy(spec)
        applied: list[str] = []

        for finding in findings:
            for path, value in (finding.parameters or {}).items():
                if self._set_path(revised, path, value):
                    applied.append(f"{path} = {value!r}  ({finding.description})")
                    continue
                applied.append(f"SKIPPED unknown parameter {path!r}")

        if not any(f.parameters for f in findings):
            applied.extend(self._heuristic_corrections(revised, findings))

        return revised, applied

    @staticmethod
    def _set_path(spec: ShotSpec, dotted: str, value: Any) -> bool:
        """Set ``camera.lens_mm`` style paths on a ShotSpec."""
        parts = dotted.split(".")
        if not parts:
            return False
        target: Any = spec
        for part in parts[:-1]:
            if part.startswith("subjects[") and part.endswith("]"):
                index = int(part[len("subjects["):-1])
                if index >= len(spec.subjects):
                    return False
                target = spec.subjects[index]
            elif hasattr(target, part):
                target = getattr(target, part)
            else:
                return False
        final = parts[-1]
        if not hasattr(target, final):
            return False
        try:
            setattr(target, final, value)
        except (AttributeError, TypeError):
            return False
        return True

    @staticmethod
    def _heuristic_corrections(spec: ShotSpec, findings: list[QualityFinding]) -> list[str]:
        """Category-level fixes when the model gave no explicit parameters."""
        applied: list[str] = []
        for finding in findings:
            category = finding.category

            if category == "lighting":
                if "too dark" in finding.description.lower() or "underlit" in finding.description.lower():
                    spec.lighting.key_energy *= 1.6
                    spec.lighting.fill_energy *= 1.5
                    applied.append(
                        f"raised key to {spec.lighting.key_energy:.0f} and fill to "
                        f"{spec.lighting.fill_energy:.0f} (underlit)"
                    )
                elif "too bright" in finding.description.lower() or "blown" in finding.description.lower():
                    spec.lighting.key_energy *= 0.65
                    spec.lighting.fill_energy *= 0.7
                    applied.append(f"lowered key to {spec.lighting.key_energy:.0f} (overexposed)")

            elif category == "framing":
                order = ["extreme_wide", "wide", "full", "medium_wide", "medium",
                         "medium_close", "close_up", "extreme_close_up"]
                if "too small" in finding.description.lower() or "too far" in finding.description.lower():
                    idx = order.index(spec.camera.shot_size) if spec.camera.shot_size in order else 4
                    spec.camera.shot_size = order[min(len(order) - 1, idx + 1)]
                    applied.append(f"tightened framing to {spec.camera.shot_size}")
                elif "too large" in finding.description.lower() or "too close" in finding.description.lower():
                    idx = order.index(spec.camera.shot_size) if spec.camera.shot_size in order else 4
                    spec.camera.shot_size = order[max(0, idx - 1)]
                    applied.append(f"loosened framing to {spec.camera.shot_size}")

            elif category == "composition":
                applied.append(
                    "composition flagged but no parameter delta supplied; "
                    "recorded for the Director to re-plan"
                )

        return applied

"""Blender Agent — turns a :class:`ShotSpec` into a real Blender scene.

This agent owns the Blender session. No other agent drives it.

Honest limitation, stated plainly: this milestone has no character or
environment *assets* yet, so subjects are built as proxy geometry (a
blocked-in figure at the right height and position). That is enough for the
Vision Agent to judge composition, scale, framing and lighting — which is what
the observe/correct loop needs to function. Importing real canonical character
assets is the next milestone and the Asset Registry is already shaped for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class PreviewRender:
    """A preview frame plus the background plate that goes with it."""

    image_path: Path
    background_path: Path | None
    engine: str = ""
    plate_note: str = ""


def subject_object_names(spec: ShotSpec) -> list[str]:
    """The scene objects that represent this shot's subjects.

    Names are deterministic — ``_build_subject_proxy`` derives them from the
    subject name — so the list can be reconstructed from the spec without
    threading state through the build.
    """
    names: list[str] = []
    for subject in spec.subjects:
        names.extend([f"{subject.name}_body", f"{subject.name}_head"])
    return names


class BlenderAgent(Agent):
    name = "blender_agent"
    role = "Constructs and revises Blender scenes from structured shot specs."

    def __init__(self, context: AgentContext, session: BlenderSession,
                 render: RenderConfig) -> None:
        super().__init__(context)
        self.session = session
        self.render = render

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
        return ShotBuildResult(
            shot_id=spec.shot_id,
            version=version,
            blend_path=str(blend_path),
            object_count=state["object_count"],
            camera=state.get("active_camera") or "",
            notes=notes,
        )

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
        """Ground plane plus a backdrop.

        Stands in for a real environment asset. It gives the Vision Agent a
        horizon and something for the key light to fall on, without which
        framing and exposure are impossible to judge.
        """
        self.session.create_primitive(
            "plane", name="ENV_Ground", location=(0.0, 0.0, 0.0), size=200.0,
        )
        self.session.create_material(
            "MAT_Ground", base_color=(0.16, 0.15, 0.14), roughness=0.92, metallic=0.0,
        )
        self.session.assign_material("MAT_Ground", ["ENV_Ground"])

        # A backdrop wall keeps the frame from opening onto infinite void.
        self.session.create_primitive(
            "plane", name="ENV_Backdrop",
            location=(0.0, 14.0, 10.0),
            rotation=(math.radians(90.0), 0.0, 0.0),
            size=60.0,
        )
        self.session.create_material(
            "MAT_Backdrop", base_color=(0.10, 0.11, 0.14), roughness=1.0,
        )
        self.session.assign_material("MAT_Backdrop", ["ENV_Backdrop"])

        if spec.environment:
            notes.append(
                f"environment described as {spec.environment!r}; built as a proxy "
                "stage (ground + backdrop) pending real environment assets"
            )
        else:
            notes.append("no environment specified; used a neutral proxy stage")

    def _build_subjects(self, spec: ShotSpec, notes: list[str]) -> None:
        if not spec.subjects:
            notes.append("shot has no subjects; only the stage was built")
            return

        for index, subject in enumerate(spec.subjects):
            self._build_subject_proxy(subject, index)
            if not subject.asset_id:
                notes.append(
                    f"subject {subject.name!r} has no canonical asset yet; "
                    "represented by proxy geometry"
                )

    def _build_subject_proxy(self, subject: SubjectSpec, index: int) -> None:
        """A blocked-in humanoid figure: body, head, and a facing marker.

        Deliberately crude. Its job is to occupy the right volume of frame at
        the right height so framing, scale and lighting can be evaluated.
        """
        height = max(0.2, float(subject.height_m))
        x, y, z = subject.location
        sx, sy, sz = subject.scale

        body_height = height * 0.78
        head_radius = height * 0.11
        # Blocky shoulders so the silhouette reads as a figure, not a post.
        body_width = height * 0.26 * sx

        self.session.create_primitive(
            "cube", name=f"{subject.name}_body",
            location=(x, y, z + body_height / 2.0),
            scale=(body_width / 2.0, height * 0.10 * sy, body_height / 2.0),
        )
        self.session.create_primitive(
            "sphere", name=f"{subject.name}_head",
            location=(x, y, z + body_height + head_radius * 0.9),
            radius=head_radius * sx,
        )

        entry_material = f"MAT_{subject.name}"
        # A distinct hue per subject makes them tellable apart in a preview,
        # which matters when the Vision Agent reports "subject A occludes B".
        palette = [
            (0.55, 0.32, 0.24), (0.24, 0.36, 0.55), (0.30, 0.48, 0.32),
            (0.52, 0.44, 0.22), (0.44, 0.26, 0.46),
        ]
        color = palette[index % len(palette)]
        self.session.create_material(
            entry_material, base_color=color, roughness=0.72, metallic=0.0,
        )
        self.session.assign_material(
            entry_material, [f"{subject.name}_body", f"{subject.name}_head"]
        )

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
        self.session.create_camera(
            name="CAM_Main",
            lens_mm=lens,
            location=location,
            look_at=look_at,
            dof_distance=spec.camera.dof_distance,
            fstop=spec.camera.fstop,
            make_active=True,
        )

    def _solve_camera(
        self, spec: ShotSpec
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
        """Derive camera placement from shot size, angle and the subject.

        Solves the thin-lens framing equation so the subject occupies the
        intended fraction of frame height:  distance = framing_height * f / sensor_height.
        """
        camera = spec.camera
        subject = spec.subjects[0] if spec.subjects else None

        if subject is not None:
            look_height = _LOOK_AT_HEIGHT.get(camera.shot_size, 0.58)
            look_at = (
                subject.location[0],
                subject.location[1],
                subject.location[2] + subject.height_m * look_height,
            )
            height = max(0.2, subject.height_m)
        else:
            look_at = (0.0, 0.0, 1.0)
            height = 1.8

        fill = _FILL_RATIO.get(camera.shot_size, 0.80)
        framing_height = height / fill
        lens = float(camera.lens_mm or _DEFAULT_LENS.get(camera.shot_size, 50.0))

        aspect = self.render.preview_width / max(1, self.render.preview_height)
        # Blender's AUTO sensor fit maps sensor_width onto the larger axis.
        sensor_height = 36.0 / aspect if aspect >= 1.0 else 36.0
        distance = (framing_height * lens) / max(1e-6, sensor_height)
        distance = max(0.35, distance)

        dz = _ANGLE_HEIGHT.get(camera.angle, 0.0) * (height / 1.8)
        # Slight lateral offset for over-the-shoulder so the near figure sits
        # at frame edge rather than dead centre.
        dx = -0.55 * height if camera.angle == "over_shoulder" else 0.0
        if camera.angle == "birds_eye":
            dy = -distance * 0.35
        else:
            dy = -distance

        location = (
            look_at[0] + dx,
            look_at[1] + dy,
            look_at[2] + dz,
        )
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

        names = subject_object_names(spec)
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
        # the preview is registered and mirrored the moment it exists.
        artifact_id = self.ctx.db.register_artifact(
            self.ctx.project_id, "preview", str(path),
            f"{spec.shot_id} v{version:03d} preview",
            shot_id=spec.shot_id, scene_id=spec.scene_id,
            metadata={"version": version, **result},
        )
        ws.mirror_preview(spec.shot_id, version)
        self.emit(EventKind.PREVIEW_READY,
                  f"Preview ready for {spec.shot_id} (v{version:03d})",
                  task=task, shot_id=spec.shot_id, scene_id=spec.scene_id,
                  payload={"path": str(path), "artifact_id": artifact_id,
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

"""Director — the AI CEO of spec section 3.

Two jobs:

* **Plan.** Turn an owner objective into structured shots.
* **Decide.** Run the build → preview → observe → evaluate → correct loop until
  a shot passes its own quality criteria, or the iteration budget runs out.

The Director is the only agent that decides a shot is finished, and the only
one that escalates to the owner (spec section 17).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core.spec import (
    AudioRequirement,
    CameraSpec,
    LightingSpec,
    QualityReport,
    ShotSpec,
    SubjectSpec,
)
from ..core.task import QAStatus, Task, TaskStatus
from ..gateway import Capability
from ..post.ffmpeg import VideoEncoder
from .base import Agent, AgentContext
from .blender_agent import BlenderAgent
from .vision_agent import VisionAgent

log = logging.getLogger(__name__)

#: Lighting presets, so a mood word from the plan becomes concrete light.
#: Values are tuned for the proxy stage; real environments will want their own.
LIGHTING_PRESETS: dict[str, dict[str, Any]] = {
    "natural": {
        "key_energy": 1400, "key_color": (1.0, 0.97, 0.93),
        "fill_energy": 500, "fill_color": (0.80, 0.86, 1.0),
        "rim_energy": 700, "world_color": (0.09, 0.10, 0.13), "world_strength": 1.0,
    },
    "golden_hour": {
        "key_energy": 2600, "key_color": (1.0, 0.72, 0.42),
        "fill_energy": 420, "fill_color": (0.62, 0.70, 0.95),
        "rim_energy": 1400, "rim_color": (1.0, 0.80, 0.55),
        "world_color": (0.14, 0.10, 0.09), "world_strength": 1.0,
    },
    "blue_hour": {
        "key_energy": 700, "key_color": (0.62, 0.72, 1.0),
        "fill_energy": 320, "fill_color": (0.45, 0.55, 0.95),
        "rim_energy": 600, "rim_color": (0.70, 0.80, 1.0),
        "world_color": (0.06, 0.09, 0.16), "world_strength": 1.2,
    },
    "overcast": {
        "key_energy": 1800, "key_color": (0.92, 0.94, 1.0),
        "fill_energy": 1100, "fill_color": (0.90, 0.93, 1.0),
        "rim_energy": 400, "world_color": (0.16, 0.17, 0.19), "world_strength": 1.4,
    },
    "harsh_noon": {
        "key_energy": 4200, "key_color": (1.0, 0.98, 0.92),
        "fill_energy": 350, "fill_color": (0.70, 0.80, 1.0),
        "rim_energy": 500, "world_color": (0.12, 0.14, 0.18), "world_strength": 1.1,
    },
    "moonlight": {
        "key_energy": 520, "key_color": (0.58, 0.70, 1.0),
        "fill_energy": 150, "fill_color": (0.35, 0.48, 0.90),
        "rim_energy": 900, "rim_color": (0.75, 0.85, 1.0),
        "world_color": (0.03, 0.05, 0.11), "world_strength": 1.0,
    },
    "candlelit": {
        "key_energy": 320, "key_color": (1.0, 0.62, 0.28),
        "fill_energy": 70, "fill_color": (0.85, 0.55, 0.30),
        "rim_energy": 220, "rim_color": (1.0, 0.70, 0.35),
        "world_color": (0.02, 0.015, 0.01), "world_strength": 0.6,
    },
    "neon": {
        "key_energy": 1500, "key_color": (1.0, 0.25, 0.62),
        "fill_energy": 900, "fill_color": (0.20, 0.85, 1.0),
        "rim_energy": 1600, "rim_color": (0.35, 0.95, 1.0),
        "world_color": (0.05, 0.02, 0.09), "world_strength": 1.0,
    },
    "firelight": {
        "key_energy": 900, "key_color": (1.0, 0.52, 0.18),
        "fill_energy": 160, "fill_color": (0.90, 0.45, 0.20),
        "rim_energy": 700, "rim_color": (1.0, 0.62, 0.25),
        "world_color": (0.03, 0.02, 0.015), "world_strength": 0.7,
    },
    "high_key": {
        "key_energy": 3000, "key_color": (1.0, 0.99, 0.98),
        "fill_energy": 2200, "fill_color": (0.98, 0.99, 1.0),
        "rim_energy": 900, "world_color": (0.30, 0.31, 0.33), "world_strength": 1.6,
    },
    "low_key": {
        "key_energy": 900, "key_color": (1.0, 0.95, 0.88),
        "fill_energy": 60, "fill_color": (0.50, 0.60, 0.85),
        "rim_energy": 1300, "rim_color": (0.85, 0.92, 1.0),
        "world_color": (0.015, 0.018, 0.026), "world_strength": 0.5,
    },
    "silhouette": {
        "key_energy": 5200, "key_color": (1.0, 0.85, 0.65),
        "fill_energy": 8, "fill_color": (0.30, 0.40, 0.70),
        "rim_energy": 3000, "rim_color": (1.0, 0.90, 0.75),
        "world_color": (0.35, 0.22, 0.12), "world_strength": 2.2,
    },
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "logline": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What happens on screen in this shot.",
                    },
                    "duration_s": {"type": "number", "minimum": 0.5, "maximum": 60.0},
                    "shot_size": {
                        "type": "string",
                        "enum": ["extreme_wide", "wide", "full", "medium_wide",
                                 "medium", "medium_close", "close_up",
                                 "extreme_close_up"],
                    },
                    "angle": {
                        "type": "string",
                        "enum": ["eye_level", "low", "high", "birds_eye",
                                 "worms_eye", "dutch", "over_shoulder"],
                    },
                    "movement": {
                        "type": "string",
                        "enum": ["static", "pan_left", "pan_right", "tilt_up",
                                 "tilt_down", "dolly_in", "dolly_out", "truck_left",
                                 "truck_right", "crane_up", "handheld"],
                    },
                    "lens_mm": {"type": "number", "minimum": 14.0, "maximum": 200.0},
                    "lighting_mood": {
                        "type": "string",
                        "enum": sorted(LIGHTING_PRESETS),
                    },
                    "environment": {"type": "string"},
                    "dramatic_purpose": {"type": "string"},
                    "subjects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "height_m": {"type": "number", "minimum": 0.2,
                                             "maximum": 4.0},
                                "location": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "[x, y, z] in metres",
                                },
                                "action": {"type": "string"},
                                "appearance": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                    "quality_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Concrete, checkable statements about what a good "
                            "frame for this shot looks like."
                        ),
                    },
                },
                "required": ["description", "duration_s", "shot_size"],
            },
        },
    },
    "required": ["shots"],
}

DIRECTOR_SYSTEM = (
    "You are the director of a small animation studio. You plan films as "
    "precisely specified shots that a 3D artist can build without asking "
    "follow-up questions. You are concrete about camera, lens and lighting, "
    "and you never pad a plan with filler shots. You respond only with JSON."
)


@dataclass
class ShotProduction:
    """The outcome of producing one shot."""

    shot_id: str
    approved: bool
    version: int
    rounds: int
    blend_path: str = ""
    preview_path: str = ""
    frames_dir: str = ""
    video_path: str = ""
    final_report: QualityReport | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    title: str
    logline: str
    shots: list[ShotSpec]
    synthetic: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "logline": self.logline,
            "synthetic": self.synthetic,
            "notes": self.notes,
            "shots": [s.to_dict() for s in self.shots],
        }


class Director(Agent):
    name = "director"
    role = "Plans the production and decides when a shot is good enough."

    def __init__(self, context: AgentContext, blender: BlenderAgent,
                 vision: VisionAgent, encoder: VideoEncoder,
                 *, max_rounds: int = 4, fps: int = 24,
                 final_width: int = 1280, final_height: int = 720) -> None:
        super().__init__(context)
        self.blender = blender
        self.vision = vision
        self.encoder = encoder
        self.max_rounds = max(1, max_rounds)
        self.fps = fps
        self.final_width = final_width
        self.final_height = final_height

    # -- planning ----------------------------------------------------------

    def plan_shots(self, objective: str, target_duration_s: float,
                   *, style: str = "", extra_context: str = "") -> PlanResult:
        """Break an owner objective into shots (spec sections 9 and 12)."""
        self.announce(f"planning: {objective}")
        prompt = "\n".join(filter(None, [
            "Plan a short cinematic sequence.",
            "",
            f"Objective: {objective}",
            f"Target total duration: {target_duration_s:.1f} seconds",
            f"Visual style: {style}" if style else "",
            extra_context,
            "",
            "Requirements:",
            f"  - The shot durations must sum to approximately {target_duration_s:.1f}s.",
            "  - Use the fewest shots that serve the objective; do not pad.",
            "  - For each shot give concrete camera, lens and lighting choices.",
            "  - For each shot give 2-4 quality_criteria that a reviewer could",
            "    check by looking at a single rendered frame.",
            "  - Give each subject an approximate height_m and a [x, y, z]",
            "    location in metres. Put the camera-facing side at negative y.",
            "  - The scene is staged with proxy figures, so describe action and",
            "    framing rather than detailed character appearance.",
        ]))

        try:
            response = self.think(
                Capability.PLANNING, prompt,
                system=DIRECTOR_SYSTEM,
                json_schema=PLAN_SCHEMA,
                max_tokens=3000,
                temperature=0.4,
            )
            plan = self._plan_from_response(response.data, target_duration_s)
            plan.synthetic = response.synthetic
            if response.synthetic:
                plan.notes.append(
                    "No planning model was reachable, so this plan came from the "
                    "deterministic fallback rather than a model. Set up LM Studio "
                    "and re-run to get a real plan."
                )
            return plan
        except Exception as exc:  # noqa: BLE001 - never block production on planning
            self.log.warning("planning model failed (%s); using fallback plan", exc)
            plan = self._fallback_plan(objective, target_duration_s)
            plan.notes.append(f"Planning model unavailable ({exc}); used fallback plan.")
            return plan

    def _plan_from_response(self, data: dict | None,
                            target_duration_s: float) -> PlanResult:
        if not data or not data.get("shots"):
            return self._fallback_plan("unspecified", target_duration_s)

        shots: list[ShotSpec] = []
        for index, raw in enumerate(data["shots"]):
            try:
                shots.append(self._shot_from_plan(raw, index))
            except Exception as exc:  # noqa: BLE001 - skip a malformed shot
                self.log.warning("skipping malformed shot %d in plan: %s", index, exc)

        if not shots:
            return self._fallback_plan("unspecified", target_duration_s)

        return PlanResult(
            title=str(data.get("title") or "Untitled"),
            logline=str(data.get("logline") or ""),
            shots=shots,
        )

    @staticmethod
    def _shot_from_plan(raw: dict[str, Any], index: int) -> ShotSpec:
        shot_id = f"SC01_SH{index + 1:02d}"
        mood = str(raw.get("lighting_mood") or "natural")
        preset = LIGHTING_PRESETS.get(mood, LIGHTING_PRESETS["natural"])
        lighting = LightingSpec(mood=mood, **preset)

        subjects = []
        for subject in raw.get("subjects") or []:
            location = subject.get("location") or [0.0, 0.0, 0.0]
            if not isinstance(location, (list, tuple)) or len(location) != 3:
                location = [0.0, 0.0, 0.0]
            subjects.append(SubjectSpec(
                name=str(subject.get("name", "Subject")),
                height_m=float(subject.get("height_m", 1.8)),
                location=tuple(float(v) for v in location),
                action=str(subject.get("action", "")),
                appearance=str(subject.get("appearance", "")),
            ))

        return ShotSpec(
            shot_id=shot_id,
            scene_id="SC01",
            ordinal=index,
            duration_s=max(0.5, float(raw.get("duration_s", 4.0))),
            description=str(raw.get("description", "")),
            camera=CameraSpec(
                shot_size=str(raw.get("shot_size", "medium")),
                angle=str(raw.get("angle", "eye_level")),
                movement=str(raw.get("movement", "static")),
                lens_mm=float(raw.get("lens_mm") or 0.0),
            ),
            lighting=lighting,
            subjects=subjects,
            environment=str(raw.get("environment", "")),
            quality_criteria=[str(c) for c in raw.get("quality_criteria") or []],
            dramatic_purpose=str(raw.get("dramatic_purpose", "")),
            audio=AudioRequirement(),
        )

    @staticmethod
    def _fallback_plan(objective: str, target_duration_s: float) -> PlanResult:
        """A single, well-formed shot that always builds.

        Not a creative substitute for a model — it is a floor, so the pipeline
        can be proven end to end before a model is configured.
        """
        shot = ShotSpec(
            shot_id="SC01_SH01",
            scene_id="SC01",
            ordinal=0,
            duration_s=max(1.0, target_duration_s),
            description=(
                f"A single locked-off medium shot establishing the scene: {objective}"
            ),
            camera=CameraSpec(
                shot_size="medium", angle="eye_level", movement="static", lens_mm=50.0,
            ),
            lighting=LightingSpec(mood="natural", **LIGHTING_PRESETS["natural"]),
            subjects=[SubjectSpec(name="Figure", height_m=1.8,
                                  location=(0.0, 0.0, 0.0), action="standing")],
            environment="neutral studio stage",
            quality_criteria=[
                "The figure is clearly visible and occupies a medium-shot "
                "fraction of the frame height.",
                "The frame is correctly exposed: neither crushed to black nor "
                "clipped to white.",
                "There is visible separation between the figure and the "
                "background.",
            ],
            dramatic_purpose="Establish the scene.",
        )
        return PlanResult(
            title="Fallback single shot",
            logline=objective,
            shots=[shot],
            synthetic=True,
        )

    # -- shot production loop (spec section 6) ----------------------------

    def produce_shot(self, parent: Task, spec: ShotSpec) -> ShotProduction:
        """Build, look at it, fix it, repeat. Then render and encode."""
        self.announce(f"producing {spec.shot_id}")
        result = ShotProduction(shot_id=spec.shot_id, approved=False, version=0, rounds=0)
        current = spec

        for round_no in range(1, self.max_rounds + 1):
            result.rounds = round_no
            result.version = round_no
            self.ctx.workspace.prepare_shot_version(spec.shot_id, round_no)

            try:
                build = self.blender.build_shot(parent, current, round_no, fps=self.fps)
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"round {round_no}: build failed — {exc}")
                self.log.error("build failed on round %d: %s", round_no, exc)
                break

            result.blend_path = build.blend_path
            result.notes.extend(build.notes)

            preview = self.blender.render_preview(parent, current, round_no)
            result.preview_path = str(preview.image_path)

            report = self.vision.inspect(
                parent, current, str(preview.image_path),
                background_path=(str(preview.background_path)
                                 if preview.background_path else None),
                iteration=round_no,
            )

            self.ctx.db.add_shot_version(
                self.ctx.project_id, spec.shot_id, blend_path=build.blend_path,
                render_path=str(preview.image_path), qa=report.to_dict(),
                notes=f"round {round_no}",
            )
            result.history.append({
                "round": round_no,
                "version": round_no,
                "score": report.score,
                "passed": report.passed,
                "findings": [f.to_dict() for f in report.findings],
                "revision": [],
            })

            if report.passed:
                result.approved = True
                result.final_report = report
                self.announce(
                    f"{spec.shot_id} passed review on round {round_no} "
                    f"(score {report.score:.2f})"
                )
                break

            blocking = report.blocking_findings
            if round_no >= self.max_rounds:
                result.final_report = report
                result.notes.append(
                    f"reached the {self.max_rounds}-round limit without passing; "
                    "shipping the best version for the owner to judge"
                )
                break

            revised, applied = self.blender.apply_corrections(current, blocking)
            result.history[-1]["revision"] = applied
            if not applied:
                result.notes.append(
                    f"round {round_no}: reviewer flagged problems but no "
                    "correction could be derived; stopping early"
                )
                result.final_report = report
                break

            if revised.to_dict() == current.to_dict():
                # The correction was a no-op — typically "tighten" on a shot
                # already at the tightest size. Rebuilding would produce a
                # byte-identical frame and we would spin until the round limit.
                # Stop and hand the shot to the owner instead of pretending to
                # have tried.
                result.notes.append(
                    f"round {round_no}: the only available correction would not "
                    "change the shot, so iterating further cannot help"
                )
                result.final_report = report
                break

            self.announce(f"revising {spec.shot_id}: {'; '.join(applied[:3])}")
            current = revised

        if result.final_report is None:
            return result

        # -- final render of whichever version we settled on --
        self.ctx.workspace.prepare_shot_version(spec.shot_id, result.version)
        try:
            frames_dir = self.blender.render_final_sequence(
                parent, current, result.version, fps=self.fps
            )
            result.frames_dir = str(frames_dir)
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"final render failed — {exc}")
            self.log.error("final render failed for %s: %s", spec.shot_id, exc)
            return result

        # -- encode the shot --
        try:
            video = self.ctx.workspace.shot_video(spec.shot_id, result.version)
            self.encoder.encode_frames(frames_dir, video, fps=self.fps)
            result.video_path = str(video)
            # Attach the finished artifacts to this round's existing version
            # row rather than adding another one.
            self.ctx.db.update_shot_version(
                self.ctx.project_id, spec.shot_id, result.version,
                blend_path=result.blend_path,
                render_path=str(frames_dir),
                video_path=str(video),
                qa=result.final_report.to_dict(),
                notes="final",
            )
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"shot encode failed — {exc}")
            self.log.error("encode failed for %s: %s", spec.shot_id, exc)
            return result

        versions = self.ctx.db.list_shot_versions(self.ctx.project_id, spec.shot_id)
        if versions:
            self.ctx.db.approve_shot_version(
                self.ctx.project_id, spec.shot_id, result.version
            )

        if result.approved:
            self.ctx.db.set_shot_status(self.ctx.project_id, spec.shot_id, "APPROVED")
            self.ctx.db.record_qa(
                self.ctx.project_id, "shot", spec.shot_id, "final",
                QAStatus.PASSED, score=result.final_report.score,
                findings=[f.to_dict() for f in result.final_report.findings],
            )
        else:
            self.ctx.db.set_shot_status(self.ctx.project_id, spec.shot_id, "REVIEW")
            self.ctx.db.record_qa(
                self.ctx.project_id, "shot", spec.shot_id, "final",
                QAStatus.NEEDS_REVISION, score=result.final_report.score,
                findings=[f.to_dict() for f in result.final_report.findings],
            )
            # Only bother the owner when the result is genuinely poor.
            if result.final_report.score < 0.35:
                self.ask_owner(
                    f"{spec.shot_id} did not meet its quality criteria after "
                    f"{result.rounds} revision rounds (score "
                    f"{result.final_report.score:.2f}). "
                    "How should I proceed?",
                    ["Ship it anyway", "Try a different interpretation",
                     "Change the creative brief", "Stop and let me look"],
                    urgency="high",
                    context={"shot_id": spec.shot_id,
                             "findings": [f.to_dict() for f in result.final_report.findings]},
                )
        return result

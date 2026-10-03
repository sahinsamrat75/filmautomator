"""The AI visual-director loop, exposed over MCP.

This is the surface that turns an external model from a requester into a
director. The cycle is:

    plan -> build -> render preview -> ACTUAL IMAGE -> AI inspects
         -> AI corrects -> rebuild -> preview again -> approve -> final render

Every tool here is safe by construction. There is no way to run shell commands,
execute Python, or delete a path. Corrections are expressed as ShotSpec field
changes and applied to a structured spec, so a bad correction produces a bad
frame rather than a damaged machine.

The one genuinely irreversible operation -- deleting rendered frames -- lives in
the storage governor and is only reachable once a shot's MP4 has been encoded,
probed and registered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..agents import AgentContext, BlenderAgent, VisionAgent
from ..core.finalization import SHOT_FINAL, finalize_shot, invalidate_shot, resume_point
from ..core.spec import ShotSpec
from ..core.storage import StorageGovernor, StorageRefused, StorageState
from ..core.task import QAStatus, Task
from ..gateway import ModelGateway
from ..post.ffmpeg import FFmpegError, VideoEncoder
from .preview import build_preview_metadata, deliver_preview
from .protocol import json_text, tool_result
from .tools import (
    MCPError,
    ToolContext,
    fail,
    ok,
    optional_float,
    optional_int,
    require_bool,
    require_str,
    tool,
)

log = logging.getLogger(__name__)


def _shot_spec(db: Any, project_id: str, shot_id: str) -> ShotSpec:
    """Load a shot's spec, with a clear error when it is absent."""
    from .tools import MCPError

    shot = db.get_shot(project_id, shot_id)
    if shot is None:
        raise MCPError(-32602, f"no such shot: {shot_id} in project {project_id}")
    spec = shot.get("spec") or {}
    if not spec:
        raise MCPError(-32602, f"shot {shot_id} has no spec recorded")
    spec.setdefault("shot_id", shot_id)
    return ShotSpec.from_dict(spec)


def _agent_context(ctx: ToolContext, project_id: str) -> AgentContext:
    project, workspace = ctx.project_and_workspace(project_id)
    return AgentContext(
        gateway=ModelGateway(ctx.config.gateway),
        config=ctx.config,
        project_id=project_id,
        db=ctx.db,
        workspace=workspace,
        session=ctx.session(),
        events=ctx.events,
    )


def _camera_metadata(spec: ShotSpec) -> dict[str, Any]:
    """The camera facts that change how an image should be read."""
    return {
        "shot_size": spec.camera.shot_size,
        "angle": spec.camera.angle,
        "movement": spec.camera.movement,
        "lens_mm": spec.camera.lens_mm,
        "location": list(spec.camera.location),
        "look_at": list(spec.camera.look_at),
    }


def _render_metadata(ctx: ToolContext) -> dict[str, Any]:
    render = ctx.config.render
    return {
        "preview_engine": render.preview_engine,
        "preview_samples": render.preview_samples,
        "preview_resolution": f"{render.preview_width}x{render.preview_height}",
        "final_engine": render.final_engine,
        "final_samples": render.final_samples,
        "final_resolution": f"{render.final_width}x{render.final_height}",
        "fps": render.fps,
    }


# ===========================================================================
# THE VISUAL DIRECTOR LOOP
# ===========================================================================


@tool("render_shot_preview",
      "Build the shot in Blender, render a preview, and return the ACTUAL IMAGE "
      "for you to look at. This is the core of the visual director loop: the "
      "response contains the rendered pixels, not just a path. Judge the frame, "
      "then call correct_shot to change it and call this again.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "include_image": {"type": "boolean",
                                    "description": "Set false to get metadata "
                                                   "only (no image attached)."}},
      required=["project_id", "shot_id"])
def tool_render_shot_preview(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    project, workspace = ctx.project_and_workspace(project_id)
    spec = _shot_spec(ctx.db, project_id, shot_id)

    context = _agent_context(ctx, project_id)
    blender = BlenderAgent(context, ctx.session(), ctx.config.render)
    task = Task(objective=f"preview {shot_id}", agent="blender_agent",
                project_id=project_id)

    build = blender.build_shot(task, spec, 1)
    preview = blender.render_preview(task, spec, 1)

    mirrored = workspace.mirror_preview(spec.shot_id, 1)
    preview_path = str(mirrored or preview.image_path)
    ctx.db.register_artifact(
        project_id, "preview", preview_path,
        f"{shot_id} preview v001", shot_id=shot_id, scene_id=spec.scene_id,
        metadata={"version": 1, "engine": preview.engine,
                  "blend": build.blend_path,
                  "shot_local_copy": str(preview.image_path)},
    )
    ctx.emit(project_id, "preview.ready",
             f"{shot_id} preview rendered ({preview.engine})",
             agent="blender_agent", shot_id=shot_id,
             payload={"preview": str(preview.image_path)})

    metadata = build_preview_metadata(
        shot_id=shot_id,
        scene_id=spec.scene_id,
        duration_s=spec.duration_s,
        fps=ctx.config.render.fps,
        camera=_camera_metadata(spec),
        render_settings=_render_metadata(ctx),
        artifact_path=preview_path,
        version=1,
        qa=ctx.db.latest_qa(project_id, shot_id, "preview"),
        extra={
            "blend_path": build.blend_path,
            "background_plate": str(preview.background_path or ""),
            "lighting": {"mood": spec.lighting.mood},
            "description": spec.description,
            "subjects": [s.name for s in spec.subjects],
            "environment": spec.environment,
        },
    )

    delivery = deliver_preview(
        shot_id, preview.image_path,
        metadata=metadata,
        client_capabilities=ctx.client_capabilities,
        include_image=require_bool(args, "include_image", True),
    )

    blocks = [json_text(delivery.to_dict())]
    if delivery.image_block:
        blocks.append(delivery.image_block)
    return tool_result(blocks)


@tool("correct_shot",
      "Change a shot's camera, lens, lighting or framing after looking at its "
      "preview. Every field is optional; only what you pass is changed. Then "
      "call render_shot_preview again to see the result.",
      properties={
          "project_id": {"type": "string"},
          "shot_id": {"type": "string"},
          "shot_size": {"type": "string",
                        "description": "extreme_wide, wide, full, medium_wide, "
                                       "medium, medium_close, close_up or "
                                       "extreme_close_up."},
          "angle": {"type": "string",
                    "description": "eye_level, low, high, birds_eye, worms_eye, "
                                   "dutch or over_shoulder."},
          "movement": {"type": "string",
                       "description": "static, dolly_in, dolly_out, pan_left, "
                                      "pan_right, tilt_up, tilt_down and others."},
          "lens_mm": {"type": "number", "description": "Focal length, e.g. 35."},
          "camera_height_m": {"type": "number",
                              "description": "Height of the camera above the "
                                             "ground, e.g. 1.2 for chest height."},
          "camera_distance_m": {"type": "number",
                                "description": "How far the camera sits from the "
                                               "subject."},
          "camera_location": {"type": "array", "items": {"type": "number"},
                              "description": "Explicit [x, y, z] camera position."},
          "look_at": {"type": "array", "items": {"type": "number"},
                      "description": "Explicit [x, y, z] point the camera aims at."},
          "key_energy": {"type": "number"},
          "fill_energy": {"type": "number"},
          "rim_energy": {"type": "number"},
          "world_strength": {"type": "number"},
          "exposure_compensation": {"type": "number"},
          "duration_s": {"type": "number"},
          "description": {"type": "string"},
          "reason": {"type": "string",
                     "description": "Why you are making this change, e.g. 'the "
                                    "head is too close to the top of frame'."},
      },
      required=["project_id", "shot_id"])
def tool_correct_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    _shot_spec(ctx.db, project_id, shot_id)

    spec = _shot_spec(ctx.db, project_id, shot_id)
    applied: list[str] = []

    if value := require_str(args, "shot_size"):
        spec.camera.shot_size = value
        applied.append(f"camera.shot_size = {value}")
    if value := require_str(args, "angle"):
        spec.camera.angle = value
        applied.append(f"camera.angle = {value}")
    if value := require_str(args, "movement"):
        spec.camera.movement = value
        applied.append(f"camera.movement = {value}")
    if (value := optional_float(args, "lens_mm", 0.0)):
        spec.camera.lens_mm = value
        applied.append(f"camera.lens_mm = {value}")

    # Camera height and distance are the two corrections an AI reaches for most
    # often after looking at a frame: "too close" and "headroom is wrong". They
    # are expressed the way a cinematographer thinks, then converted to the
    # world-space position the renderer needs.
    height = optional_float(args, "camera_height_m", 0.0)
    distance = optional_float(args, "camera_distance_m", 0.0)
    if height or distance:
        current = list(spec.camera.location)
        if distance:
            current[1] = -abs(distance)
        if height:
            current[2] = height
        spec.camera.location = (current[0], current[1], current[2])
        applied.append(
            f"camera.location = {tuple(round(c, 3) for c in spec.camera.location)}"
            + (f" (height {height} m)" if height else "")
            + (f" (distance {distance} m)" if distance else "")
        )

    if location := args.get("camera_location"):
        values = [float(v) for v in location][:3]
        if len(values) == 3:
            spec.camera.location = tuple(values)
            applied.append(f"camera.location = {tuple(values)}")
    if look_at := args.get("look_at"):
        values = [float(v) for v in look_at][:3]
        if len(values) == 3:
            spec.camera.look_at = tuple(values)
            applied.append(f"camera.look_at = {tuple(values)}")

    for field_name, attribute in (("key_energy", "key_energy"),
                                  ("fill_energy", "fill_energy"),
                                  ("rim_energy", "rim_energy"),
                                  ("world_strength", "world_strength"),
                                  ("exposure_compensation",
                                   "exposure_compensation")):
        value = optional_float(args, field_name, 0.0)
        if value:
            setattr(spec.lighting, attribute, value)
            applied.append(f"lighting.{attribute} = {value}")

    if (value := optional_float(args, "duration_s", 0.0)):
        spec.duration_s = value
        applied.append(f"duration_s = {value}")
    if value := require_str(args, "description"):
        spec.description = value
        applied.append("description updated")

    if not applied:
        return fail(
            "no changes were requested; pass at least one field to change "
            "(shot_size, lens_mm, camera_height_m, camera_distance_m, "
            "key_energy, ...)"
        )

    ctx.db.upsert_shot(project_id, shot_id, spec.scene_id, spec.ordinal,
                       spec.duration_s, spec.to_dict())
    reason = require_str(args, "reason")
    ctx.emit(project_id, "shot.corrected",
             f"{shot_id} corrected: {', '.join(applied[:3])}"
             + (f" ({reason})" if reason else ""),
             agent="ai_director", shot_id=shot_id,
             payload={"applied": applied, "reason": reason})

    return ok({
        "shot_id": shot_id,
        "applied": applied,
        "reason": reason,
        "spec": spec.to_dict(),
        "next": (f"call render_shot_preview(project_id={project_id!r}, "
                 f"shot_id={shot_id!r}) to see the change"),
    })


@tool("finalize_shot",
      "Render a shot at final quality, encode it, verify it against the "
      "finalization contract, mark it FINAL, and release its frames for "
      "cleanup. Frames are only released once the MP4 has been verified, so a "
      "failed render or a failed encode keeps them.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "width": {"type": "integer"},
                  "height": {"type": "integer"},
                  "engine": {"type": "string"},
                  "release_frames": {"type": "boolean",
                                     "description": "Delete the rendered frames "
                                                    "once the shot is verified "
                                                    "FINAL. Defaults to true."}},
      required=["project_id", "shot_id"])
def tool_finalize_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    project, workspace = ctx.project_and_workspace(project_id)
    spec = _shot_spec(ctx.db, project_id, shot_id)

    if (width := optional_int(args, "width")):
        ctx.config.render.final_width = width
    if (height := optional_int(args, "height")):
        ctx.config.render.final_height = height
    if engine := require_str(args, "engine"):
        ctx.config.render.final_engine = engine

    # Refuse before spending minutes of render time on work that cannot fit.
    governor = _governor(ctx)
    try:
        governor.ensure_capacity(
            needed_bytes=int(ctx.config.storage.estimated_render_gb * 1024 ** 3))
    except StorageRefused as refusal:
        return fail(
            f"refusing to start the final render: {refusal}",
            detail=refusal.usage.to_dict(),
        )

    context = _agent_context(ctx, project_id)
    blender = BlenderAgent(context, ctx.session(), ctx.config.render)
    task = Task(objective=f"final {shot_id}", agent="blender_agent",
                project_id=project_id)

    blender.build_shot(task, spec, 1)
    frames = blender.render_final_sequence(task, spec, 1)

    encoder = VideoEncoder()
    video = workspace.shot_video(shot_id, 1)
    try:
        encoder.encode_frames(frames, video, fps=ctx.config.render.fps)
    except FFmpegError as exc:
        # The encode failed, so the frames are still the only copy. Say so
        # explicitly rather than leaving the caller to infer it.
        return fail(
            f"the render finished but encoding failed: {exc}",
            detail={"frames_kept": str(frames), "shot_id": shot_id,
                    "frames_released": False,
                    "note": "frames were kept because no verified MP4 replaced them"},
        )

    ctx.db.register_artifact(
        project_id, "shot_video", str(video), f"{shot_id} final",
        shot_id=shot_id, scene_id=spec.scene_id,
        metadata={"version": 1, "fps": ctx.config.render.fps,
                  "frames_dir": str(frames)},
    )

    verdict = finalize_shot(
        ctx.db, project_id, shot_id, video,
        expected_duration_s=spec.duration_s, encoder=encoder,
        events=ctx.events,
    )

    payload: dict[str, Any] = {
        "shot_id": shot_id,
        "video": str(video),
        "frames_dir": str(frames),
        **verdict.to_dict(),
    }

    # Frames are released only from a passing verdict. This is the promotion
    # point the whole storage policy depends on.
    if verdict.frames_releasable and require_bool(args, "release_frames", True):
        cleanup = governor.cleanup(project_id=project_id)
        payload["storage"] = {
            "freed_bytes": cleanup["freed_bytes"],
            "freed_gb": cleanup["freed_gb"],
            "removed_count": cleanup["removed_count"],
            "state_after": cleanup["state_after"],
        }
        payload["frames_released"] = True
    else:
        payload["frames_released"] = False
        if not verdict.frames_releasable:
            payload["frames_note"] = (
                "frames were kept: the shot did not satisfy the finalization "
                "contract, so the MP4 is not yet a verified replacement"
            )

    return ok(payload)


@tool("invalidate_shot",
      "Mark a finalized shot INVALIDATED so only that shot is regenerated. "
      "Use this when a finalized shot turns out to be unusable; recovery leaves "
      "every other finalized shot alone.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "reason": {"type": "string"}},
      required=["project_id", "shot_id"])
def tool_invalidate_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    return ok(invalidate_shot(ctx.db, project_id, shot_id,
                              require_str(args, "reason"), events=ctx.events))


@tool("get_resume_point",
      "Where production should resume: the first shot that is not FINAL. "
      "Finalized shots are never rebuilt.",
      properties={"project_id": {"type": "string"}},
      required=["project_id"])
def tool_get_resume_point(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    require_str(args, "project_id")
    ctx.project_and_workspace(project_id)
    return ok(resume_point(ctx.db, project_id))


# ===========================================================================
# STORAGE GOVERNOR
# ===========================================================================


def _governor(ctx: ToolContext) -> StorageGovernor:
    """A storage governor bound to this context's workspace and registry."""
    return StorageGovernor(
        ctx.config.workspace,
        thresholds=ctx.config.storage.thresholds(),
        db=ctx.db,
        encoder=VideoEncoder(),
        events=ctx.events,
    )


@tool("get_storage_status",
      "Working storage: how much Filmautomator is using against its ceiling, "
      "what is reclaimable, and what is being held back because its "
      "replacement is not verified yet. States are NORMAL, WARNING, "
      "AGGRESSIVE_CLEANUP and HARD_LIMIT.",
      properties={"project_id": {"type": "string",
                                 "description": "Report only this project's "
                                                "share of the workspace."}})
def tool_get_storage_status(ctx: ToolContext, args: dict) -> dict:
    governor = _governor(ctx)
    usage = governor.measure()
    payload = usage.to_dict()
    payload["cleanup_enabled"] = ctx.config.storage.cleanup_enabled

    project_id = require_str(args, "project_id")
    if project_id:
        ctx.project_and_workspace(project_id)
        project_root = ctx.config.workspace / project_id
        project_bytes = 0
        for path in project_root.rglob("*") if project_root.is_dir() else []:
            if path.is_file() and not path.is_symlink():
                try:
                    project_bytes += path.stat().st_size
                except OSError:
                    continue
        payload["project_id"] = project_id
        payload["project_bytes"] = project_bytes
        payload["project_gb"] = round(project_bytes / 1024 ** 3, 3)
    return ok(payload)


@tool("cleanup_storage",
      "Delete disposable production artifacts to reclaim space. Only files "
      "whose replacement has been promoted and verified are removed: rendered "
      "frames of a shot survive until that shot's MP4 exists, is non-empty and "
      "probes as valid video. Final movies, shot videos, reports and scenes are "
      "never deleted. Pass dry_run to see what would go without deleting it.",
      properties={"project_id": {"type": "string"},
                  "target_gb": {"type": "number",
                                "description": "Stop once this much has been "
                                               "freed. 0 frees everything safe."},
                  "dry_run": {"type": "boolean"},
                  "confirm": {"type": "boolean",
                              "description": "Required unless dry_run is true."}},
      required=["confirm"])
def tool_cleanup_storage(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    dry_run = require_bool(args, "dry_run", False)

    if not dry_run and not require_bool(args, "confirm", False):
        return fail(
            "refusing to delete anything without confirmation: pass "
            "confirm=true, or dry_run=true to preview what would be removed"
        )
    if project_id:
        ctx.project_and_workspace(project_id)

    governor = _governor(ctx)
    target = int((optional_float(args, "target_gb", 0.0) or 0.0) * 1024 ** 3)
    result = governor.cleanup(target_bytes=target, project_id=project_id,
                              dry_run=dry_run)
    result["rule"] = (
        "an intermediate is deleted only after its replacement has been "
        "promoted and verified"
    )
    if not dry_run and result.get("freed_bytes"):
        ctx.emit(project_id or "system", "storage.cleaned",
                 f"reclaimed {result['freed_gb']:.3f} GB "
                 f"({result['removed_count']} file(s))",
                 agent="storage_governor",
                 payload={"freed_bytes": result["freed_bytes"],
                          "project_id": project_id})
    return ok(result)


@tool("resume_production_from",
      "Resume production from where it stopped. Finalized shots are never "
      "rebuilt: this reports which shots are FINAL, which are pending, and the "
      "shot production should pick up at. Use it after a failure, or to continue "
      "a film part-way.",
      properties={"project_id": {"type": "string"},
                  "objective": {"type": "string",
                                "description": "Objective to record for the "
                                               "resumed run."},
                  "duration_s": {"type": "number"},
                  "offline": {"type": "boolean"},
                  "engine": {"type": "string"},
                  "width": {"type": "integer"},
                  "height": {"type": "integer"},
                  "only_shots": {"type": "array", "items": {"type": "string"},
                                 "description": "Restrict the resumed run to "
                                                "these shot ids."}},
      required=["project_id"])
def tool_resume_production_from(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    ctx.project_and_workspace(project_id)

    resume = resume_point(ctx.db, project_id)
    pending = resume["pending"]
    if not pending:
        return ok({
            "resumed": False,
            **resume,
            "note": ("every shot is FINAL; nothing to resume. Use "
                     "invalidate_shot to force one to be rebuilt."),
        })

    # Only pending shots are rebuilt, so a finalized shot is never re-rendered.
    targets = list(args.get("only_shots") or []) or pending
    ctx.emit(project_id, "production.resumed",
             f"resuming at {resume['resume_shot_id']} "
             f"({len(resume['finalized'])} shot(s) already FINAL)",
             agent="director",
             payload={"resume": resume, "targets": targets})

    started = tool_start_production(ctx, {
        "project_id": project_id,
        "objective": require_str(args, "objective") or "continue this production",
        "duration_s": optional_float(args, "duration_s", 0.0) or 0.0,
        "offline": require_bool(args, "offline", True),
        "engine": require_str(args, "engine"),
        "width": optional_int(args, "width"),
        "height": optional_int(args, "height"),
        "only_shots": targets,
    })
    result = json.loads(started["content"][0]["text"])
    return ok({"resumed": True, **resume, "targets": targets, "run": result})

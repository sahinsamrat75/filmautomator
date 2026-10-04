"""The Filmmaker tool surface.

Every capability the external AI gets is declared here. Deliberately absent:
there is no tool that runs a shell command, none that deletes an arbitrary
path, and none that executes arbitrary code. The Blender control server has an
``evaluate`` operation for the internal agent, but it is not reachable from
MCP — an external model cannot ask this system to run Python.

Destructive operations (``delete_project``) are reversible: the project is
moved aside, never removed.
"""

from __future__ import annotations

import base64
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents import AgentContext, BlenderAgent, VisionAgent
from ..agents.director import Director
from ..core.events import EventKind
from ..core.finalization import SHOT_FINAL, invalidate_shot
from ..core.integrity import verify_project
from ..core.spec import ShotSpec
from ..core.task import QAStatus, Task, TaskStatus
from ..core.workspace import Workspace
from ..gateway import ModelGateway
from ..post.ffmpeg import FFmpegError, VideoEncoder
from ..qa import FinalQA
from ..render_jobs import frames_progress, render_jobs
from ..runtime import RunState
from ..producer import ProductionRequest
from .preview import build_preview_metadata, deliver_preview
from .protocol import image_content, json_text, text_content, tool_result
from .tools import (
    MCPError,
    ToolContext,
    fail,
    get_tool,
    list_tools,
    ok,
    optional_float,
    optional_int,
    require_bool,
    require_project,
    require_str,
    tool,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _image_block(path: str | Path) -> dict[str, Any] | None:
    """Inline an image so the model sees exactly what the owner sees."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    suffix = candidate.suffix.lstrip(".").lower() or "png"
    mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
    return image_content(
        base64.b64encode(candidate.read_bytes()).decode("ascii"), mime
    )


def _agent_context(ctx: ToolContext, project_id: str) -> AgentContext:
    """Build an agent context for ad-hoc (non-production) operations."""
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


def _shot_spec(db, project_id: str, shot_id: str) -> ShotSpec:
    shot = db.get_shot(project_id, shot_id)
    if shot is None:
        raise MCPError(-32602, f"no such shot: {shot_id} in project {project_id}")
    spec = shot.get("spec") or {}
    if not spec:
        raise MCPError(-32602, f"shot {shot_id} has no spec recorded")
    spec.setdefault("shot_id", shot_id)
    return ShotSpec.from_dict(spec)


# ===========================================================================
# PROJECT MANAGEMENT
# ===========================================================================


@tool("create_project",
      "Create a new film project. Returns the project_id used by every other "
      "tool. Does not start production — call start_production when ready.",
      properties={
          "name": {"type": "string",
                   "description": "Project name, e.g. 'AWAKE_EP01'."},
          "objective": {"type": "string",
                        "description": "What the film should be, in plain "
                                       "language. Can also be given to "
                                       "start_production."},
          "duration_s": {"type": "number",
                         "description": "Target runtime in seconds."},
          "style": {"type": "string", "description": "Visual style guidance."},
      },
      required=["name"])
def tool_create_project(ctx: ToolContext, args: dict) -> dict:
    name = require_str(args, "name").strip()
    if not name:
        return fail("name is required")
    objective = require_str(args, "objective")
    duration = optional_float(args, "duration_s", 10.0) or 10.0
    style = require_str(args, "style")

    project_id = ctx.db.create_project(
        name=name, objective=objective, workspace=str(ctx.config.workspace),
        metadata={"duration_s": duration, "style": style,
                  "created_via": "mcp"},
    )
    workspace = Workspace.create(Path(ctx.config.workspace), project_id, name)
    ctx.emit(project_id, EventKind.PROJECT_CREATED, f"Project {name!r} created",
             agent="director", payload={"name": name})
    return ok({
        "project_id": project_id,
        "name": name,
        "workspace": str(workspace.root),
        "final_movie_will_be": str(workspace.final_video()),
    })


@tool("list_projects", "List every film project on this machine.")
def tool_list_projects(ctx: ToolContext, args: dict) -> dict:
    projects = ctx.db.list_projects()
    out = []
    for project in projects:
        progress = ctx.db.progress(project["project_id"])
        out.append({
            "project_id": project["project_id"],
            "name": project["name"],
            "objective": project["objective"],
            "status": project["status"],
            "created_at": project["created_at"],
            "shots": progress["shots_total"],
            "shots_approved": progress["shots_approved"],
        })
    return ok({"projects": out, "count": len(out)})


@tool("get_project", "Full detail for one project, including its artifact paths.",
      properties={"project_id": {"type": "string"}}, required=["project_id"])
def tool_get_project(ctx: ToolContext, args: dict) -> dict:
    project, workspace = ctx.project_and_workspace(require_str(args, "project_id"))
    progress = ctx.db.progress(project["project_id"])
    final = ctx.db.latest_artifact(project["project_id"], "final_movie")
    return ok({
        **project,
        "workspace": str(workspace.root),
        "progress": progress,
        "scenes": len(ctx.db.list_scenes(project["project_id"])),
        "shots": [
            {"shot_id": s["shot_id"], "status": s["status"],
             "duration_s": s["duration_s"]}
            for s in ctx.db.list_shots(project["project_id"])
        ],
        "final_movie": final,
        "run_state": ctx.runner.status() if
        (ctx.runner.run and ctx.runner.run.project_id == project["project_id"])
        else None,
    })


@tool("delete_project",
      "Move a project out of the way. The project is moved to a .trash folder "
      "inside the workspace, not erased, so this is reversible by hand. "
      "Requires confirm=true.",
      properties={
          "project_id": {"type": "string"},
          "confirm": {"type": "boolean",
                      "description": "Must be true. Guards against an "
                                     "accidental destructive call."},
      },
      required=["project_id", "confirm"], destructive=True)
def tool_delete_project(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not require_bool(args, "confirm"):
        return fail(
            "delete_project requires confirm=true. The project will be moved "
            "to <workspace>/.trash/ rather than deleted."
        )
    project, workspace = ctx.project_and_workspace(project_id)

    if ctx.runner.run and ctx.runner.run.project_id == project_id \
            and ctx.runner.is_busy:
        return fail("that project has a production running; stop it first")

    trash = Path(ctx.config.workspace) / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = trash / f"{project_id}-{stamp}"
    moved = False
    if workspace.root.exists():
        shutil.move(str(workspace.root), str(destination))
        moved = True

    ctx.db.update_project(project_id, status="DELETED")
    ctx.emit(project_id, EventKind.PROJECT_DELETED,
             f"Project {project['name']!r} moved to trash", agent="director",
             payload={"moved_to": str(destination) if moved else ""})
    return ok({
        "project_id": project_id,
        "deleted": True,
        "moved_to": str(destination) if moved else "(workspace not present)",
        "note": "Files were moved, not erased. Restore by moving the folder back.",
    })


# ===========================================================================
# PRODUCTION
# ===========================================================================


@tool("start_production",
      "Start producing a film. Returns immediately — production runs in the "
      "background so this call does not block for minutes. Poll "
      "get_production_status, or watch events, to follow progress. Only one "
      "production runs at a time.",
      properties={
          "objective": {"type": "string",
                        "description": "What to make, in plain language. "
                                       "Required unless project_id is given."},
          "project_id": {"type": "string",
                         "description": "Produce into an existing project."},
          "duration_s": {"type": "number", "description": "Target runtime."},
          "style": {"type": "string"},
          "plan_file": {"type": "string",
                        "description": "Path to a JSON shot plan to use "
                                       "verbatim instead of asking the "
                                       "Director to invent one."},
          "offline": {"type": "boolean",
                      "description": "Skip the model entirely and use the "
                                     "deterministic fallback plan."},
          "width": {"type": "integer"}, "height": {"type": "integer"},
          "engine": {"type": "string",
                     "description": "Final render engine: CYCLES, "
                                    "BLENDER_EEVEE or BLENDER_WORKBENCH."},
      })
def tool_start_production(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    objective = require_str(args, "objective")
    if not objective and not project_id:
        return fail("give either an objective or a project_id to produce into")

    cfg = ctx.config
    if (width := optional_int(args, "width")):
        cfg.render.final_width = width
    if (height := optional_int(args, "height")):
        cfg.render.final_height = height
    if engine := require_str(args, "engine"):
        cfg.render.final_engine = engine

    request = ProductionRequest(
        objective=objective,
        duration_s=optional_float(args, "duration_s", 10.0) or 10.0,
        style=require_str(args, "style"),
        plan_file=require_str(args, "plan_file"),
        offline=require_bool(args, "offline", False),
        project_id=project_id,
    )
    try:
        run = ctx.runner.start(request, project_id=project_id)
    except RuntimeError as exc:
        return fail(str(exc))
    return ok({
        "started": True,
        "project_id": run.project_id or project_id,
        "state": run.state.value,
        "note": "Production runs in the background. Poll get_production_status.",
    })


@tool("pause_production",
      "Pause the running production at the next safe point. A render in flight "
      "finishes first; nothing is left half-written.")
def tool_pause_production(ctx: ToolContext, args: dict) -> dict:
    return ok({"paused": ctx.runner.pause(), **ctx.runner.status()})


@tool("resume_production", "Resume a paused production.")
def tool_resume_production(ctx: ToolContext, args: dict) -> dict:
    return ok({"resumed": ctx.runner.resume(), **ctx.runner.status()})


@tool("stop_production",
      "Stop the running production. It halts at the next safe point and "
      "whatever has been produced so far is kept.")
def tool_stop_production(ctx: ToolContext, args: dict) -> dict:
    return ok({"stopping": ctx.runner.stop(), **ctx.runner.status()})


@tool("get_production_status",
      "Current production state: run state, progress, current shot and stage, "
      "and what the agents are doing right now. Pass a project_id to ask about "
      "a project rather than the live run; the answer is re-checked against the "
      "files on disk, so COMPLETED means the film is there now.",
      properties={"project_id": {"type": "string",
                                 "description": "Optional. Report this project "
                                                "instead of the live run."}})
def tool_get_production_status(ctx: ToolContext, args: dict) -> dict:
    return ok(ctx.runner.status(require_str(args, "project_id")))


# ===========================================================================
# DIRECTOR
# ===========================================================================


@tool("submit_objective",
      "Give the Director a creative objective. This is how you say things like "
      "'make Akira's entrance more dramatic' — the Director decides what "
      "production work that implies. Starts production.",
      properties={
          "objective": {"type": "string"},
          "project_id": {"type": "string"},
          "duration_s": {"type": "number"},
          "style": {"type": "string"},
      },
      required=["objective"])
def tool_submit_objective(ctx: ToolContext, args: dict) -> dict:
    return tool_start_production(ctx, args)


@tool("get_director_status",
      "What the Director is working on: the active plan, its shots, and any "
      "decision waiting on a human.",
      properties={"project_id": {"type": "string"}})
def tool_get_director_status(ctx: ToolContext, args: dict) -> dict:
    run = ctx.runner.run
    project_id = require_str(args, "project_id") or (run.project_id if run else "")
    if not project_id:
        return ok({"state": "IDLE", "note": "no active production"})

    shots = ctx.db.list_shots(project_id)
    pending = ctx.db.pending_decisions(project_id)
    return ok({
        "project_id": project_id,
        "run": ctx.runner.status() if run and run.project_id == project_id else None,
        "activity": ctx.events.current_activity(project_id),
        "shots": [
            {"shot_id": s["shot_id"], "status": s["status"],
             "description": (s.get("spec") or {}).get("description", "")}
            for s in shots
        ],
        "pending_decisions": pending,
    })


@tool("get_director_decision",
      "List creative decisions waiting on a human. Only genuinely ambiguous or "
      "destructive choices are escalated.",
      properties={"project_id": {"type": "string"}})
def tool_get_director_decision(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"pending": [], "count": 0})
    pending = ctx.db.pending_decisions(project_id)
    return ok({"pending": pending, "count": len(pending)})


@tool("approve_director_decision",
      "Answer a decision the Director escalated. The answer must be one of the "
      "options offered, or free text if the decision allows it.",
      properties={
          "decision_id": {"type": "string"},
          "answer": {"type": "string"},
      },
      required=["decision_id", "answer"])
def tool_approve_director_decision(ctx: ToolContext, args: dict) -> dict:
    decision_id = require_str(args, "decision_id")
    answer = require_str(args, "answer")
    decision = ctx.db.get_decision(decision_id)
    if decision is None:
        return fail(f"no such decision: {decision_id}")
    if decision["status"] != "PENDING":
        return fail(f"decision {decision_id} is already {decision['status']}")
    ctx.db.resolve_decision(decision_id, answer)
    ctx.emit(decision["project_id"], EventKind.DECISION_RESOLVED,
             f"Owner decided: {answer}", agent="owner",
             payload={"decision_id": decision_id, "answer": answer})
    return ok({"decision_id": decision_id, "answer": answer, "resolved": True})


# ===========================================================================
# TASKS
# ===========================================================================


@tool("list_tasks",
      "List production tasks, optionally filtered by status.",
      properties={
          "project_id": {"type": "string"},
          "status": {"type": "string",
                     "description": "PENDING, PLANNING, RUNNING, WAITING, "
                                    "REVIEW, FAILED, RETRYING, APPROVED, "
                                    "COMPLETED or BLOCKED."},
      })
def tool_list_tasks(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"tasks": [], "count": 0})

    status_text = require_str(args, "status").upper()
    status = None
    if status_text:
        try:
            status = TaskStatus(status_text)
        except ValueError:
            return fail(f"unknown status {status_text!r}")
    tasks = ctx.db.list_tasks(project_id, status)
    return ok({
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks),
    })


@tool("get_task", "Full detail for one task, including its artifacts and notes.",
      properties={"task_id": {"type": "string"}}, required=["task_id"])
def tool_get_task(ctx: ToolContext, args: dict) -> dict:
    task = ctx.db.get_task(require_str(args, "task_id"))
    if task is None:
        return fail("no such task")
    return ok(task.to_dict())


@tool("retry_task",
      "Retry a failed task. For a shot task this re-runs production of that "
      "shot. Other task kinds are reset to PENDING and re-run on the next "
      "production.",
      properties={"task_id": {"type": "string"}}, required=["task_id"])
def tool_retry_task(ctx: ToolContext, args: dict) -> dict:
    task = ctx.db.get_task(require_str(args, "task_id"))
    if task is None:
        return fail("no such task")
    if ctx.runner.is_busy:
        return fail("a production is already running; stop it before retrying")

    shot_id = require_str(task.inputs, "shot_id")
    if not shot_id:
        # The task objective carries the shot id for shot tasks.
        for token in task.objective.split():
            if token.startswith("SC") and "_SH" in token:
                shot_id = token.strip(":")
                break

    if shot_id:
        task.retry_count += 1
        task.touch(TaskStatus.RETRYING)
        ctx.db.save_task(task)
        request = ProductionRequest(
            objective="", project_id=task.project_id, only_shots=[shot_id],
        )
        try:
            run = ctx.runner.start(request, project_id=task.project_id)
        except RuntimeError as exc:
            return fail(str(exc))
        return ok({"retrying": True, "shot_id": shot_id,
                   "project_id": task.project_id, "state": run.state.value})

    task.retry_count += 1
    task.touch(TaskStatus.PENDING)
    task.error = ""
    ctx.db.save_task(task)
    return ok({
        "retrying": False,
        "reset_to": "PENDING",
        "note": "This task is not tied to a single shot, so it will re-run on "
                "the next start_production rather than on its own.",
    })


@tool("cancel_task", "Cancel a task that has not completed.",
      properties={"task_id": {"type": "string"},
                  "reason": {"type": "string"}},
      required=["task_id"])
def tool_cancel_task(ctx: ToolContext, args: dict) -> dict:
    task = ctx.db.get_task(require_str(args, "task_id"))
    if task is None:
        return fail("no such task")
    if task.status.is_terminal:
        return fail(f"task is already {task.status.value}")
    reason = require_str(args, "reason") or "cancelled"
    task.touch(TaskStatus.FAILED)
    task.error = f"cancelled: {reason}"
    ctx.db.save_task(task)
    ctx.emit(task.project_id, EventKind.TASK_FAILED, task.error,
             agent=task.agent, task_id=task.task_id)
    return ok({"task_id": task.task_id, "status": "FAILED", "reason": reason})


@tool("approve_task", "Mark a task's work as approved.",
      properties={"task_id": {"type": "string"}}, required=["task_id"])
def tool_approve_task(ctx: ToolContext, args: dict) -> dict:
    task = ctx.db.get_task(require_str(args, "task_id"))
    if task is None:
        return fail("no such task")
    task.qa_status = QAStatus.PASSED
    task.touch(TaskStatus.APPROVED)
    ctx.db.save_task(task)
    return ok({"task_id": task.task_id, "status": "APPROVED"})


@tool("reject_task", "Reject a task's work and record why.",
      properties={"task_id": {"type": "string"}, "reason": {"type": "string"}},
      required=["task_id"])
def tool_reject_task(ctx: ToolContext, args: dict) -> dict:
    task = ctx.db.get_task(require_str(args, "task_id"))
    if task is None:
        return fail("no such task")
    reason = require_str(args, "reason") or "rejected by reviewer"
    task.qa_status = QAStatus.FAILED
    task.note(f"rejected: {reason}")
    task.touch(TaskStatus.REVIEW)
    ctx.db.save_task(task)
    return ok({"task_id": task.task_id, "status": "REVIEW", "reason": reason})


# ===========================================================================
# SCENES
# ===========================================================================


@tool("list_scenes", "List the scenes in a project.",
      properties={"project_id": {"type": "string"}})
def tool_list_scenes(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"scenes": [], "count": 0})
    scenes = ctx.db.list_scenes(project_id)
    for scene in scenes:
        scene["shots"] = [
            s["shot_id"] for s in ctx.db.list_shots(project_id, scene["scene_id"])
        ]
    return ok({"scenes": scenes, "count": len(scenes)})


@tool("get_scene", "Detail for one scene and the shots it contains.",
      properties={"project_id": {"type": "string"},
                  "scene_id": {"type": "string"}},
      required=["scene_id"])
def tool_get_scene(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    scene_id = require_str(args, "scene_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    scenes = [s for s in ctx.db.list_scenes(project_id) if s["scene_id"] == scene_id]
    if not scenes:
        return fail(f"no such scene: {scene_id}")
    shots = ctx.db.list_shots(project_id, scene_id)
    return ok({**scenes[0], "shots": [s["spec"] | {"status": s["status"]}
                                      for s in shots]})


@tool("create_scene",
      "Create a scene to group shots under.",
      properties={
          "project_id": {"type": "string"},
          "scene_id": {"type": "string", "description": "e.g. SC02"},
          "title": {"type": "string"},
          "synopsis": {"type": "string"},
          "ordinal": {"type": "integer"},
      },
      required=["project_id", "scene_id"])
def tool_create_scene(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    scene_id = require_str(args, "scene_id")
    require_project(ctx.db, project_id)
    ordinal = optional_int(args, "ordinal", len(ctx.db.list_scenes(project_id)))
    ctx.db.upsert_scene(
        project_id, scene_id, ordinal or 0,
        require_str(args, "title"), {"synopsis": require_str(args, "synopsis")},
    )
    return ok({"scene_id": scene_id, "project_id": project_id, "created": True})


@tool("render_scene_preview",
      "Render a fresh preview of every shot in a scene and return the images. "
      "This builds each shot in Blender, so it takes a few seconds per shot.",
      properties={"project_id": {"type": "string"},
                  "scene_id": {"type": "string"}},
      required=["project_id", "scene_id"])
def tool_render_scene_preview(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    scene_id = require_str(args, "scene_id")
    require_project(ctx.db, project_id)
    shots = ctx.db.list_shots(project_id, scene_id)
    if not shots:
        return fail(f"scene {scene_id} has no shots")

    context = _agent_context(ctx, project_id)
    blender = BlenderAgent(context, ctx.session(), ctx.config.render)
    blocks: list[dict[str, Any]] = []
    rendered: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        spec = _shot_spec(ctx.db, project_id, shot["shot_id"])
        result = blender.build_shot(Task(objective="scene preview",
                                         agent="blender_agent",
                                         project_id=project_id),
                                    spec, 1)
        preview = blender.render_preview(
            Task(objective="scene preview", agent="blender_agent",
                 project_id=project_id), spec, 1
        )
        image = _image_block(preview.image_path)
        if image:
            blocks.append(text_content(f"{spec.shot_id}:"))
            blocks.append(image)
        rendered.append({"shot_id": spec.shot_id, "preview": str(preview.image_path),
                         "blend": result.blend_path})
    return tool_result(blocks or [json_text({"rendered": rendered})])


# ===========================================================================
# SHOTS
# ===========================================================================


@tool("list_shots", "List the shots in a project.",
      properties={"project_id": {"type": "string"},
                  "scene_id": {"type": "string"}})
def tool_list_shots(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"shots": [], "count": 0})
    scene_id = require_str(args, "scene_id") or None
    shots = ctx.db.list_shots(project_id, scene_id)
    out = []
    for shot in shots:
        spec = shot.get("spec") or {}
        out.append({
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "ordinal": shot.get("ordinal", 0),
            "status": shot["status"],
            "duration_s": shot["duration_s"],
            "description": spec.get("description", ""),
            "shot_size": (spec.get("camera") or {}).get("shot_size", ""),
            "approved_version": shot.get("approved_version"),
        })
    return ok({"shots": out, "count": len(out)})


@tool("get_shot", "Full specification and version history for one shot.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"}},
      required=["shot_id"])
def tool_get_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    shot = ctx.db.get_shot(project_id, shot_id)
    if shot is None:
        return fail(f"no such shot: {shot_id}")
    versions = ctx.db.list_shot_versions(project_id, shot_id)
    return ok({
        **shot,
        "versions": [
            {"version": v.version_no, "approved": v.approved,
             "blend": v.blend_path, "video": v.video_path,
             "qa_score": (v.qa or {}).get("score")}
            for v in versions
        ],
    })


@tool("create_shot",
      "Add or replace a shot specification in a project. Give the camera and "
      "lighting fields you care about; the rest default sensibly.",
      properties={
          "project_id": {"type": "string"},
          "shot_id": {"type": "string", "description": "e.g. SC01_SH04"},
          "description": {"type": "string"},
          "duration_s": {"type": "number"},
          "shot_size": {"type": "string",
                        "description": "extreme_wide, wide, full, medium_wide, "
                                       "medium, medium_close, close_up or "
                                       "extreme_close_up."},
          "angle": {"type": "string",
                    "description": "eye_level, low, high, birds_eye, worms_eye, "
                                   "dutch or over_shoulder."},
          "lens_mm": {"type": "number"},
          "movement": {"type": "string",
                       "description": "static, dolly_in, dolly_out, pan_left, "
                                      "pan_right, tilt_up, tilt_down, crane_up "
                                      "or handheld."},
          "camera_location": {"type": "array", "items": {"type": "number"},
                              "description": "Explicit [x, y, z] camera position."},
          "look_at": {"type": "array", "items": {"type": "number"},
                      "description": "Explicit [x, y, z] point the camera aims at."},
          "lighting_mood": {"type": "string",
                            "description": "natural, golden_hour, blue_hour, "
                                           "overcast, harsh_noon, moonlight, "
                                           "candlelit, neon, firelight, "
                                           "high_key, low_key or silhouette."},
          "environment": {"type": "string"},
          "subject_name": {"type": "string"},
          "subject_height_m": {"type": "number"},
          "subject_action": {"type": "string",
                             "description": "idle, walk, enter, reach, lean or "
                                            "react. Drives the character pose."},
      },
      required=["project_id", "shot_id"])
def tool_create_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    require_project(ctx.db, project_id)
    scene_id = require_str(args, "scene_id") or "SC01"

    # Ordinal: deterministic, unique within the scene. An existing shot keeps
    # its ordinal (re-creating must not renumber the film); a new shot takes
    # the next free position. Assembly order depends on this being stable.
    existing_row = ctx.db.get_shot(project_id, shot_id)
    if existing_row is not None:
        ordinal = int(existing_row.get("ordinal") or 0)
    else:
        siblings = ctx.db.list_shots(project_id, scene_id)
        ordinal = len(siblings)

    spec = {
        "shot_id": shot_id,
        "scene_id": scene_id,
        "ordinal": ordinal,
        "description": require_str(args, "description"),
        "duration_s": optional_float(args, "duration_s", 4.0) or 4.0,
        "camera": {
            "shot_size": require_str(args, "shot_size") or "medium",
            "angle": require_str(args, "angle") or "eye_level",
        },
        "lighting": {"mood": require_str(args, "lighting_mood") or "natural"},
        "environment": require_str(args, "environment"),
        "subjects": [],
        "quality_criteria": [],
    }
    if args.get("lens_mm") is not None:
        spec["camera"]["lens_mm"] = optional_float(args, "lens_mm", 0.0) or 0.0
    if movement := require_str(args, "movement"):
        spec["camera"]["movement"] = movement
    if camera_location := args.get("camera_location"):
        values = [float(v) for v in camera_location][:3]
        if len(values) == 3:
            spec["camera"]["location"] = values
    if look_at := args.get("look_at"):
        values = [float(v) for v in look_at][:3]
        if len(values) == 3:
            spec["camera"]["look_at"] = values
    if name := require_str(args, "subject_name"):
        subject: dict[str, Any] = {
            "name": name,
            "height_m": optional_float(args, "subject_height_m", 1.8) or 1.8,
            "location": [0.0, 0.0, 0.0],
        }
        if action := require_str(args, "subject_action"):
            subject["action"] = action
        spec["subjects"] = [subject]

    parsed = ShotSpec.from_dict(spec)
    ctx.db.upsert_shot(project_id, shot_id, parsed.scene_id, parsed.ordinal,
                       parsed.duration_s, parsed.to_dict())
    ctx.emit(project_id, EventKind.SHOT_CREATED, f"Shot {shot_id} defined",
             agent="director", shot_id=shot_id, payload={"spec": parsed.to_dict()})
    return ok({"shot_id": shot_id, "created": True, "spec": parsed.to_dict()})


@tool("render_shot_final",
      "Render a shot at final quality and encode it to video. Does not run "
      "the Vision review loop — use start_production for that.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "width": {"type": "integer"}, "height": {"type": "integer"},
                  "engine": {"type": "string"}},
      required=["project_id", "shot_id"])
def tool_render_shot_final(ctx: ToolContext, args: dict) -> dict:
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

    frames_dir = workspace.shot_render_dir(shot_id, 1)
    frames_total = spec.frame_count_at(ctx.config.render.fps)

    def body() -> dict[str, Any]:
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
            raise RuntimeError(
                f"render succeeded but encoding failed: {exc} "
                f"(frames kept at {frames})"
            ) from exc

        ctx.db.register_artifact(project_id, "shot_video", str(video),
                                 f"{shot_id} final", shot_id=shot_id)
        ctx.emit(project_id, EventKind.RENDER_COMPLETED,
                 f"{shot_id} rendered and encoded", agent="blender_agent",
                 shot_id=shot_id, payload={"video": str(video)})
        return {"shot_id": shot_id, "frames": str(frames), "video": str(video),
                "frames_dir": str(frames)}

    job = render_jobs().submit(
        "final_render", body, project_id=project_id, shot_id=shot_id,
        label=f"final render {shot_id}",
        progress=frames_progress(str(frames_dir), frames_total),
    )
    return ok({
        "shot_id": shot_id,
        "job_id": job.job_id,
        "state": job.state.value,
        "note": ("the render is running in the background; poll get_render "
                 "with this shot_id until state is COMPLETED or FAILED. The "
                 "MCP server stays responsive while it renders."),
    })


@tool("approve_shot",
      "Approve a shot, recording which preview version you are approving as "
      "the visual basis. Works from previews: a final render is NOT required "
      "to approve. Finalization later renders at final quality against that "
      "approved look.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "version": {"type": "integer",
                              "description": "Preview version to approve. "
                                             "Defaults to the latest preview."}},
      required=["project_id", "shot_id"])
def tool_approve_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    require_project(ctx.db, project_id)

    # The candidate set is the shot's PREVIEW versions — what the director
    # actually looked at. Approval is a statement about a look, so it binds
    # to a preview whether or not a final render exists yet.
    preview_versions: list[int] = []
    for render in ctx.db.list_renders(project_id, shot_id):
        if render.get("kind") == "preview":
            version_no = int(render.get("version_no") or 0)
            if version_no and version_no not in preview_versions:
                preview_versions.append(version_no)
    if not preview_versions:
        # Fall back to the artifact registry for previews recorded there.
        for artifact in ctx.db.list_artifacts(project_id):
            if (artifact.get("kind") == "preview"
                    and artifact.get("shot_id") == shot_id):
                metadata = artifact.get("metadata") or {}
                version_no = int(metadata.get("version") or 0) if isinstance(
                    metadata, dict) else 0
                if version_no and version_no not in preview_versions:
                    preview_versions.append(version_no)
    preview_versions.sort()

    if not preview_versions:
        return fail(
            f"{shot_id} has no rendered preview to approve — call "
            f"render_shot_preview first"
        )

    requested = optional_int(args, "version")
    if requested is not None:
        if requested not in preview_versions:
            return fail(
                f"{shot_id} has no preview v{requested:03d}; known preview "
                f"versions: {preview_versions}"
            )
        version = requested
    else:
        version = preview_versions[-1]

    # Make sure the version round exists, then mark it approved. The preview
    # path is recorded as the render basis on that same row, so the approved
    # look stays traceable to the exact image that was accepted.
    preview_path = ""
    for render in ctx.db.list_renders(project_id, shot_id):
        if (render.get("kind") == "preview"
                and int(render.get("version_no") or 0) == version):
            preview_path = str(render.get("path") or "")
            break
    known = {v.version_no for v in ctx.db.list_shot_versions(project_id, shot_id)}
    if version not in known:
        ctx.db.add_shot_version(project_id, shot_id, version_no=version,
                                render_path=preview_path,
                                notes="approved from preview")
    else:
        ctx.db.update_shot_version(project_id, shot_id, version,
                                   render_path=preview_path)
    ctx.db.approve_shot_version(project_id, shot_id, version)
    ctx.emit(project_id, EventKind.SHOT_APPROVED,
             f"{shot_id} approved on preview v{version:03d}", agent="owner",
             shot_id=shot_id,
             payload={"version": version, "preview_path": preview_path,
                      "basis": "preview"})
    return ok({
        "shot_id": shot_id,
        "approved_version": version,
        "basis": "preview",
        "preview_path": preview_path,
        "preview_versions": preview_versions,
        "note": ("approval recorded against the preview; finalize_shot will "
                 "render the final quality output for this look"),
    })


@tool("reject_shot", "Reject a shot and record why.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "reason": {"type": "string"}},
      required=["project_id", "shot_id"])
def tool_reject_shot(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    reason = require_str(args, "reason") or "rejected by owner"
    ctx.db.set_shot_status(project_id, shot_id, "REVIEW")
    ctx.emit(project_id, EventKind.SHOT_REJECTED,
             f"{shot_id} rejected: {reason}", agent="owner", shot_id=shot_id,
             payload={"reason": reason})
    return ok({"shot_id": shot_id, "status": "REVIEW", "reason": reason})


# ===========================================================================
# BLENDER
# ===========================================================================


@tool("inspect_blender",
      "Is Blender running, what version, and what is currently loaded.")
def tool_inspect_blender(ctx: ToolContext, args: dict) -> dict:
    try:
        session = ctx.session()
        info = session.ping()
    except Exception as exc:  # noqa: BLE001
        return fail(f"Blender is not available: {exc}")
    return ok({
        "running": session.running,
        "version": info.get("blender_version"),
        "filepath": info.get("filepath") or "(unsaved)",
        "object_count": info.get("object_count"),
        "restarts": session.restart_count,
    })


@tool("inspect_scene",
      "Full structured state of the scene currently in Blender: objects, "
      "cameras, lights, materials, timing and render settings.")
def tool_inspect_scene(ctx: ToolContext, args: dict) -> dict:
    return ok(ctx.session().scene_state())


@tool("inspect_objects",
      "Just the object list with transforms — a lighter read than "
      "inspect_scene.")
def tool_inspect_objects(ctx: ToolContext, args: dict) -> dict:
    state = ctx.session().scene_state()
    return ok({
        "objects": [
            {"name": o["name"], "type": o["type"], "location": o["location"],
             "rotation_euler": o["rotation_euler"], "scale": o["scale"],
             "dimensions": o.get("dimensions")}
            for o in state["objects"]
        ],
        "count": state["object_count"],
    })


@tool("get_viewport_preview",
      "A quick image of the current Blender scene from the active camera. "
      "Cheaper and lower fidelity than render_shot_preview.",
      properties={"width": {"type": "integer"}, "height": {"type": "integer"}})
def tool_get_viewport_preview(ctx: ToolContext, args: dict) -> dict:
    session = ctx.session()
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="fa_viewport_")) / "viewport.png"
    result = session.viewport_screenshot(
        path,
        width=optional_int(args, "width", 640) or 640,
        height=optional_int(args, "height", 360) or 360,
    )
    image = _image_block(path)
    if image is None:
        return fail("viewport render produced no image")
    return tool_result([json_text({"path": result["filepath"],
                                   "mode": result.get("mode", "")}), image])


# ===========================================================================
# VISION
# ===========================================================================


@tool("inspect_preview",
      "Run the Vision Agent over a shot's latest preview and return its "
      "evaluation: pass or fail, a score, and per-issue findings.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"}},
      required=["project_id", "shot_id"])
def tool_inspect_preview(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    project, workspace = ctx.project_and_workspace(project_id)

    preview = ctx.db.latest_artifact(project_id, "preview", shot_id)
    if preview is None or not Path(preview["path"]).is_file():
        return fail(
            f"{shot_id} has no preview yet. Call render_shot_preview first.",
        )

    spec = _shot_spec(ctx.db, project_id, shot_id)
    context = _agent_context(ctx, project_id)
    vision = VisionAgent(context)
    plate = preview["path"].replace("preview.png", "preview_background.png")
    task = Task(objective=f"inspect {shot_id}", agent="vision_agent",
                project_id=project_id)
    report = vision.inspect(
        task, spec, preview["path"],
        background_path=plate if Path(plate).is_file() else None,
    )
    return ok({"shot_id": shot_id, **report.to_dict()})


@tool("get_visual_evaluation",
      "The most recent stored Vision Agent evaluation for a shot.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"}},
      required=["project_id", "shot_id"])
def tool_get_visual_evaluation(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    record = ctx.db.latest_qa(project_id, shot_id, "preview")
    if record is None:
        return ok({"shot_id": shot_id, "evaluation": None,
                   "note": "no evaluation recorded yet"})
    return ok({"shot_id": shot_id, "evaluation": record})


# ===========================================================================
# ARTIFACTS
# ===========================================================================


@tool("list_artifacts",
      "Everything a project has produced: previews, renders, videos, reports. "
      "Each entry reports whether the file is actually present on disk.",
      properties={"project_id": {"type": "string"},
                  "kind": {"type": "string",
                           "description": "preview, render, shot_video, "
                                          "final_movie, timeline, qa_report, "
                                          "report."}})
def tool_list_artifacts(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"artifacts": [], "count": 0})
    kind = require_str(args, "kind") or None
    artifacts = ctx.db.list_artifacts(project_id, kind)
    missing = [a for a in artifacts if not a["exists"]]
    return ok({
        "artifacts": artifacts,
        "count": len(artifacts),
        "missing": len(missing),
        "note": (f"{len(missing)} registered artifact(s) are no longer on disk"
                 if missing else "every registered artifact is present"),
    })


@tool("get_artifact", "One artifact by id.",
      properties={"artifact_id": {"type": "string"}}, required=["artifact_id"])
def tool_get_artifact(ctx: ToolContext, args: dict) -> dict:
    artifact = ctx.db.get_artifact(require_str(args, "artifact_id"))
    if artifact is None:
        return fail("no such artifact")
    return ok(artifact)


@tool("get_preview",
      "Return the latest preview image for a shot (or the project) as an actual "
      "image you can look at, with its camera, duration and render settings. "
      "If your client cannot receive images the response says so explicitly.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "include_image": {"type": "boolean",
                                    "description": "Set false for metadata only."}})
def tool_get_preview(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return fail("project_id is required")

    artifact = ctx.db.latest_artifact(project_id, "preview", shot_id or None)
    if artifact is None:
        return fail("no preview has been produced yet; call render_shot_preview")

    path = Path(artifact["path"])
    if not path.is_file() and artifact.get("shot_id"):
        # Once a shot is FINAL the storage governor releases the in-shot preview
        # along with its frames. The mirrored copy under previews/ survives,
        # because a small still image is worth keeping as the shot's visual
        # record — fall back to it rather than reporting the preview as lost.
        mirror = ctx.config.workspace / project_id / "previews" / (
            f"{artifact['shot_id']}_v001.png"
        )
        if mirror.is_file():
            path = mirror

    spec: dict = {}
    if artifact.get("shot_id"):
        try:
            spec = ShotSpec.from_dict(
                (ctx.db.get_shot(project_id, artifact["shot_id"]) or {})
                .get("spec") or {}
            ).to_dict()
        except Exception:  # noqa: BLE001 - metadata is a bonus, not required
            spec = {}

    delivery = deliver_preview(
        artifact.get("shot_id") or shot_id or "",
        path,
        metadata=build_preview_metadata(
            shot_id=artifact.get("shot_id", ""),
            scene_id=artifact.get("scene_id", ""),
            duration_s=spec.get("duration_s", 0.0),
            fps=ctx.config.render.fps,
            camera=(spec.get("camera") or {}),
            render_settings={
                "preview_engine": ctx.config.render.preview_engine,
                "preview_resolution":
                    f"{ctx.config.render.preview_width}x"
                    f"{ctx.config.render.preview_height}",
            },
            artifact_path=str(path),
            extra={"created_at": artifact.get("created_at", "")},
        ),
        client_capabilities=ctx.client_capabilities,
        include_image=require_bool(args, "include_image", True),
    )
    blocks = [json_text(delivery.to_dict())]
    if delivery.image_block:
        blocks.append(delivery.image_block)
    return tool_result(blocks)


@tool("get_render",
      "Render lifecycle and output. While a final render or finalize job is "
      "running this returns its state (QUEUED, RENDERING, COMPLETED, FAILED, "
      "CANCELLED) plus frame progress straight from the job registry — it "
      "never waits on Blender. When the job completed, its full result "
      "(frames, video, verification) is included. With no job it falls back "
      "to the last rendered frames on disk.",
      properties={"project_id": {"type": "string"},
                  "shot_id": {"type": "string"},
                  "job_id": {"type": "string",
                             "description": "Pin one specific render job."}})
def tool_get_render(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    shot_id = require_str(args, "shot_id")

    # Lifecycle first: job state is read from the registry, never from
    # Blender, so this stays responsive no matter what Blender is doing.
    jobs = render_jobs()
    job_id = require_str(args, "job_id")
    job = jobs.get(job_id) if job_id else None
    if job is None and project_id:
        job = jobs.latest(project_id=project_id, shot_id=shot_id or "")
    if job is not None:
        payload = {"render_job": job.to_dict()}
        if job.terminal and job.state.value == "COMPLETED" and job.result:
            payload["result"] = job.result
        if not job.terminal:
            payload["note"] = (
                "a render/finalize job is in progress; this response came "
                "from the job registry and did not wait for Blender"
            )
        return ok(payload)

    artifact = ctx.db.latest_artifact(project_id, "render", shot_id or None)
    if artifact is None:
        return fail("no final render has been produced yet")
    from ..post.ffmpeg import find_frame_sequence

    frames = find_frame_sequence(artifact["path"])
    return ok({**artifact, "frame_count": len(frames),
               "first_frame": str(frames[0]) if frames else "",
               "state": "COMPLETED"})


@tool("cancel_render",
      "Cancel a queued render job before it starts. A job already RENDERING "
      "inside Blender cannot be interrupted without risking the scene, and "
      "this says so instead of pretending to stop it.",
      properties={"job_id": {"type": "string"}},
      required=["job_id"])
def tool_cancel_render(ctx: ToolContext, args: dict) -> dict:
    job_id = require_str(args, "job_id")
    ok_cancel, message = render_jobs().cancel(job_id)
    if not ok_cancel:
        return fail(message)
    return ok({"job_id": job_id, "state": "CANCELLED", "note": message})


@tool("get_final_movie",
      "The finished film: path, duration, resolution, frame rate and codec. "
      "The file is verified on disk and probed before this returns, so a path "
      "from here is one that exists.",
      properties={"project_id": {"type": "string"}})
def tool_get_final_movie(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return fail("project_id is required")

    artifact = ctx.db.latest_artifact(project_id, "final_movie")
    if artifact is None or not Path(artifact["path"]).is_file():
        integrity = verify_project(ctx.db, project_id, encoder=VideoEncoder(),
                                   probe_final=False)
        return fail(
            "this project has no finished movie on disk"
            + (f" — {integrity.summary}" if integrity.problems else "")
            + ". Check get_production_status; the run may have been marked "
              "COMPLETED before its output went missing.",
            detail=integrity.to_dict() if integrity.problems else None,
        )

    encoder = VideoEncoder()
    info: dict[str, Any] = {}
    try:
        probe = encoder.probe(artifact["path"])
        info = {
            "duration_s": round(probe.duration_s, 2),
            "resolution": f"{probe.width}x{probe.height}",
            "fps": round(probe.fps, 2),
            "codec": probe.video_codec,
            "has_audio": probe.has_audio,
            "size_mb": round(Path(artifact["path"]).stat().st_size / 1e6, 2),
            "verified": True,
        }
    except FFmpegError as exc:
        return fail(
            f"the final movie exists but does not probe as valid video: {exc}",
            detail={"path": artifact["path"]},
        )

    project = ctx.db.get_project(project_id) or {}
    return ok({
        "project": project.get("name", project_id),
        "project_id": project_id,
        "status": "COMPLETED",
        "final_movie": artifact["path"],
        **info,
    })


@tool("assemble_movie",
      "Assemble every FINAL shot video into the finished film, verify it with "
      "ffprobe, run final QA, and mark the project COMPLETED. Call this after "
      "each shot has been finalized with finalize_shot. Fails honestly if no "
      "shot has produced a verified MP4.",
      properties={"project_id": {"type": "string"}},
      required=["project_id"])
def tool_assemble_movie(ctx: ToolContext, args: dict) -> dict:
    from ..post.ffmpeg import FFmpegError, VideoEncoder
    from ..qa import FinalQA

    project_id = require_str(args, "project_id")
    _, workspace = ctx.project_and_workspace(project_id)

    shots = ctx.db.list_shots(project_id)
    final_shot_ids: list[str] = []
    shot_videos: list[Path] = []
    frames_by_shot: dict[str, Path] = {}
    for shot in shots:
        if shot.get("status") != SHOT_FINAL:
            continue
        artifact = ctx.db.latest_artifact(project_id, "shot_video",
                                          shot["shot_id"])
        if artifact is None or not Path(artifact["path"]).is_file():
            invalidate_shot(ctx.db, project_id, shot["shot_id"],
                            "shot is marked FINAL but its video is missing",
                            events=ctx.events)
            continue
        final_shot_ids.append(shot["shot_id"])
        shot_videos.append(Path(artifact["path"]))
        frames_dir = (artifact.get("metadata") or {}).get("frames_dir")
        if frames_dir and Path(frames_dir).is_dir():
            frames_by_shot[shot["shot_id"]] = Path(frames_dir)

    if not shot_videos:
        return fail(
            "no FINAL shot videos exist yet; finalize at least one shot with "
            "finalize_shot before assembling the movie",
            detail={"final_shots": len(final_shot_ids)},
        )

    # Re-verify each shot video before it enters the cut: the registry may
    # point at a file that vanished, and a stale FINAL is not a movie.
    encoder = VideoEncoder()
    valid: list[tuple[str, Path]] = []
    for shot_id, video in zip(final_shot_ids, shot_videos):
        try:
            probe = encoder.probe(video)
            if probe.duration_s <= 0:
                invalidate_shot(ctx.db, project_id, shot_id,
                                "shot video failed ffprobe during assembly",
                                events=ctx.events)
                continue
        except FFmpegError:
            invalidate_shot(ctx.db, project_id, shot_id,
                            "shot video failed ffprobe during assembly",
                            events=ctx.events)
            continue
        valid.append((shot_id, video))
    if not valid:
        return fail(
            "every final shot failed ffprobe; the timeline is empty. "
            "Regenerate the affected shots.",
            detail={"invalidated": final_shot_ids},
        )
    final_shot_ids = [s for s, _ in valid]
    shot_videos = [v for _, v in valid]

    timeline = workspace.editorial_dir / "timeline.mp4"
    try:
        if len(shot_videos) == 1:
            import shutil
            shutil.copy2(shot_videos[0], timeline)
        else:
            encoder.concat(shot_videos, timeline)
    except FFmpegError as exc:
        return fail(f"timeline assembly failed: {exc}")
    ctx.db.register_artifact(project_id, "timeline", str(timeline),
                             "Assembled picture cut")

    final = workspace.final_video()
    expected_audio = any(bool(s.get("audio")) for s in shots)
    try:
        encoder.mux_audio(timeline, final, audio_tracks=[])
    except FFmpegError as exc:
        import shutil
        shutil.copy2(timeline, final)
    ctx.db.register_artifact(project_id, "final_movie", str(final),
                             "Assembled final movie",
                             metadata={"shots": len(shot_videos),
                                       "fps": ctx.config.render.fps})

    # Final QA over the real artifact. Rebuild ShotSpecs from the final shots
    # so the report reflects what is actually in the cut.
    final_specs: list[ShotSpec] = []
    for shot_id in final_shot_ids:
        shot = ctx.db.get_shot(project_id, shot_id)
        if not shot:
            continue
        spec_data = shot.get("spec") or {}
        spec_data.setdefault("shot_id", shot_id)
        final_specs.append(ShotSpec.from_dict(spec_data))
    qa = FinalQA(encoder, fps=ctx.config.render.fps,
                 width=ctx.config.render.final_width,
                 height=ctx.config.render.final_height)
    qa_report = qa.run(final_specs, shot_videos, final,
                       frames_by_shot=frames_by_shot,
                       expected_audio=expected_audio,
                       finalized_shots=set(final_shot_ids))

    # Publish the QA verdict and a production report as artifacts. Nothing else
    # on this path did: only the full Producer ever registered either one, so a
    # production assembled shot by shot through MCP failed its own integrity
    # gate (core.integrity.REQUIRED_KINDS demands both) however well it went.
    qa_path = workspace.qa_report()
    qa_path.write_text(json.dumps(qa_report.to_dict(), indent=2),
                       encoding="utf-8")
    ctx.db.record_qa(
        project_id, "production", project_id, "final",
        QAStatus.PASSED if qa_report.passed else QAStatus.FAILED,
        score=qa_report.score,
        findings=[c.to_dict() for c in qa_report.failures],
    )
    ctx.db.register_artifact(project_id, "qa_report", str(qa_path), "QA report")

    scored = encoder.probe(final)
    report_path = workspace.report("production_report.txt")
    report_path.write_text(json.dumps({
        "project_id": project_id,
        "shots": final_shot_ids,
        "shot_count": len(final_shot_ids),
        "final_movie": str(final),
        "size_bytes": Path(final).stat().st_size,
        "codec": scored.video_codec,
        "resolution": f"{scored.width}x{scored.height}",
        "fps": round(scored.fps, 2),
        "duration_s": round(scored.duration_s, 3),
        "qa_passed": qa_report.passed,
        "qa_score": round(qa_report.score, 3),
    }, indent=2), encoding="utf-8")
    ctx.db.register_artifact(project_id, "report", str(report_path),
                             "Production report")

    probe = encoder.probe(final)
    payload: dict[str, Any] = {
        "final": True,
        "final_movie": str(final),
        "shots": len(shot_videos),
        "shot_ids": final_shot_ids,
        "duration_s": round(probe.duration_s, 2),
        "resolution": f"{probe.width}x{probe.height}",
        "fps": round(probe.fps, 2),
        "codec": probe.video_codec,
        "size_bytes": Path(final).stat().st_size,
        "qa_status": "PASSED" if qa_report.passed else "FAILED",
        "verified": True,
    }
    ctx.emit(project_id, EventKind.EDITING_COMPLETED,
             f"Assembled {len(shot_videos)} shot(s) into the final movie",
             agent="editorial", payload={"final_movie": str(final)})
    return ok(payload)


# ===========================================================================
# AGENTS
# ===========================================================================

_AGENT_CATALOGUE = [
    ("director", "Plans the production and decides when a shot is good enough."),
    ("blender_agent", "Builds and revises scenes in Blender."),
    ("vision_agent", "Inspects rendered frames and reports defects."),
    ("final_qa", "Runs objective acceptance checks over the finished film."),
    ("editorial", "Assembles shots into a timeline and encodes the film."),
]


@tool("list_agents", "The agents that make up the production organisation.")
def tool_list_agents(ctx: ToolContext, args: dict) -> dict:
    active = ctx.runner.status().get("activity", {}).get("agent", "")
    return ok({
        "agents": [
            {"name": name, "role": role, "active": name == active}
            for name, role in _AGENT_CATALOGUE
        ],
        "count": len(_AGENT_CATALOGUE),
    })


@tool("get_agent_status",
      "What each agent is doing right now, derived from the live event stream.",
      properties={"project_id": {"type": "string"}})
def tool_get_agent_status(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    activity = ctx.events.current_activity(project_id) if project_id else {}
    tasks = ctx.db.list_tasks(project_id) if project_id else []
    by_agent: dict[str, dict[str, int]] = {}
    for task in tasks:
        bucket = by_agent.setdefault(task.agent, {})
        bucket[task.status.value] = bucket.get(task.status.value, 0) + 1
    return ok({"current_activity": activity, "by_agent": by_agent})


@tool("get_agent_activity",
      "The production event feed — what the agents have been doing, newest "
      "first.",
      properties={"project_id": {"type": "string"},
                  "limit": {"type": "integer"},
                  "agent": {"type": "string",
                            "description": "Only events from this agent."}})
def tool_get_agent_activity(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    if not project_id:
        run = ctx.runner.run
        project_id = run.project_id if run else ""
    if not project_id:
        return ok({"events": [], "count": 0})

    limit = optional_int(args, "limit", 50) or 50
    wanted = require_str(args, "agent")
    events = ctx.db.list_events(project_id, limit=limit * 3 if wanted else limit)
    if wanted:
        events = [e for e in events if e["agent"] == wanted][:limit]
    return ok({
        "events": [
            {"seq": e["seq"], "kind": e["kind"], "agent": e["agent"],
             "message": e["message"], "shot_id": e["shot_id"],
             "created_at": e["created_at"]}
            for e in events
        ],
        "count": len(events),
    })


# ===========================================================================
# QA
# ===========================================================================


@tool("run_qa",
      "Run the objective final QA checks over a project's existing artifacts: "
      "frame completeness, codecs, duration, resolution and audio.",
      properties={"project_id": {"type": "string"}})
def tool_run_qa(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    project, workspace = ctx.project_and_workspace(project_id)

    shots = ctx.db.list_shots(project_id)
    if not shots:
        return fail("this project has no shots to check")

    specs: list[ShotSpec] = []
    videos: list[Path] = []
    frames_by_shot: dict[str, Path] = {}
    for shot in shots:
        try:
            specs.append(_shot_spec(ctx.db, project_id, shot["shot_id"]))
        except MCPError:
            continue
        video = ctx.db.latest_artifact(project_id, "shot_video", shot["shot_id"])
        videos.append(Path(video["path"]) if video else Path("/nonexistent"))
        frames = ctx.db.latest_artifact(project_id, "render", shot["shot_id"])
        if frames:
            frames_by_shot[shot["shot_id"]] = Path(frames["path"])

    final = ctx.db.latest_artifact(project_id, "final_movie")
    qa = FinalQA(
        VideoEncoder(),
        fps=ctx.config.render.fps,
        width=ctx.config.render.final_width,
        height=ctx.config.render.final_height,
    )
    report = qa.run(
        specs, videos, Path(final["path"]) if final else None,
        frames_by_shot=frames_by_shot,
        expected_audio=False,
    )

    path = workspace.qa_report()
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    ctx.db.record_qa(
        project_id, "production", project_id, "final",
        QAStatus.PASSED if report.passed else QAStatus.FAILED,
        score=report.score,
        findings=[c.to_dict() for c in report.failures],
    )
    ctx.db.register_artifact(project_id, "qa_report", str(path), "QA report")
    ctx.emit(project_id,
             EventKind.QA_PASSED if report.passed else EventKind.QA_FAILED,
             f"QA {'passed' if report.passed else 'failed'} "
             f"(score {report.score:.2f})",
             agent="final_qa", payload={"report": report.to_dict()})
    return ok({"report_path": str(path), **report.to_dict()})


@tool("get_qa_status",
      "Latest QA verdict for a project, re-checked against the files on disk.",
      properties={"project_id": {"type": "string"}})
def tool_get_qa_status(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    record = ctx.db.latest_qa(project_id, project_id, "final")
    integrity = verify_project(ctx.db, project_id, encoder=VideoEncoder())

    if record is None:
        return ok({"project_id": project_id, "status": "UNREVIEWED",
                   "integrity": integrity.to_dict()})

    payload = {
        "project_id": project_id,
        "status": record["status"],
        "score": record["score"],
        "at": record["created_at"],
        "failures": record["findings"],
        "integrity": integrity.to_dict(),
    }
    # A recorded PASS describes the moment it ran. If the output is gone now,
    # saying PASSED would be repeating a claim the filesystem contradicts.
    if record["status"] == "PASSED" and not integrity.ok:
        payload["status"] = "FAILED"
        payload["recorded_status"] = "PASSED"
        payload["note"] = (
            "QA recorded a pass, but the output it checked is no longer on "
            f"disk: {integrity.summary}"
        )
    return ok(payload)


@tool("get_qa_report", "The full stored QA report for a project.",
      properties={"project_id": {"type": "string"}})
def tool_get_qa_report(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    project, workspace = ctx.project_and_workspace(project_id)
    path = workspace.qa_report()
    if not path.is_file():
        return fail("no QA report has been written yet; call run_qa")
    return ok({"path": str(path),
               "report": json.loads(path.read_text(encoding="utf-8"))})


# ===========================================================================
# CONTROL
# ===========================================================================


@tool("pause", "Pause whatever production is running.")
def tool_pause(ctx: ToolContext, args: dict) -> dict:
    return ok({"paused": ctx.runner.pause(), **ctx.runner.status()})


@tool("resume", "Resume a paused production.")
def tool_resume(ctx: ToolContext, args: dict) -> dict:
    return ok({"resumed": ctx.runner.resume(), **ctx.runner.status()})


@tool("stop", "Stop the running production.")
def tool_stop(ctx: ToolContext, args: dict) -> dict:
    return ok({"stopping": ctx.runner.stop(), **ctx.runner.status()})


__all__ = ["list_tools", "get_tool"]

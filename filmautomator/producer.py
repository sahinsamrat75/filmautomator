"""Producer — runs a full production from an owner objective to a movie.

This is the vertical slice of spec section 22, assembled:

    objective
      -> Director plans shots
      -> Blender Agent builds the scene
      -> preview render
      -> Vision Agent inspects
      -> Director corrects and repeats
      -> final render
      -> FFmpeg encodes each shot and the timeline
      -> Final QA
      -> production report

Everything before the first render is cheap; everything after is I/O bound on
Blender. Progress is reported through a callback so the owner interface can
show live status without the producer knowing what the interface is.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import AgentContext, BlenderAgent, Director, PlanResult, ShotProduction, VisionAgent
from .blender.session import BlenderSession, BlenderUnavailable
from .config import AppConfig, find_blender, load_config
from .core.events import EventBus, EventKind
from .core.integrity import verify_project
from .core.project import ProjectDB
from .core.spec import ShotSpec
from .core.task import QAStatus, Task, TaskStatus
from .core.workspace import Workspace
from .gateway import ModelGateway
from .post.ffmpeg import FFmpegError, VideoEncoder
from .qa import FinalQA, QAReport

log = logging.getLogger(__name__)

ProgressFn = Callable[..., None]


class DependencyMissing(RuntimeError):
    """A required external tool is absent, with owner-facing instructions."""

    def __init__(self, message: str, requirements: list[str]) -> None:
        super().__init__(message)
        self.requirements = requirements


@dataclass
class ProductionRequest:
    """What the owner asked for."""

    objective: str
    duration_s: float = 10.0
    style: str = ""
    title: str = ""
    extra_context: str = ""
    #: Skip the model and use the deterministic fallback plan.
    offline: bool = False
    #: A JSON file holding a shot plan to use verbatim instead of asking the
    #: Director to invent one. Lets the owner hand in their own shot list.
    plan_file: str = ""
    #: Produce into an existing project instead of creating a new one. This is
    #: what lets an MCP client create a project and then start production
    #: against it as a separate step.
    project_id: str = ""
    #: Restrict production to these shot ids. Used when re-rendering or
    #: retrying a single shot rather than the whole film.
    only_shots: list[str] = field(default_factory=list)


@dataclass
class ProductionResult:
    project_id: str
    workspace: str
    success: bool
    plan: PlanResult | None = None
    shots: list[ShotProduction] = field(default_factory=list)
    final_video: str = ""
    qa: QAReport | None = None
    #: Filesystem verification of the output, run before success was decided.
    integrity: Any = None
    report_path: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Project:   {self.project_id}",
            f"Workspace: {self.workspace}",
            f"Result:    {'SUCCESS' if self.success else 'INCOMPLETE'}",
        ]
        if self.plan:
            lines.append(f"Title:     {self.plan.title}")
            lines.append(f"Shots:     {len(self.plan.shots)} planned, "
                         f"{len(self.shots)} produced")
        approved = sum(1 for s in self.shots if s.approved)
        lines.append(f"Approved:  {approved}/{len(self.shots)} shot(s) passed review")
        if self.final_video:
            lines.append(f"Movie:     {self.final_video}")
        if self.qa:
            lines.append(f"Final QA:  {'PASSED' if self.qa.passed else 'FAILED'} "
                         f"(score {self.qa.score:.2f})")
        if self.report_path:
            lines.append(f"Report:    {self.report_path}")
        if self.error:
            lines.append(f"Error:     {self.error}")
        for note in self.notes:
            lines.append(f"Note:      {note}")
        return "\n".join(lines)


class Producer:
    """Owns the whole pipeline for one production run."""

    def __init__(self, config: AppConfig | None = None,
                 progress: ProgressFn | None = None,
                 events: EventBus | None = None) -> None:
        self.config = config or load_config()
        self.progress = progress
        self.gateway = ModelGateway(self.config.gateway)
        self.encoder = VideoEncoder()
        self.events = events if events is not None else EventBus()
        self.session: BlenderSession | None = None
        self._db: ProjectDB | None = None
        self._workspace: Workspace | None = None
        #: Optional cooperative stop/pause gate, injected by ProductionRunner.
        self.control: Any = None
        #: Called with the project id as soon as it is known. The runner uses
        #: this to bind cross-process control to the right project before any
        #: real work starts.
        self.on_project_started: Any = None
        #: The objective production is actually working to, which may come from
        #: an existing project row rather than the request.
        self._objective: str = ""

    def checkpoint(self) -> None:
        """Yield to a pending pause or stop request, if one is set."""
        if self.control is not None:
            self.control.checkpoint()

    # -- progress ----------------------------------------------------------

    def announce(self, message: str, **fields: Any) -> None:
        log.info(message)
        if self.progress is not None:
            try:
                self.progress(message, **fields)
            except Exception:  # noqa: BLE001 - UI failure must not stop production
                pass

    # -- preflight ---------------------------------------------------------

    def preflight(self, *, require_ffmpeg: bool = True) -> list[str]:
        """Check required tools, raising with actionable instructions."""
        missing: list[str] = []

        # Resolve the executable the same way BlenderSession does, rather than
        # trusting the config to already carry it. A config constructed in code
        # (rather than via load_config) legitimately has this unset, and
        # insisting "Blender is missing" while it sits in /Applications would be
        # a straightforwardly wrong thing to tell the owner.
        if self.config.blender.executable is None:
            self.config.blender.executable = find_blender()

        blender = self.config.blender.executable
        if blender is None or not Path(blender).is_file():
            missing.append(
                "DEPENDENCY REQUIRED: Blender\n"
                "WHY:               The 3D production engine. Nothing can be built or rendered without it.\n"
                "FREE/OPEN-SOURCE:  Yes — GPL, no account, no credit card.\n"
                "WHAT YOU NEED:     Run:  brew install --cask blender\n"
                "                   Or set FA_BLENDER_PATH to an existing Blender binary."
            )

        if require_ffmpeg and not self.encoder.available:
            missing.append(
                "DEPENDENCY REQUIRED: FFmpeg + FFprobe\n"
                "WHY:               Encodes shot frames into video and assembles the timeline.\n"
                "FREE/OPEN-SOURCE:  Yes — LGPL/GPL, no account, no credit card.\n"
                "WHAT YOU NEED:     Run:  brew install ffmpeg\n"
                "                   Or set FA_FFMPEG_PATH / FA_FFPROBE_PATH."
            )

        if missing:
            raise DependencyMissing(
                "Cannot start production — required tools are missing:\n\n"
                + "\n\n".join(missing),
                missing,
            )
        return missing

    def gateway_report(self) -> str:
        return self.gateway.report().render()

    # -- lifecycle ---------------------------------------------------------

    def start(self, request: ProductionRequest,
              *, require_ffmpeg: bool = True) -> tuple[str, Workspace, ProjectDB]:
        """Create the project, workspace, database and Blender session.

        ``require_ffmpeg=False`` is for planning-only runs, which never encode.
        """
        self.preflight(require_ffmpeg=require_ffmpeg)

        name = request.title or request.objective[:60] or "Untitled production"
        self._db = ProjectDB(Path(self.config.workspace) / "registry.db")

        existing = (self._db.get_project(request.project_id)
                    if request.project_id else None)
        if existing is not None:
            # Producing into a project that already exists: keep its identity
            # and its artifacts, and let a new objective update the brief.
            project_id = existing["project_id"]
            name = existing["name"]
            if request.objective:
                self._db.update_project(project_id, objective=request.objective)
            objective = request.objective or existing.get("objective", "")
        else:
            project_id = self._db.create_project(
                name=name,
                objective=request.objective,
                workspace=str(self.config.workspace),
                metadata={
                    "duration_s": request.duration_s,
                    "style": request.style,
                    "offline": request.offline,
                },
            )
            objective = request.objective

        self._objective = objective
        self._workspace = Workspace.create(
            Path(self.config.workspace), project_id, name
        )
        self.events.bind_db(self._db)

        if self.on_project_started is not None:
            try:
                self.on_project_started(project_id)
            except Exception:  # noqa: BLE001 - a bookkeeping hook, not production
                log.debug("on_project_started hook failed", exc_info=True)

        self.announce(f"project {project_id} created at {self._workspace.root}")
        self.events.emit(
            project_id,
            # Only announce creation when this run actually created it. Reusing
            # an existing project used to emit a second "created" event, which
            # made the history claim the project was made twice.
            EventKind.PROJECT_CREATED if existing is None
            else EventKind.PRODUCTION_STARTED,
            (f"Project {name!r} created" if existing is None
             else f"Producing into existing project {name!r}"),
            payload={"name": name, "objective": objective,
                     "workspace": str(self._workspace.root)},
        )

        self.session = BlenderSession(self.config.blender)
        try:
            self.session.start()
        except BlenderUnavailable as exc:
            raise DependencyMissing(
                f"Blender could not be started: {exc}",
                [str(exc)],
            ) from exc
        blender_version = self.session.ping().get("blender_version", "?")
        self.announce(f"Blender {blender_version} ready")
        self.events.emit(
            project_id, EventKind.BLENDER_STARTED,
            f"Blender {blender_version} ready",
            agent="blender_agent",
            payload={"version": blender_version,
                     "executable": str(self.config.blender.executable or "")},
        )
        return project_id, self._workspace, self._db

    def stop(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session = None
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> "Producer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- the run -----------------------------------------------------------

    def run(self, request: ProductionRequest) -> ProductionResult:
        """Produce a movie from an objective. Never raises for production
        problems — they come back as a result with ``success=False`` so the
        owner sees what happened rather than a stack trace."""
        project_id = ""
        try:
            project_id, workspace, db = self.start(request)
        except DependencyMissing as exc:
            return ProductionResult(
                project_id="", workspace="", success=False, error=str(exc)
            )

        result = ProductionResult(
            project_id=project_id,
            workspace=str(workspace.root),
            success=False,
        )
        # May differ from the request when producing into an existing project.
        objective = self._objective or request.objective

        try:
            bus = self.events
            bus.emit(project_id, EventKind.PRODUCTION_STARTED,
                     f"Production started: {objective}",
                     agent="director",
                     payload={"objective": objective,
                              "duration_s": request.duration_s})

            context = AgentContext(
                gateway=self.gateway,
                config=self.config,
                project_id=project_id,
                db=db,
                workspace=workspace,
                session=self.session,
                progress=self._agent_progress,
                events=bus,
                control=self.control,
            )
            blender_agent = BlenderAgent(context, self.session, self.config.render)
            vision_agent = VisionAgent(context)
            director = Director(
                context, blender_agent, vision_agent, self.encoder,
                max_rounds=self.config.limits.max_shot_revision_rounds,
                fps=self.config.render.fps,
                final_width=self.config.render.final_width,
                final_height=self.config.render.final_height,
            )

            # 1-2. Plan.
            plan_task = Task(
                objective=f"Plan a {request.duration_s:.0f}s sequence: {objective}",
                agent="director", project_id=project_id,
                inputs={"objective": objective,
                        "duration_s": request.duration_s},
                priority=10,
            )
            director.start_task(plan_task)

            if request.plan_file:
                plan = self._load_plan_file(request.plan_file)
                plan.notes.append(f"Plan loaded from {request.plan_file}.")
            elif request.offline:
                plan = Director._fallback_plan(objective, request.duration_s)
                plan.notes.append("Offline mode requested; deterministic plan used.")
            else:
                plan = director.plan_shots(
                    objective, request.duration_s,
                    style=request.style, extra_context=request.extra_context,
                )
            result.plan = plan
            result.notes.extend(plan.notes)
            director.complete_task(
                plan_task, shots=[s.to_dict() for s in plan.shots],
                title=plan.title, synthetic=plan.synthetic,
            )
            self.announce(
                f"plan ready: {len(plan.shots)} shot(s), "
                f"{sum(s.duration_s for s in plan.shots):.1f}s total"
            )

            # Register the plan in the project database.
            db.upsert_scene(project_id, "SC01", 0, plan.title, {"logline": plan.logline})
            for shot_spec in plan.shots:
                db.upsert_shot(
                    project_id, shot_spec.shot_id, shot_spec.scene_id,
                    shot_spec.ordinal, shot_spec.duration_s, shot_spec.to_dict(),
                )

            # 3-8. Produce each shot: build, preview, inspect, correct, final render.
            shot_videos: list[Path] = []
            frames_by_shot: dict[str, Path] = {}
            shots_to_produce = plan.shots
            if request.only_shots:
                wanted = set(request.only_shots)
                shots_to_produce = [s for s in plan.shots if s.shot_id in wanted]
                missing = wanted - {s.shot_id for s in plan.shots}
                if missing:
                    result.notes.append(
                        "requested shots not present in the plan: "
                        + ", ".join(sorted(missing))
                    )
                if not shots_to_produce:
                    result.error = (
                        "none of the requested shots exist in this project's plan"
                    )
                    result.report_path = str(self._write_report(result, project_id))
                    return result

            for shot_spec in shots_to_produce:
                self.checkpoint()
                shot_task = Task(
                    objective=f"Produce {shot_spec.shot_id}: {shot_spec.description}",
                    agent="director", project_id=project_id,
                    parent_task_id=plan_task.task_id,
                    inputs=shot_spec.to_dict(), priority=50,
                )
                director.start_task(shot_task)
                try:
                    production = director.produce_shot(shot_task, shot_spec)
                    result.shots.append(production)
                    if production.video_path:
                        shot_videos.append(Path(production.video_path))
                    if production.frames_dir:
                        frames_by_shot[shot_spec.shot_id] = Path(production.frames_dir)
                    director.complete_task(
                        shot_task, approved=production.approved,
                        version=production.version, rounds=production.rounds,
                        video=production.video_path,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad shot != dead run
                    director.fail_task(shot_task, str(exc))
                    result.notes.append(f"{shot_spec.shot_id} failed: {exc}")
                    log.error("shot %s failed:\n%s", shot_spec.shot_id,
                              traceback.format_exc())

            if not shot_videos:
                result.error = (
                    "No shot produced a video, so there is no timeline to assemble."
                )
                result.report_path = str(self._write_report(result, project_id))
                return result

            # 9. Assemble the timeline.
            self.announce(f"assembling timeline from {len(shot_videos)} shot(s)")
            bus.emit(project_id, EventKind.EDITING_STARTED,
                     f"Assembling timeline from {len(shot_videos)} shot(s)",
                     agent="editorial", payload={"shots": len(shot_videos)})
            timeline = workspace.editorial_dir / "timeline.mp4"
            try:
                if len(shot_videos) == 1:
                    # A single shot needs no concat pass; copy it into place.
                    import shutil
                    shutil.copy2(shot_videos[0], timeline)
                else:
                    self.encoder.concat(shot_videos, timeline)
            except FFmpegError as exc:
                result.error = f"Timeline assembly failed: {exc}"
                bus.emit(project_id, EventKind.PRODUCTION_FAILED, result.error,
                         agent="editorial")
                result.report_path = str(self._write_report(result, project_id))
                return result

            db.register_artifact(project_id, "timeline", str(timeline),
                                 "Assembled picture cut")
            bus.emit(project_id, EventKind.EDITING_COMPLETED,
                     "Timeline assembled", agent="editorial",
                     payload={"timeline": str(timeline)})

            # 10. Final delivery file, always under final/ and named for the
            # project so the owner never has to hunt for it.
            final = workspace.final_video()
            expected_audio = any(s.audio.dialogue or s.audio.sfx or s.audio.ambience
                                 for s in plan.shots)
            try:
                self.encoder.mux_audio(timeline, final, audio_tracks=[])
            except FFmpegError as exc:
                result.notes.append(
                    f"Audio muxing skipped ({exc}); delivering the silent cut."
                )
                import shutil
                shutil.copy2(timeline, final)

            result.final_video = str(final)
            db.register_artifact(project_id, "final_movie", str(final),
                                 f"{plan.title} — final movie",
                                 metadata={"shots": len(plan.shots),
                                           "fps": self.config.render.fps})

            # 11. Final QA.
            self.announce("running final QA")
            bus.emit(project_id, EventKind.QA_STARTED, "Running final QA",
                     agent="final_qa")
            qa = FinalQA(
                self.encoder,
                fps=self.config.render.fps,
                width=self.config.render.final_width,
                height=self.config.render.final_height,
            )
            qa_report = qa.run(
                plan.shots, shot_videos, final,
                frames_by_shot=frames_by_shot,
                expected_audio=expected_audio,
            )
            result.qa = qa_report
            db.record_qa(
                project_id, "production", project_id, "final",
                QAStatus.PASSED if qa_report.passed else QAStatus.FAILED,
                score=qa_report.score,
                findings=[c.to_dict() for c in qa_report.failures],
            )

            corrective = qa_report.corrective_tasks()
            for item in corrective:
                db.log_event(
                    project_id, "corrective_task", item["objective"],
                    item,
                )

            qa_report_path = workspace.qa_report()
            qa_report_path.write_text(
                json.dumps(qa_report.to_dict(), indent=2), encoding="utf-8"
            )
            db.register_artifact(project_id, "qa_report", str(qa_report_path),
                                 "Final QA report")
            bus.emit(
                project_id,
                EventKind.QA_PASSED if qa_report.passed else EventKind.QA_FAILED,
                (f"Final QA passed (score {qa_report.score:.2f})"
                 if qa_report.passed else
                 f"Final QA failed — {len(qa_report.failures)} check(s)"),
                agent="final_qa",
                payload={"report": qa_report.to_dict()},
            )

            # The production report is written BEFORE success is decided, and
            # is itself part of what gets verified. Deciding first and writing
            # afterwards is how a production could be called complete while its
            # own report was missing.
            result.report_path = str(self._write_report(result, project_id))

            # Verify against the filesystem rather than the database. Everything
            # above *registered* paths; this is the only step that proves the
            # files exist, are non-empty, and that the deliverable is real video.
            integrity = verify_project(db, project_id, encoder=self.encoder)
            result.integrity = integrity
            problems = [p.render() for p in integrity.problems]

            result.success = qa_report.passed and integrity.ok
            if not qa_report.passed:
                result.notes.append(
                    f"{len(qa_report.failures)} QA check(s) failed; "
                    "corrective tasks were logged."
                )
            if not integrity.ok:
                result.error = (
                    "production did not produce verifiable output: "
                    + "; ".join(problems)
                )
            result.notes.append(f"artifact integrity: {integrity.summary}")

            if result.success:
                headline = f"Production complete — {final.name}"
            elif problems:
                headline = f"Production FAILED verification: {problems[0]}"
            else:
                headline = "Production finished with QA failures"

            bus.emit(
                project_id,
                EventKind.PRODUCTION_COMPLETED if result.success
                else EventKind.PRODUCTION_FAILED,
                headline,
                agent="director",
                payload={"final_movie": str(final),
                         "success": result.success,
                         "integrity": integrity.to_dict()},
            )
            return result

        except Exception as exc:  # noqa: BLE001 - report, never crash the owner's run
            log.error("production failed:\n%s", traceback.format_exc())
            result.error = f"{type(exc).__name__}: {exc}"
            try:
                if project_id:
                    self._db.log_event(project_id, "production_failed", str(exc))
                    result.report_path = str(self._write_report(result, project_id))
            except Exception:  # noqa: BLE001
                pass
            return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _load_plan_file(path: str | Path) -> PlanResult:
        """Read a shot plan the owner supplied.

        Accepts either a bare ``{"shots": [...]}`` document or a previously
        written production report, so an earlier run's plan can be re-used.
        """
        import json

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"plan file not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))

        if "plan" in payload and isinstance(payload["plan"], dict):
            payload = payload["plan"]
        raw_shots = payload.get("shots")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise ValueError(f"plan file {source} contains no shots")

        shots = [ShotSpec.from_dict(s) for s in raw_shots]
        # Re-number so a hand-written plan cannot produce duplicate ids that
        # would collide in the database and on disk.
        for index, shot in enumerate(shots):
            shot.ordinal = index
            if not shot.shot_id:
                shot.shot_id = f"SC01_SH{index + 1:02d}"

        return PlanResult(
            title=str(payload.get("title") or "Owner-supplied plan"),
            logline=str(payload.get("logline") or ""),
            shots=shots,
        )

    def _agent_progress(self, message: str, agent: str = "") -> None:
        self.announce(message, agent=agent)

    def _write_report(self, result: ProductionResult, project_id: str) -> Path:
        """A human-readable production report (spec section 24, step 25)."""
        workspace = self._workspace
        assert workspace is not None
        lines = [
            "PRODUCTION REPORT",
            "=" * 60,
            result.summary(),
            "",
        ]
        if result.plan:
            lines += [
                "PLAN",
                "-" * 60,
                f"Title:   {result.plan.title}",
                f"Logline: {result.plan.logline}",
                "",
            ]
            for spec in result.plan.shots:
                lines.append(
                    f"  {spec.shot_id}  {spec.duration_s:5.1f}s  "
                    f"{spec.camera.shot_size:<16} {spec.camera.angle:<12} "
                    f"{spec.camera.lens_mm:.0f}mm  {spec.lighting.mood}"
                )
                if spec.description:
                    lines.append(f"      {spec.description}")
            lines.append("")

        lines += ["SHOT REVIEW HISTORY", "-" * 60]
        for shot in result.shots:
            verdict = "approved" if shot.approved else "not approved"
            lines.append(
                f"  {shot.shot_id}: {shot.rounds} round(s), {verdict}, "
                f"final version v{shot.version:03d}"
            )
            for entry in shot.history:
                lines.append(
                    f"      round {entry['round']}: score {entry['score']:.2f}, "
                    f"{len(entry['findings'])} finding(s)"
                )
                for finding in entry["findings"]:
                    lines.append(
                        f"          [{finding['severity']}/{finding['category']}] "
                        f"{finding['description']}"
                    )
                for revision in entry.get("revision") or []:
                    lines.append(f"          -> {revision}")
            for note in shot.notes:
                lines.append(f"      note: {note}")
        lines.append("")

        if result.qa:
            lines += ["FINAL QA", "-" * 60, result.qa.render(), ""]

        if result.notes:
            lines += ["NOTES", "-" * 60]
            lines += [f"  - {n}" for n in result.notes]
            lines.append("")

        lines += [
            "ARTIFACTS",
            "-" * 60,
            f"  workspace:  {result.workspace}",
            f"  movie:      {result.final_video or '(none)'}",
            f"  disk usage: {workspace.disk_usage_mb():.1f} MB",
            "",
        ]

        report_path = workspace.report("production_report.txt")
        report_path.write_text("\n".join(lines), encoding="utf-8")

        workspace.write_json("reports/production_report.json", {            "project_id": project_id,
            "success": result.success,
            "error": result.error,
            "final_video": result.final_video,
            "shots": [
                {"shot_id": s.shot_id, "approved": s.approved,
                 "rounds": s.rounds, "version": s.version,
                 "video": s.video_path, "preview": s.preview_path,
                 "blend": s.blend_path, "notes": s.notes,
                 "history": s.history}
                for s in result.shots
            ],
            "qa": result.qa.to_dict() if result.qa else None,
            "plan": result.plan.to_dict() if result.plan else None,
            "notes": result.notes,
        })
        if self._db is not None:
            try:
                self._db.register_artifact(
                    project_id, "report", str(report_path), "Production report"
                )
            except Exception:  # noqa: BLE001 - a report is not worth failing over
                pass
        return report_path

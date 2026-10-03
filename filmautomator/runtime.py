"""Production runtime: a controllable, observable production run.

An MCP tool call must not block for the several minutes a production takes, and
the dashboard needs something to pause, resume and stop. So a production runs
on a worker thread and exposes its state through this object.

Only one production runs at a time. That is a real constraint, not a shortcut:
a production owns a Blender session, and this system is built to run on one
machine. :meth:`ProductionRunner.start` refuses rather than silently
interleaving two productions onto one Blender.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .core.events import EventBus, EventKind
from .core.integrity import verify_project
from .core.project import ProjectDB
from .producer import DependencyMissing, Producer, ProductionRequest, ProductionResult

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_between(started_at: str, finished_at: str = "") -> str:
    """Human elapsed time between two ISO timestamps."""
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        end = (datetime.fromisoformat(finished_at) if finished_at
               else datetime.now(timezone.utc))
    except ValueError:
        return ""
    seconds = max(0.0, (end - start).total_seconds())
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def revoke_completed_run(db: Any, events: EventBus | None, project_id: str,
                         integrity: dict[str, Any]) -> str:
    """Downgrade a COMPLETED run whose output is gone, and persist it.

    Persisting matters: reporting FAILED from one reader while the row still
    says COMPLETED means the next reader — another process, the dashboard, a
    later session — gets the false answer all over again. Returns the message
    written, or "" if there was nothing to revoke.
    """
    try:
        stored = db.get_run(project_id)
    except Exception:  # noqa: BLE001
        return ""
    if not stored or stored.get("state") != RunState.COMPLETED.value:
        return ""

    first = integrity["problems"][0] if integrity.get("problems") else {}
    message = (
        "production previously reported COMPLETED but its output is no longer "
        f"on disk: {first.get('detail', 'missing artifacts')} "
        f"({first.get('path', '')})"
    )
    try:
        db.upsert_run(project_id, state=RunState.FAILED.value, stage="output missing",
                      error=message)
    except Exception:  # noqa: BLE001
        log.debug("could not persist revocation for %s", project_id, exc_info=True)

    if events is not None:
        try:
            events.emit(project_id, EventKind.PRODUCTION_FAILED, message,
                        agent="final_qa",
                        payload={"integrity": integrity, "revoked": True})
        except Exception:  # noqa: BLE001
            pass
    return message


def verify_run_integrity(db: Any, project_id: str, state: str,
                         encoder: Any = None) -> dict[str, Any] | None:
    """Re-check a finished run's output against the filesystem.

    Completion is decided once, when the production ends. Nothing guarantees
    the files are still there afterwards — a moved folder, a cleanup tool, or a
    mistaken delete all leave the database happily claiming success. Any read
    of run state therefore re-verifies, so ``COMPLETED`` means "the movie is on
    disk right now", not "it was, briefly, at 18:25".

    Existence only, no ffprobe: this runs on every status read and the question
    it answers is "is the output still there?". Whether the file is valid video
    was checked at completion and is checked again by ``get_final_movie`` when
    the deliverable is actually fetched.
    """
    if state != RunState.COMPLETED.value or not project_id:
        return None
    try:
        return verify_project(db, project_id, encoder=encoder,
                              probe_final=False).to_dict()
    except Exception:  # noqa: BLE001 - verification must never break status reads
        log.debug("integrity check failed for %s", project_id, exc_info=True)
        return None


class RunState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ProductionStopped(Exception):
    """Raised inside the pipeline when the owner has asked it to stop."""


class ProductionControl:
    """A cooperative stop/pause gate checked at safe points in the pipeline.

    Cooperative rather than preemptive on purpose: killing a render mid-frame
    would leave a half-written frame sequence and a Blender session in an
    unknown state. Checking between operations means stopping always leaves
    something consistent behind.

    Also polls the database for control requests. The dashboard and the MCP
    server are separate processes from whichever one is running the production,
    so a pause asked for in the browser has to travel through storage.
    """

    #: How often a paused production wakes to look for a resume request.
    POLL_INTERVAL_S = 1.5

    def __init__(self, db: Any = None, project_id: str = "") -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._stopped = threading.Event()
        self._db = db
        self._project_id = project_id

    def bind(self, db: Any, project_id: str) -> None:
        """Attach storage so cross-process control requests become visible."""
        self._db = db
        self._project_id = project_id

    def checkpoint(self) -> None:
        """Block while paused; raise if stopped."""
        self._poll_requests()
        if self._stopped.is_set():
            raise ProductionStopped("production stopped by owner")

        # A plain wait() would never wake to notice a resume requested from
        # another process, so wait in slices and re-poll each time.
        while not self._resume.wait(timeout=self.POLL_INTERVAL_S):
            self._poll_requests()
            if self._stopped.is_set():
                raise ProductionStopped("production stopped by owner")

        if self._stopped.is_set():
            raise ProductionStopped("production stopped by owner")

    def _poll_requests(self) -> None:
        """Apply any control request another process has filed."""
        if self._db is None or not self._project_id:
            return
        try:
            pending = self._db.pending_controls(self._project_id)
        except Exception:  # noqa: BLE001 - control is best-effort, never fatal
            return
        for request in pending:
            action = request.get("action", "")
            try:
                if action == "pause":
                    self.pause()
                elif action == "resume":
                    self.resume()
                elif action == "stop":
                    self.stop()
                self._db.consume_control(request["seq"])
            except Exception:  # noqa: BLE001
                pass

    @property
    def is_paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stopped.set()
        self._resume.set()  # release anything blocked in checkpoint()


@dataclass
class ProductionRun:
    """Everything known about the current (or last) production."""

    project_id: str
    objective: str
    state: RunState = RunState.IDLE
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    error: str = ""
    #: Set once the pipeline has produced a plan, so the dashboard can show
    #: "shot 2 of 3" rather than an opaque spinner.
    shots_total: int = 0
    shots_done: int = 0
    current_shot: str = ""
    current_stage: str = "starting"
    result: ProductionResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        if self.state is RunState.COMPLETED:
            return 100.0
        if self.shots_total <= 0:
            return 0.0
        return round(100.0 * self.shots_done / self.shots_total, 1)

    @property
    def is_active(self) -> bool:
        return self.state in {RunState.RUNNING, RunState.PAUSED, RunState.STOPPING}

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "objective": self.objective,
            "state": self.state.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "shots_total": self.shots_total,
            "shots_done": self.shots_done,
            "current_shot": self.current_shot,
            "current_stage": self.current_stage,
            "percent": self.percent,
            "notes": self.notes,
        }


class ProductionRunner:
    """Starts productions on a worker thread and exposes their state."""

    def __init__(self, config: AppConfig | None = None,
                 events: EventBus | None = None) -> None:
        self.config = config or load_config()
        self.events = events if events is not None else EventBus()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._control: ProductionControl | None = None
        self._run: ProductionRun | None = None
        self._producer: Producer | None = None
        self._db_handle: ProjectDB | None = None

    # -- storage -----------------------------------------------------------

    def db(self) -> ProjectDB:
        """The shared registry. Also how run state reaches other processes."""
        with self._lock:
            if self._db_handle is None:
                registry = Path(self.config.workspace) / "registry.db"
                self._db_handle = ProjectDB(registry)
                self.events.bind_db(self._db_handle)
            return self._db_handle

    def _persist(self, run: ProductionRun) -> None:
        """Publish the run's state so the dashboard and other processes see it.

        Best-effort: losing a status update is not worth failing a render over.
        """
        if not run.project_id:
            return
        try:
            self.db().upsert_run(
                run.project_id,
                state=run.state.value,
                objective=run.objective,
                stage=run.current_stage,
                current_shot=run.current_shot,
                shots_total=run.shots_total,
                shots_done=run.shots_done,
                percent=run.percent,
                error=run.error,
                pid=os.getpid(),
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
        except Exception:  # noqa: BLE001
            log.debug("could not persist run state", exc_info=True)

    def close(self) -> None:
        with self._lock:
            if self._db_handle is not None:
                try:
                    self._db_handle.close()
                except Exception:  # noqa: BLE001
                    pass
                self._db_handle = None

    # -- state -------------------------------------------------------------

    @property
    def run(self) -> ProductionRun | None:
        with self._lock:
            return self._run

    @property
    def control(self) -> ProductionControl | None:
        with self._lock:
            return self._control

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._run is not None and self._run.is_active

    def status(self, project_id: str = "") -> dict[str, Any]:
        """Run state for the live production, or for a named project.

        Without a project id this reports whatever is running now. With one it
        reports that project's recorded state — reconciled against the
        filesystem, so a project whose output has gone missing is not still
        described as COMPLETED just because it once was.
        """
        with self._lock:
            run = self._run
            if project_id and (run is None or run.project_id != project_id):
                return self._status_from_store(project_id)
            if run is None:
                return {"state": RunState.IDLE.value, "active": False}
            data = run.to_dict()
            data["active"] = run.is_active
            data["activity"] = self.events.current_activity(run.project_id)

            if run.state is RunState.COMPLETED and run.project_id:
                integrity = verify_run_integrity(
                    self.db(), run.project_id, run.state.value
                )
                if integrity is not None:
                    data["integrity"] = integrity
                    if not integrity["ok"]:
                        self._revoke(run, integrity)
                        data["state"] = RunState.FAILED.value
                        data["success_revoked"] = True
                        data["error"] = run.error
            return data

    def _status_from_store(self, project_id: str) -> dict[str, Any]:
        """Status for a project this process is not currently running."""
        try:
            stored = self.db().get_run(project_id)
        except Exception:  # noqa: BLE001
            stored = None

        if not stored:
            return {
                "project_id": project_id, "state": RunState.IDLE.value,
                "active": False,
                "note": "no production has been run for this project",
            }

        state = stored.get("state", RunState.IDLE.value)
        integrity = verify_run_integrity(self.db(), project_id, state)
        if integrity is not None and not integrity["ok"]:
            revoke_completed_run(self.db(), self.events, project_id, integrity)
            stored = self.db().get_run(project_id) or stored
            state = stored.get("state", RunState.FAILED.value)

        started = stored.get("started_at", "")
        finished = stored.get("finished_at", "")
        return {
            "project_id": project_id,
            "objective": stored.get("objective", ""),
            "state": state,
            "active": state in {RunState.RUNNING.value, RunState.PAUSED.value,
                                RunState.STOPPING.value},
            "current_stage": stored.get("stage", ""),
            "current_shot": stored.get("current_shot", ""),
            "shots_total": stored.get("shots_total", 0),
            "shots_done": stored.get("shots_done", 0),
            "percent": stored.get("percent", 0.0),
            "error": stored.get("error", ""),
            "started_at": started,
            "finished_at": finished,
            "elapsed": _elapsed_between(started, finished),
            "activity": self.events.current_activity(project_id),
            "integrity": integrity,
        }

    def _revoke(self, run: ProductionRun, integrity: dict[str, Any]) -> None:
        """Downgrade a COMPLETED run whose output no longer exists."""
        if run.state is RunState.FAILED:
            return  # already revoked; do not spam the event log
        run.state = RunState.FAILED
        run.current_stage = "output missing"
        run.error = revoke_completed_run(
            self.db(), self.events, run.project_id, integrity
        ) or "production output is no longer on disk"
        log.warning("revoking COMPLETED for %s: %s", run.project_id, run.error)
        self._persist(run)

    # -- control -----------------------------------------------------------

    def start(self, request: ProductionRequest,
              *, project_id: str | None = None) -> ProductionRun:
        """Begin a production. Returns immediately; work happens on a thread."""
        with self._lock:
            if self.is_busy:
                raise RuntimeError(
                    "a production is already running (project "
                    f"{self._run.project_id if self._run else '?'}). Stop it "
                    "before starting another — this system drives one Blender "
                    "session and runs on one machine."
                )
            control = ProductionControl(self.db(), project_id or "")
            run = ProductionRun(
                project_id=project_id or "",
                objective=request.objective,
                state=RunState.RUNNING,
                current_stage="planning",
            )
            self._control = control
            self._run = run
            self._persist(run)
            thread = threading.Thread(
                target=self._worker, args=(request, run, control),
                name="filmautomator-production", daemon=True,
            )
            self._thread = thread
        thread.start()
        return run

    def pause(self) -> bool:
        with self._lock:
            control, run = self._control, self._run
            if control is None or run is None or not run.is_active:
                return False
            control.pause()
            run.state = RunState.PAUSED
            run.current_stage = "paused"
            self._persist(run)
            self.events.emit(run.project_id, EventKind.PRODUCTION_PAUSED,
                             "Production paused", agent="director")
            return True

    def resume(self) -> bool:
        with self._lock:
            control, run = self._control, self._run
            if control is None or run is None or run.state is not RunState.PAUSED:
                return False
            control.resume()
            run.state = RunState.RUNNING
            self._persist(run)
            self.events.emit(run.project_id, EventKind.PRODUCTION_RESUMED,
                             "Production resumed", agent="director")
            return True

    def stop(self) -> bool:
        with self._lock:
            control, run = self._control, self._run
            if control is None or run is None or not run.is_active:
                return False
            run.state = RunState.STOPPING
            run.current_stage = "stopping"
            control.stop()
            self._persist(run)
            self.events.emit(run.project_id, EventKind.PRODUCTION_STOPPED,
                             "Stop requested; finishing the current operation",
                             agent="director")
            return True

    def wait(self, timeout: float | None = None) -> ProductionRun | None:
        """Block until the current production finishes. Used by tests."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.run

    # -- worker ------------------------------------------------------------

    def _worker(self, request: ProductionRequest, run: ProductionRun,
                control: ProductionControl) -> None:
        producer = Producer(
            self.config,
            progress=self._make_progress(run),
            events=self.events,
        )
        with self._lock:
            self._producer = producer
        # The pipeline checks this between operations so pause and stop take
        # effect without ever interrupting a render mid-frame.
        producer.control = control
        producer.on_project_started = (
            lambda pid: self._register_project(run, control, pid)
        )

        try:
            result = producer.run(request)
            with self._lock:
                run.result = result
                run.project_id = result.project_id or run.project_id
                run.shots_total = len(result.shots) or run.shots_total
                run.shots_done = sum(1 for s in result.shots if s.approved)
                run.notes.extend(result.notes)
                run.finished_at = _now()
                if control.is_stopped and not result.success:
                    run.state = RunState.STOPPED
                    run.current_stage = "stopped"
                elif result.success:
                    run.state = RunState.COMPLETED
                    run.current_stage = "complete"
                else:
                    run.state = RunState.FAILED
                    run.current_stage = "failed"
                    run.error = result.error
                self._persist(run)
        except ProductionStopped as exc:
            with self._lock:
                run.state = RunState.STOPPED
                run.current_stage = "stopped"
                run.error = str(exc)
                run.finished_at = _now()
                self._persist(run)
        except DependencyMissing as exc:
            with self._lock:
                run.state = RunState.FAILED
                run.current_stage = "blocked"
                run.error = str(exc)
                run.finished_at = _now()
                self._persist(run)
        except Exception as exc:  # noqa: BLE001 - a crash must not kill the server
            log.error("production worker crashed:\n%s", traceback.format_exc())
            with self._lock:
                run.state = RunState.FAILED
                run.current_stage = "failed"
                run.error = f"{type(exc).__name__}: {exc}"
                run.finished_at = _now()
                self._persist(run)
        finally:
            try:
                producer.stop()
            except Exception:  # noqa: BLE001
                pass
            with self._lock:
                self._producer = None

    def _register_project(self, run: ProductionRun, control: ProductionControl,
                          project_id: str) -> None:
        """Bind run state and control to the project the producer settled on."""
        with self._lock:
            run.project_id = project_id
            control.bind(self.db(), project_id)
            self._persist(run)

    def _make_progress(self, run: ProductionRun):
        """Translate producer announcements into run-state updates."""
        def progress(message: str, agent: str = "", **_kw: Any) -> None:
            with self._lock:
                if agent:
                    run.current_stage = message
                lowered = message.lower()
                # The producer announces shot production and shot approval;
                # those are the two moments worth reflecting in the progress
                # bar. Anything else just updates the activity line.
                if "producing " in lowered:
                    run.current_shot = message.split("producing ", 1)[1].split()[0]
                elif "passed review" in lowered or "approved" in lowered:
                    run.shots_done += 1
                self._persist(run)
        return progress

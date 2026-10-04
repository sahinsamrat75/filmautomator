"""Render job lifecycle — the MCP control plane must never block on Blender.

A final-quality Blender render legitimately takes many minutes. Before this
module existed the whole render ran inside the MCP ``tools/call`` handler, so
while one shot was rendering every other request — ``get_render`` included —
sat behind it until the client gave up. Worse, when the control socket timed
out, the session restarted Blender and re-ran the render against an empty
scene, destroying the frames the timeout had interrupted.

The model now:

    MCP request -> submit job (state QUEUED, returns job_id immediately)
                -> worker thread owns the Blender render (state RENDERING)
                -> MCP stays responsive; get_render reads the registry
                -> completion detected (state COMPLETED / FAILED)
                -> artifact verified by the job body itself
                -> caller continues the pipeline

States are exactly the production vocabulary: QUEUED, RENDERING, COMPLETED,
FAILED, CANCELLED. A queued job can be cancelled outright; a rendering job
cannot be interrupted safely (Blender mid-frame has no rollback), so cancel
reports that honestly instead of pretending.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RenderState(str, Enum):
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


#: States after which a job will never change again.
TERMINAL_STATES = frozenset({
    RenderState.COMPLETED, RenderState.FAILED, RenderState.CANCELLED,
})


@dataclass
class RenderJob:
    """One long-running render/finalize operation and its observable state."""

    job_id: str
    kind: str
    project_id: str = ""
    shot_id: str = ""
    label: str = ""
    state: RenderState = RenderState.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ""
    result: dict[str, Any] | None = None
    #: Optional live progress, e.g. {"frames_done": 12, "frames_total": 72}.
    progress: dict[str, Any] | None = None
    #: Caller-provided progress probe, invoked on each status read.
    _progress_fn: Callable[[], dict[str, Any]] | None = field(
        default=None, repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        progress = None
        if self._progress_fn is not None and not self.terminal:
            try:
                progress = self._progress_fn()
            except Exception:  # noqa: BLE001 - progress is best-effort
                progress = None
        if progress is None:
            progress = self.progress
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "project_id": self.project_id,
            "shot_id": self.shot_id,
            "label": self.label,
            "state": self.state.value,
            "terminal": self.terminal,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wait_s": round(
                ((self.finished_at or time.time()) - self.created_at), 3),
            "error": self.error,
        }
        if progress:
            payload["progress"] = progress
        if self.result is not None:
            payload["result"] = self.result
        return payload


class RenderJobManager:
    """Process-wide registry of render jobs with worker threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, RenderJob] = {}
        self._order: list[str] = []
        self._seq = 0
        #: Optional sink for lifecycle events (the MCP layer wires EventBus).
        self.on_state: Callable[[RenderJob], None] | None = None

    # -- lifecycle ----------------------------------------------------------

    def submit(self, kind: str, body: Callable[[], dict[str, Any]], *,
               project_id: str = "", shot_id: str = "", label: str = "",
               progress: Callable[[], dict[str, Any]] | None = None
               ) -> RenderJob:
        """Queue ``body`` on a worker thread; returns immediately."""
        with self._lock:
            self._seq += 1
            job = RenderJob(
                job_id=f"job_{int(time.time())}_{self._seq:04d}",
                kind=kind, project_id=project_id, shot_id=shot_id,
                label=label, _progress_fn=progress,
            )
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
        self._notify(job)
        thread = threading.Thread(
            target=self._run, args=(job, body),
            name=f"render-job-{job.job_id}", daemon=True,
        )
        thread.start()
        return job

    def _run(self, job: RenderJob, body: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            if job.state is not RenderState.QUEUED:
                # Cancelled between submit and pickup; honour it.
                return
            job.state = RenderState.RENDERING
            job.started_at = time.time()
        self._notify(job)
        try:
            result = body()
        except BaseException as exc:  # noqa: BLE001 - a job must always settle
            with self._lock:
                job.state = RenderState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()
                job.result = {"traceback": traceback.format_exc(limit=12)}
            self._notify(job)
            return
        with self._lock:
            job.state = RenderState.COMPLETED
            job.result = result if isinstance(result, dict) else {"value": result}
            job.finished_at = time.time()
        self._notify(job)

    def cancel(self, job_id: str) -> tuple[bool, str]:
        """Cancel a queued job. A rendering job cannot be interrupted safely."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, f"no such job: {job_id}"
            if job.terminal:
                return False, f"job already {job.state.value}"
            if job.state is RenderState.RENDERING:
                return False, (
                    "the render is already running inside Blender; it cannot "
                    "be interrupted without risking the scene — wait for it to "
                    "finish or fail on its own"
                )
            job.state = RenderState.CANCELLED
            job.finished_at = time.time()
        self._notify(job)
        return True, "cancelled before it started"

    # -- reads --------------------------------------------------------------

    def get(self, job_id: str) -> RenderJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self, *, project_id: str = "",
               shot_id: str = "") -> RenderJob | None:
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs[job_id]
                if project_id and job.project_id != project_id:
                    continue
                if shot_id and job.shot_id != shot_id:
                    continue
                return job
        return None

    def active(self) -> RenderJob | None:
        with self._lock:
            for job_id in self._order:
                job = self._jobs[job_id]
                if job.state in (RenderState.QUEUED, RenderState.RENDERING):
                    return job
        return None

    def list(self, *, project_id: str = "") -> list[RenderJob]:
        with self._lock:
            jobs = [self._jobs[j] for j in self._order]
        if project_id:
            jobs = [j for j in jobs if j.project_id == project_id]
        return jobs

    def _notify(self, job: RenderJob) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(job)
        except Exception:  # noqa: BLE001 - observers never break the job
            pass


_MANAGER = RenderJobManager()


def render_jobs() -> RenderJobManager:
    """The process-wide job registry (one MCP server, one production engine)."""
    return _MANAGER


def frames_progress(frames_dir: str, total: int):
    """A progress probe counting frames actually on disk while a job runs."""

    def probe() -> dict[str, Any]:
        from pathlib import Path

        directory = Path(frames_dir)
        if not directory.is_dir():
            return {"frames_done": 0, "frames_total": total}
        done = sum(
            1 for f in directory.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr"}
        )
        return {"frames_done": min(done, total), "frames_total": total}

    return probe

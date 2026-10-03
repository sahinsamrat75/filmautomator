"""Production event bus (spec section 11).

Every meaningful thing the system does is published here. Events are persisted
to SQLite so a late-joining dashboard or AI client can replay what it missed,
and fanned out to live subscribers over in-process queues so the dashboard can
update without polling the database.

The sequence number is the important detail: it is monotonic per database, so a
subscriber that disconnects and reconnects resumes from its last-seen cursor
rather than silently skipping the events that fired while it was away.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


class EventKind:
    """The event vocabulary. Strings, so they survive JSON round-trips."""

    PROJECT_CREATED = "project.created"
    PROJECT_DELETED = "project.deleted"

    PRODUCTION_STARTED = "production.started"
    PRODUCTION_PAUSED = "production.paused"
    PRODUCTION_RESUMED = "production.resumed"
    PRODUCTION_COMPLETED = "production.completed"
    PRODUCTION_FAILED = "production.failed"
    PRODUCTION_STOPPED = "production.stopped"

    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"
    TASK_BLOCKED = "task.blocked"

    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    BLENDER_STARTED = "blender.started"
    BLENDER_SCENE_LOADED = "blender.scene_loaded"
    BLENDER_SCENE_MODIFIED = "blender.scene_modified"
    BLENDER_STOPPED = "blender.stopped"

    PREVIEW_STARTED = "preview.started"
    PREVIEW_READY = "preview.ready"

    VISION_STARTED = "vision.started"
    VISION_COMPLETED = "vision.completed"

    SHOT_CREATED = "shot.created"
    SHOT_APPROVED = "shot.approved"
    SHOT_REJECTED = "shot.rejected"

    RENDER_STARTED = "render.started"
    RENDER_PROGRESS = "render.progress"
    RENDER_COMPLETED = "render.completed"

    AUDIO_STARTED = "audio.started"
    AUDIO_COMPLETED = "audio.completed"

    EDITING_STARTED = "editing.started"
    EDITING_COMPLETED = "editing.completed"

    QA_STARTED = "qa.started"
    QA_PASSED = "qa.passed"
    QA_FAILED = "qa.failed"

    DECISION_REQUESTED = "decision.requested"
    DECISION_RESOLVED = "decision.resolved"

    ARTIFACT_CREATED = "artifact.created"

    ERROR = "error"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Event:
    """One thing that happened."""

    project_id: str
    kind: str
    message: str
    agent: str = ""
    task_id: str = ""
    scene_id: str = ""
    shot_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "project_id": self.project_id,
            "kind": self.kind,
            "message": self.message,
            "agent": self.agent,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "shot_id": self.shot_id,
            "payload": self.payload,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Event":
        return cls(
            project_id=row.get("project_id", ""),
            kind=row.get("kind", ""),
            message=row.get("message", ""),
            agent=row.get("agent", "") or "",
            task_id=row.get("task_id", "") or "",
            scene_id=row.get("scene_id", "") or "",
            shot_id=row.get("shot_id", "") or "",
            payload=row.get("payload") or {},
            seq=int(row.get("seq") or 0),
            created_at=row.get("created_at", ""),
        )

    def render(self) -> str:
        """A concise line for the agent-activity feed."""
        who = f"{self.agent}: " if self.agent else ""
        return f"{who}{self.message}"


class EventBus:
    """Publishes production events to storage and to live subscribers."""

    def __init__(self, db: Any = None, *, history_limit: int = 5000) -> None:
        self._db = db
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.RLock()
        #: Most recent event, so a dashboard that just connected has something
        #: to show before any new event arrives.
        self._latest: dict[str, Event] = {}
        self._history_limit = history_limit

    # -- subscription ------------------------------------------------------

    def bind_db(self, db: Any) -> None:
        """Attach persistent storage.

        The database is created after the bus in some startup orders, so the
        binding is settable rather than constructor-only.
        """
        with self._lock:
            self._db = db

    def subscribe(self, maxsize: int = 1000) -> queue.Queue:
        """Register a live subscriber. Returns the queue it will receive on."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -- publishing --------------------------------------------------------

    def emit(self, project_id: str, kind: str, message: str, *,
             agent: str = "", task_id: str = "", scene_id: str = "",
             shot_id: str = "", payload: dict[str, Any] | None = None) -> Event:
        """Record an event and push it to every live subscriber.

        A subscriber that is too slow to drain its queue loses events rather
        than blocking production — the database still has them, and the
        subscriber can recover from its cursor.
        """
        event = Event(
            project_id=project_id, kind=kind, message=message, agent=agent,
            task_id=task_id, scene_id=scene_id, shot_id=shot_id,
            payload=dict(payload or {}),
        )

        if self._db is not None:
            try:
                event.seq = self._db.log_event(
                    project_id, kind, message, event.payload, task_id,
                    agent=agent, scene_id=scene_id, shot_id=shot_id,
                )
            except Exception:  # noqa: BLE001 - never let logging break production
                pass

        with self._lock:
            self._latest[project_id] = event
            stale: list[queue.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    stale.append(q)
            for q in stale:
                self._subscribers.remove(q)

        return event

    # -- reading -----------------------------------------------------------

    def latest(self, project_id: str) -> Event | None:
        with self._lock:
            return self._latest.get(project_id)

    def history(self, project_id: str, limit: int = 200,
                since_seq: int = 0) -> list[Event]:
        """Persisted history, oldest first."""
        if self._db is None:
            return []
        rows = self._db.events_after(project_id, since_seq, limit)
        return [Event.from_row(r) for r in rows]

    def replay(self, project_id: str, since_seq: int,
               limit: int = 1000) -> list[Event]:
        """Events after a cursor — what a reconnecting subscriber needs."""
        return self.history(project_id, limit, since_seq)

    # -- live snapshot -----------------------------------------------------

    def current_activity(self, project_id: str) -> dict[str, Any]:
        """What the production is doing right now, for the dashboard header."""
        event = self.latest(project_id)
        if event is None:
            return {"agent": "", "message": "", "kind": "", "shot_id": "",
                    "scene_id": "", "at": ""}
        return {
            "agent": event.agent,
            "message": event.message,
            "kind": event.kind,
            "shot_id": event.shot_id,
            "scene_id": event.scene_id,
            "at": event.created_at,
        }


#: Process-wide bus. Agents emit through the context, which carries its own
#: reference; this exists so tooling that has no context can still publish.
_DEFAULT_BUS: EventBus | None = None
_DEFAULT_LOCK = threading.Lock()


def default_bus(db: Any = None) -> EventBus:
    global _DEFAULT_BUS
    with _DEFAULT_LOCK:
        if _DEFAULT_BUS is None:
            _DEFAULT_BUS = EventBus(db)
        elif db is not None and _DEFAULT_BUS._db is None:  # noqa: SLF001
            _DEFAULT_BUS._db = db  # noqa: SLF001
        return _DEFAULT_BUS


def reset_default_bus() -> None:
    """Drop the process-wide bus. Used by tests to isolate state."""
    global _DEFAULT_BUS
    with _DEFAULT_LOCK:
        _DEFAULT_BUS = None

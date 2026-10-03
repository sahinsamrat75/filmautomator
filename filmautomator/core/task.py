"""Task model — the unit of work the orchestrator schedules.

Spec section 9: every task carries enough provenance that a failure can be
diagnosed and retried without human guesswork.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TaskStatus(str, Enum):
    """The ten states from spec section 9."""

    PENDING = "PENDING"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    REVIEW = "REVIEW"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    APPROVED = "APPROVED"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED}

    @property
    def is_successful(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.APPROVED}


class QAStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    NEEDS_REVISION = "NEEDS_REVISION"
    WAIVED = "WAIVED"


@dataclass
class Artifact:
    """Something a task produced that later tasks consume."""

    kind: str  # "blend" | "image" | "video" | "audio" | "json" | "text"
    path: str
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            kind=data.get("kind", "unknown"),
            path=data.get("path", ""),
            label=data.get("label", ""),
            metadata=data.get("metadata") or {},
        )


@dataclass
class Task:
    """One schedulable unit of production work."""

    objective: str
    agent: str
    project_id: str
    task_id: str = field(default_factory=lambda: new_id("task"))
    parent_task_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 100
    retry_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    artifacts: list[Artifact] = field(default_factory=list)
    qa_status: QAStatus = QAStatus.UNREVIEWED
    error: str = ""
    #: Free-form notes the Director uses to explain a decision to the owner.
    notes: list[str] = field(default_factory=list)

    # -- mutation ----------------------------------------------------------

    def touch(self, status: TaskStatus | None = None) -> None:
        if status is not None:
            self.status = status
        self.updated_at = _now()

    def add_artifact(self, kind: str, path: str, label: str = "", **metadata: Any) -> Artifact:
        artifact = Artifact(kind=kind, path=str(path), label=label, metadata=metadata)
        self.artifacts.append(artifact)
        self.touch()
        return artifact

    def note(self, message: str) -> None:
        self.notes.append(f"[{_now()}] {message}")
        self.touch()

    def fail(self, error: str) -> None:
        self.error = error
        self.touch(TaskStatus.FAILED)

    def can_retry(self, max_retries: int) -> bool:
        return self.retry_count < max_retries

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "parent_task_id": self.parent_task_id,
            "agent": self.agent,
            "objective": self.objective,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "qa_status": self.qa_status.value,
            "error": self.error,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            project_id=data["project_id"],
            parent_task_id=data.get("parent_task_id"),
            agent=data["agent"],
            objective=data["objective"],
            inputs=data.get("inputs") or {},
            outputs=data.get("outputs") or {},
            dependencies=data.get("dependencies") or [],
            status=TaskStatus(data.get("status", "PENDING")),
            priority=data.get("priority", 100),
            retry_count=data.get("retry_count", 0),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            artifacts=[Artifact.from_dict(a) for a in data.get("artifacts") or []],
            qa_status=QAStatus(data.get("qa_status", "UNREVIEWED")),
            error=data.get("error", ""),
            notes=data.get("notes") or [],
        )

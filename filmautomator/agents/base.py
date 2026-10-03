"""Agent base class.

Every specialist agent shares one context: the Model Gateway, the project
database, the workspace, and config. Agents never reach past the gateway to a
model, and never touch Blender except through the Blender Agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..blender.session import BlenderSession
from ..config import AppConfig
from ..core.events import EventBus, EventKind, default_bus
from ..core.project import ProjectDB
from ..core.task import Artifact, QAStatus, Task, TaskStatus
from ..core.workspace import Workspace
from ..gateway import Capability, ModelGateway, ModelResponse


@dataclass
class AgentContext:
    """Everything an agent is allowed to touch."""

    gateway: ModelGateway
    config: AppConfig
    project_id: str
    db: ProjectDB
    workspace: Workspace
    #: Shared Blender session. Only the Blender Agent should drive it.
    session: BlenderSession | None = None
    #: Set by the orchestrator so agents can report progress to the owner UI.
    progress: Any = None
    #: Production event bus. Every meaningful action is published here so the
    #: dashboard and any connected AI client can follow along.
    events: EventBus | None = None
    #: Cooperative stop/pause gate. Anything with a checkpoint() method works;
    #: None means "run to completion".
    control: Any = None
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("filmautomator"))

    def checkpoint(self) -> None:
        """Yield to the owner's pause/stop request, if one is pending.

        Called at points where stopping is safe — between shots and between
        revision rounds — never mid-render.
        """
        if self.control is not None:
            self.control.checkpoint()

    def bus(self) -> EventBus:
        """The event bus, creating a process-wide one if none was injected."""
        if self.events is None:
            self.events = default_bus(self.db)
        return self.events


class Agent:
    """Base class for all production agents."""

    #: Human-readable role name, used in logs and the owner interface.
    name: str = "agent"
    #: One-line description of what this agent is responsible for.
    role: str = ""

    def __init__(self, context: AgentContext) -> None:
        self.ctx = context
        self.log = context.log.getChild(self.name)

    # -- model access ------------------------------------------------------

    def think(
        self,
        capability: Capability,
        prompt: str,
        *,
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        images: list[Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Route one model call through the gateway.

        Logs which driver answered so a synthetic fallback is never mistaken
        for a real model decision in the audit trail.
        """
        response = self.ctx.gateway.invoke(
            capability,
            prompt,
            system=system,
            json_schema=json_schema,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        marker = " [SYNTHETIC]" if response.synthetic else ""
        self.log.info(
            "model call: capability=%s driver=%s model=%s%s",
            capability, response.driver, response.model or "-", marker,
        )
        return response

    # -- task bookkeeping --------------------------------------------------

    def start_task(self, task: Task) -> Task:
        task.touch(TaskStatus.RUNNING)
        self.ctx.db.save_task(task)
        self.emit(EventKind.TASK_STARTED, task.objective, task=task)
        self.announce(f"{self.name}: {task.objective}")
        return task

    def complete_task(self, task: Task, **outputs: Any) -> Task:
        task.outputs.update(outputs)
        task.touch(TaskStatus.COMPLETED)
        self.ctx.db.save_task(task)
        self.emit(EventKind.TASK_COMPLETED, task.objective, task=task)
        return task

    def fail_task(self, task: Task, error: str) -> Task:
        task.fail(error)
        self.ctx.db.save_task(task)
        self.emit(EventKind.TASK_FAILED, error, task=task, payload={"error": error})
        self.log.error("task %s failed: %s", task.task_id, error)
        return task

    def attach(self, task: Task, kind: str, path: str, label: str = "", **meta: Any) -> Artifact:
        artifact = task.add_artifact(kind, path, label, **meta)
        self.ctx.db.save_task(task)
        self.ctx.db.register_artifact(
            self.ctx.project_id, kind, path, label, metadata=meta,
        )
        self.emit(EventKind.ARTIFACT_CREATED, f"{kind}: {label or path}",
                  task=task, payload={"kind": kind, "path": str(path)})
        return artifact

    def set_qa(self, task: Task, status: QAStatus) -> None:
        task.qa_status = status
        self.ctx.db.save_task(task)

    # -- events ------------------------------------------------------------

    def emit(self, kind: str, message: str, *, task: Task | None = None,
             shot_id: str = "", scene_id: str = "",
             payload: dict[str, Any] | None = None) -> None:
        """Publish a production event."""
        try:
            self.ctx.bus().emit(
                self.ctx.project_id, kind, message,
                agent=self.name,
                task_id=task.task_id if task is not None else "",
                shot_id=shot_id, scene_id=scene_id, payload=payload,
            )
        except Exception:  # noqa: BLE001 - observability must not break production
            self.log.debug("event emit failed for %s", kind, exc_info=True)

    # -- owner-facing messaging -------------------------------------------

    def announce(self, message: str) -> None:
        """Push a status line to the owner interface, if one is attached."""
        if self.ctx.progress is not None:
            try:
                self.ctx.progress(message, agent=self.name)
            except Exception:  # noqa: BLE001 - UI trouble must not stop production
                pass

    def ask_owner(self, question: str, options: list[str] | None = None,
                  *, urgency: str = "normal",
                  context: dict[str, Any] | None = None) -> str:
        """Queue a decision for the owner (spec section 17)."""
        decision_id = self.ctx.db.request_decision(
            self.ctx.project_id, question, options,
            urgency=urgency, context=context or {},
        )
        self.announce(f"decision needed: {question}")
        return decision_id

"""Tool registry and the context tools run against.

A tool is a named, schema-described capability the external AI can invoke. The
registry is the single place that decides what exists, which is also what makes
the safety story auditable: there is no generic "run a command" or "delete a
path" tool, because none is registered.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..blender.session import BlenderSession
from ..config import AppConfig, load_config
from ..core.events import EventBus
from ..core.project import ProjectDB
from ..core.workspace import Workspace
from ..runtime import ProductionRunner
from .protocol import MCPError, INVALID_PARAMS, tool_result

log = logging.getLogger(__name__)


@dataclass
class Tool:
    """One callable capability."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[["ToolContext", dict[str, Any]], dict[str, Any]]
    #: Marks tools whose effect cannot be undone by another tool call.
    destructive: bool = False

    def descriptor(self) -> dict[str, Any]:
        """The shape ``tools/list`` returns."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, *, properties: dict[str, Any] | None = None,
         required: list[str] | None = None, destructive: bool = False):
    """Register a tool.

    ``properties`` is a JSON Schema property map; the full object schema is
    assembled here so each declaration stays readable.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required

    def decorate(fn: Callable[[ToolContext, dict[str, Any]], dict[str, Any]]):
        REGISTRY[name] = Tool(
            name=name, description=description, input_schema=schema,
            handler=fn, destructive=destructive,
        )
        return fn

    return decorate


def get_tool(name: str) -> Tool:
    found = REGISTRY.get(name)
    if found is None:
        raise MCPError(INVALID_PARAMS, f"unknown tool: {name}")
    return found


def list_tools() -> list[dict[str, Any]]:
    return [t.descriptor() for t in sorted(REGISTRY.values(), key=lambda t: t.name)]


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def require_str(args: dict[str, Any], key: str, *, default: str = "") -> str:
    value = args.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise MCPError(INVALID_PARAMS, f"{key!r} must be a string")
    return value


def optional_int(args: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = args.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MCPError(INVALID_PARAMS, f"{key!r} must be a number")
    return int(value)


def optional_float(args: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = args.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MCPError(INVALID_PARAMS, f"{key!r} must be a number")
    return float(value)


def require_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    if not isinstance(value, bool):
        raise MCPError(INVALID_PARAMS, f"{key!r} must be a boolean")
    return value


def require_project(db: ProjectDB, project_id: str) -> dict[str, Any]:
    if not project_id:
        raise MCPError(INVALID_PARAMS, "project_id is required")
    project = db.get_project(project_id)
    if project is None:
        raise MCPError(INVALID_PARAMS, f"no such project: {project_id}")
    return project


def workspace_for(config: AppConfig, project: dict[str, Any]) -> Workspace:
    """Reconstruct a project's workspace from its database row."""
    root = Path(project.get("workspace") or config.workspace) / project["project_id"]
    from ..core.workspace import slugify

    return Workspace(root=root, project_slug=slugify(project["name"]))


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Shared state every tool runs against."""

    config: AppConfig
    runner: ProductionRunner
    events: EventBus
    _db: ProjectDB | None = None
    _session: BlenderSession | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def create(cls, config: AppConfig | None = None) -> "ToolContext":
        cfg = config or load_config()
        events = EventBus()
        runner = ProductionRunner(cfg, events)
        return cls(config=cfg, runner=runner, events=events)

    # -- database ----------------------------------------------------------

    @property
    def db(self) -> ProjectDB:
        with self._lock:
            if self._db is None:
                registry = Path(self.config.workspace) / "registry.db"
                self._db = ProjectDB(registry)
                self.events.bind_db(self._db)
            return self._db

    # -- blender -----------------------------------------------------------

    def session(self) -> BlenderSession:
        """A Blender session for inspection and ad-hoc rendering.

        A running production already owns one, and starting a second Blender
        just to look at a scene would be wasteful and confusing — so the
        production's session is reused while it is alive.
        """
        with self._lock:
            producer = getattr(self.runner, "_producer", None)
            if producer is not None and producer.session is not None:
                return producer.session
            if self._session is None or not self._session.running:
                session = BlenderSession(self.config.blender)
                session.start()
                self._session = session
            return self._session

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                try:
                    self._session.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._session = None
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:  # noqa: BLE001
                    pass
                self._db = None

    # -- convenience -------------------------------------------------------

    def project_and_workspace(self, project_id: str) -> tuple[dict[str, Any], Workspace]:
        project = require_project(self.db, project_id)
        return project, workspace_for(self.config, project)

    def emit(self, project_id: str, kind: str, message: str, **kw: Any) -> None:
        try:
            self.events.emit(project_id, kind, message, **kw)
        except Exception:  # noqa: BLE001
            log.debug("event emit failed", exc_info=True)


def ok(payload: Any) -> dict[str, Any]:
    from .protocol import json_text

    return tool_result([json_text(payload)])


def fail(message: str, *, detail: Any = None) -> dict[str, Any]:
    from .protocol import json_text

    blocks = [json_text({"error": message, "detail": detail} if detail is not None
                        else {"error": message})]
    return tool_result(blocks, is_error=True)

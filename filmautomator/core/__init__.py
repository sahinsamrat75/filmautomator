"""Core domain: tasks, project memory, production specs, events, workspace."""

from .events import Event, EventBus, EventKind, default_bus, reset_default_bus
from .project import ProjectDB, ShotVersion
from .spec import (
    AudioRequirement,
    CameraSpec,
    LightingSpec,
    QualityFinding,
    QualityReport,
    SceneSpec,
    ShotSpec,
    SubjectSpec,
)
from .task import Artifact, QAStatus, Task, TaskStatus, new_id
from .workspace import ASSET_KINDS, AUDIO_KINDS, Workspace, slugify

__all__ = [
    "ASSET_KINDS",
    "AUDIO_KINDS",
    "Artifact",
    "AudioRequirement",
    "CameraSpec",
    "Event",
    "EventBus",
    "EventKind",
    "LightingSpec",
    "ProjectDB",
    "QAStatus",
    "QualityFinding",
    "QualityReport",
    "SceneSpec",
    "ShotSpec",
    "ShotVersion",
    "SubjectSpec",
    "Task",
    "TaskStatus",
    "Workspace",
    "default_bus",
    "new_id",
    "reset_default_bus",
    "slugify",
]

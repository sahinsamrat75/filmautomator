"""Core domain: tasks, project memory, production specs, workspace layout."""

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
from .workspace import Workspace

__all__ = [
    "Artifact",
    "AudioRequirement",
    "CameraSpec",
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
    "new_id",
]

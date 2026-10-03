"""Production agents.

Each agent owns one discipline. They communicate through structured specs and
tasks, never through free-form text handed to one another.
"""

from .base import Agent, AgentContext
from .blender_agent import (
    BlenderAgent,
    PreviewRender,
    ShotBuildResult,
    subject_object_names,
)
from .director import Director, PlanResult, ShotProduction
from .vision_agent import VisionAgent

__all__ = [
    "Agent",
    "AgentContext",
    "BlenderAgent",
    "Director",
    "PlanResult",
    "PreviewRender",
    "ShotBuildResult",
    "ShotProduction",
    "VisionAgent",
    "subject_object_names",
]

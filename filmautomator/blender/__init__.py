"""Blender control layer.

An agent controls Blender through :class:`BlenderSession`, which owns a
persistent background Blender process running `scripts/control_server.py`.
"""

from .session import (
    BlenderError,
    BlenderSession,
    BlenderUnavailable,
    diagnose_all,
)

__all__ = [
    "BlenderError",
    "BlenderSession",
    "BlenderUnavailable",
    "diagnose_all",
]

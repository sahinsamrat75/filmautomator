"""Shared fixtures. These tests must run with no Blender, no FFmpeg and no
model server, because that is the environment CI and a fresh clone will have.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filmautomator.agents import AgentContext, BlenderAgent, VisionAgent  # noqa: E402
from filmautomator.config import AppConfig  # noqa: E402
from filmautomator.core.project import ProjectDB  # noqa: E402
from filmautomator.core.workspace import Workspace  # noqa: E402
from filmautomator.gateway import ModelGateway  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "runs", "proj_test", "Test production")


@pytest.fixture
def db(tmp_path: Path) -> ProjectDB:
    database = ProjectDB(tmp_path / "registry.db")
    yield database
    database.close()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.workspace = tmp_path / "runs"
    # Point the gateway at a port nothing is listening on, so tests exercise
    # the "no model available" path deterministically.
    cfg.gateway.openai_compat_base_url = "http://127.0.0.1:9/v1"
    cfg.gateway.openai_compat_timeout_s = 0.5
    return cfg


@pytest.fixture
def gateway(config: AppConfig) -> ModelGateway:
    return ModelGateway(config.gateway)


@pytest.fixture
def context(config: AppConfig, db: ProjectDB, workspace: Workspace,
            gateway: ModelGateway) -> AgentContext:
    return AgentContext(
        gateway=gateway, config=config, project_id="proj_test",
        db=db, workspace=workspace, session=None,
    )


@pytest.fixture
def vision(context: AgentContext) -> VisionAgent:
    return VisionAgent(context)


@pytest.fixture
def blender_agent(context: AgentContext) -> BlenderAgent:
    """A Blender Agent whose session is never exercised.

    The correction logic is pure — it rewrites a ShotSpec and returns the list
    of changes — so it can be tested without Blender installed at all.
    """
    return BlenderAgent(context, session=None, render=context.config.render)


def metrics(**overrides) -> dict:
    """A believable 'good frame' measurement, with fields overridable."""
    base = {
        "filepath": "/tmp/frame.png",
        "width": 480, "height": 270, "aspect": 16 / 9,
        "luma_mean": 0.42, "luma_median": 0.41, "luma_std": 0.18,
        "luma_p05": 0.08, "luma_p95": 0.82,
        "clipped_shadows": 0.01, "clipped_highlights": 0.005,
        "edge_density": 0.05, "foreground_coverage": 0.13,
        "dark_row_fraction": 0.0, "dark_col_fraction": 0.0,
        "mean_rgb": [0.42, 0.41, 0.40],
        "histogram": [10] * 16,
    }
    base.update(overrides)
    return base

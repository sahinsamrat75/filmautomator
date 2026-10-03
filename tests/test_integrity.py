"""Artifact integrity: the pipeline must not claim success it cannot prove.

These exist because a production once reported COMPLETED, 100% and a QA pass
while every artifact it registered was `exists:false`. Completion was decided
from in-memory results and never checked against the filesystem again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from filmautomator.config import AppConfig, _resolve_workspace, load_config
from filmautomator.core.integrity import (
    FINAL_KIND,
    REQUIRED_KINDS,
    verify_project,
)
from filmautomator.core.project import ProjectDB
from filmautomator.post.ffmpeg import VideoEncoder


@pytest.fixture
def populated(tmp_path: Path) -> tuple[ProjectDB, str, dict[str, Path]]:
    """A project whose registered artifacts all really exist."""
    db = ProjectDB(tmp_path / "registry.db")
    project_id = db.create_project("Film", "a scene")
    paths: dict[str, Path] = {}
    for kind in REQUIRED_KINDS:
        target = tmp_path / f"{kind}.bin"
        target.write_bytes(b"x" * 256)
        db.register_artifact(project_id, kind, str(target), f"{kind} output")
        paths[kind] = target
    yield db, project_id, paths
    db.close()


# -- the verifier ----------------------------------------------------------


def test_a_complete_project_verifies(tmp_path: Path):
    db = ProjectDB(tmp_path / "registry.db")
    project_id = db.create_project("Film", "")
    for kind in REQUIRED_KINDS:
        target = tmp_path / f"{kind}.bin"
        target.write_bytes(b"x" * 256)
        db.register_artifact(project_id, kind, str(target), kind)
    report = verify_project(db, project_id, probe_final=False)
    assert report.ok is True
    assert report.problems == []
    assert report.checked == len(REQUIRED_KINDS)
    db.close()


def test_a_project_with_no_artifacts_is_not_ok(tmp_path: Path):
    db = ProjectDB(tmp_path / "registry.db")
    project_id = db.create_project("Film", "")
    report = verify_project(db, project_id, probe_final=False)
    assert report.ok is False
    missing = {p.artifact for p in report.problems if p.kind == "missing"}
    assert set(REQUIRED_KINDS) <= missing, report.summary
    db.close()


def test_a_registered_path_that_is_not_there_is_a_problem(populated, tmp_path):
    """The registry describing a file is not evidence the file exists."""
    db, project_id, paths = populated
    paths["final_movie"].unlink()

    report = verify_project(db, project_id, probe_final=False)
    assert report.ok is False
    problems = [p for p in report.problems if p.kind == "missing"]
    assert problems and problems[0].artifact == "final_movie"
    assert problems[0].path == str(paths["final_movie"])


def test_a_zero_byte_artifact_is_a_problem(populated):
    db, project_id, paths = populated
    paths["shot_video"].write_bytes(b"")

    report = verify_project(db, project_id, probe_final=False)
    assert report.ok is False
    assert any(p.kind == "empty" and p.artifact == "shot_video"
               for p in report.problems)


def test_a_final_movie_that_is_not_video_fails_the_probe(populated):
    db, project_id, paths = populated
    paths["final_movie"].write_text("this is not an mp4")

    report = verify_project(db, project_id, encoder=VideoEncoder(),
                            probe_final=True)
    assert report.ok is False
    assert any(p.kind == "probe_failed" for p in report.problems), report.summary


def test_a_missing_artifact_is_reported_per_kind_not_just_as_a_count(populated):
    db, project_id, paths = populated
    paths["preview"].unlink()
    paths["report"].unlink()

    report = verify_project(db, project_id, probe_final=False)
    kinds = {p.artifact for p in report.problems}
    assert {"preview", "report"} <= kinds


def test_verification_reports_the_deliverable_details(populated):
    db, project_id, paths = populated
    report = verify_project(db, project_id, probe_final=False)
    assert report.final_movie == str(paths["final_movie"])
    assert report.final_movie_bytes == 256
    assert FINAL_KIND in REQUIRED_KINDS


def test_unreadable_registry_is_a_problem_not_a_crash(tmp_path):
    class Broken:
        def list_artifacts(self, *a, **k):
            raise RuntimeError("database is gone")

    report = verify_project(Broken(), "proj_x", probe_final=False)
    assert report.ok is False
    assert report.problems[0].kind == "unreadable"


def test_report_serialises_for_the_api(populated):
    db, project_id, paths = populated
    payload = verify_project(db, project_id, probe_final=False).to_dict()
    # Must survive a JSON round-trip: this is what MCP clients receive.
    restored = json.loads(json.dumps(payload))
    assert restored["ok"] is True
    assert restored["checked"] == len(REQUIRED_KINDS)


# -- workspace path consistency -------------------------------------------
# The MCP server is launched by a GUI host with an arbitrary working directory.
# If the workspace stayed relative, the MCP run and a CLI run from the project
# root would resolve to different absolute paths and disagree about where a
# project's files live.


def test_relative_workspace_resolves_to_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_workspace("projects") == (tmp_path / "projects").resolve()
    assert _resolve_workspace("projects").is_absolute()


def test_workspace_resolution_is_independent_of_working_directory(tmp_path, monkeypatch):
    target = tmp_path / "ws"
    target.mkdir()
    monkeypatch.setenv("FA_WORKSPACE", str(target))

    monkeypatch.chdir(tmp_path)
    first = load_config().workspace
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    second = load_config().workspace

    assert first == second == target.resolve()
    assert first.is_absolute()


def test_env_workspace_wins_and_is_normalised(tmp_path, monkeypatch):
    monkeypatch.setenv("FA_WORKSPACE", str(tmp_path / "a" / ".." / "b"))
    assert load_config().workspace == (tmp_path / "b").resolve()


def test_default_workspace_is_absolute(monkeypatch, tmp_path):
    monkeypatch.delenv("FA_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_config().workspace.is_absolute()


def test_mcp_and_cli_share_one_workspace_resolution(tmp_path, monkeypatch):
    """Both doors must land on the same directory.

    The MCP server is built by `filmautomator mcp`, so building it the same way
    here and comparing against `load_config()` is the real check — the two
    command paths must not be able to disagree about where projects live.
    """
    from filmautomator.mcp.server import MCPServer

    monkeypatch.setenv("FA_WORKSPACE", str(tmp_path / "projects"))
    server = MCPServer()
    try:
        expected = load_config().workspace
        assert server.config.workspace == expected
        assert server.config.workspace.is_absolute()
        assert server.context.db.path == expected / "registry.db"
    finally:
        server.close()


# -- cleanup safety --------------------------------------------------------


def test_concat_cleanup_cannot_delete_the_videos_it_consumed(tmp_path, monkeypatch):
    """The only unlink in the post pipeline removes ffmpeg's own listing file.
    The shots it concatenated must be untouched."""
    from filmautomator.post.ffmpeg import VideoEncoder as Enc

    encoder = Enc(ffmpeg=Path("/bin/echo"))
    calls: list[list[str]] = []
    monkeypatch.setattr(encoder, "_run", lambda args, **kw: calls.append(args))

    shots = []
    for i in range(3):
        shot = tmp_path / f"shot{i}.mp4"
        shot.write_bytes(b"payload")
        shots.append(shot)
    output = tmp_path / "editorial" / "timeline.mp4"

    encoder.concat(shots, output)

    for shot in shots:
        assert shot.is_file(), f"concat removed its input: {shot}"
        assert shot.read_bytes() == b"payload"
    # The listing file it created is gone; it was never an artifact.
    leftovers = [p.name for p in output.parent.iterdir()
                 if p.name.startswith(".")]
    assert leftovers == [], leftovers


def test_no_module_deletes_a_project_tree():
    """Guard against a future cleanup routine quietly removing final output.

    The pipeline must never remove a registered artifact. `clear_shot_version`
    is the only tree removal in the codebase and it is scoped to a single
    version directory — this asserts nobody has started calling it from the
    production path.
    """
    import ast
    from pathlib import Path as P

    root = P(__file__).resolve().parents[1] / "filmautomator"
    callers: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "workspace.py":
            continue  # the definition itself
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"clear_shot_version", "rmtree"}):
                callers.append(f"{path.relative_to(root)}:{node.lineno}")
    assert callers == [], f"a cleanup routine was introduced: {callers}"

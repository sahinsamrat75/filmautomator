"""Production-readiness regressions: the defects found by real production.

Each test here corresponds to a defect that actually blocked a production run,
not to a hypothetical. The themes:

* **The control plane must never block on a render.** A final render used to
  run inside the MCP request handler, so ``get_render`` timed out behind it,
  and a socket timeout restarted Blender and re-ran the render against an
  empty scene — destroying the frames it interrupted.
* **Presence, not truthiness, decides whether a value was requested.**
  ``exposure_compensation=0`` means "set it to zero", not "ignore it".
* **Explicitness, not value comparison, decides camera precedence.** A shot
  asking for the default position must get that position, not a re-solve.
* **Ordinals order the film** and must be unique inside a scene.
* **Approval is about the preview that was looked at**, so it does not
  require a final render to exist first.
* **Storage accounting is project-scoped.** One project's held-back files
  never appear in another project's report.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from filmautomator.blender.session import BlenderBusy, BlenderSession
from filmautomator.core.project import ProjectDB
from filmautomator.core.spec import ShotSpec
from filmautomator.core.storage import CAT_FRAMES, StorageGovernor
from filmautomator.mcp.server import MCPServer
from filmautomator.mcp.tools import ToolContext
from filmautomator.render_jobs import RenderState, render_jobs


def _server(config) -> MCPServer:
    return MCPServer(context=ToolContext.create(config), config=config)


def _call(server: MCPServer, name: str, arguments: dict) -> dict:
    """Call a tool the way a client does, returning the decoded payload."""
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    result = (response or {}).get("result") or {}
    payload: dict = {}
    for block in result.get("content") or []:
        if block.get("type") == "text":
            try:
                payload = json.loads(block.get("text") or "{}")
            except json.JSONDecodeError:
                payload = {}
    payload["_is_error"] = bool(result.get("isError"))
    return payload


def _project(server: MCPServer, name: str = "Readiness") -> str:
    created = _call(server, "create_project", {"name": name,
                                               "objective": "readiness"})
    return created["project_id"]


# ---------------------------------------------------------------------------
# 1. Render lifecycle: the control plane never blocks on Blender
# ---------------------------------------------------------------------------


def test_submit_returns_immediately_and_the_job_is_observable():
    manager = render_jobs()
    gate = threading.Event()
    started = threading.Event()

    def body() -> dict:
        started.set()
        gate.wait(timeout=10)
        return {"value": 42}

    began = time.time()
    job = manager.submit("finalize", body, project_id="proj_x",
                         shot_id="SC01_SH001", label="test render")
    submit_s = time.time() - began

    assert submit_s < 2.0, "submit must not wait for the render"
    assert job.state in (RenderState.QUEUED, RenderState.RENDERING)
    assert job.job_id

    assert started.wait(timeout=5), "the worker should pick the job up"
    assert manager.get(job.job_id).state is RenderState.RENDERING

    gate.set()
    for _ in range(100):
        if manager.get(job.job_id).terminal:
            break
        time.sleep(0.05)
    done = manager.get(job.job_id)
    assert done.state is RenderState.COMPLETED
    assert done.result == {"value": 42}


def test_get_render_answers_from_the_registry_while_a_render_runs(config):
    """The exact defect: get_render blocked behind an in-flight render."""
    server = _server(config)
    manager = render_jobs()
    gate = threading.Event()
    project_id = _project(server)

    job = manager.submit(
        "finalize", lambda: (gate.wait(timeout=30), {"final": True})[1],
        project_id=project_id, shot_id="SC01_SH001", label="blocking render",
    )
    for _ in range(100):
        if manager.get(job.job_id).state is RenderState.RENDERING:
            break
        time.sleep(0.02)
    assert manager.get(job.job_id).state is RenderState.RENDERING

    began = time.time()
    status = _call(server, "get_render", {"project_id": project_id,
                                          "shot_id": "SC01_SH001"})
    elapsed = time.time() - began
    gate.set()

    assert elapsed < 5.0, (
        f"get_render took {elapsed:.1f}s while a render was running — the "
        "control plane blocked behind Blender"
    )
    assert not status["_is_error"]
    reported = status.get("render_job") or {}
    assert reported.get("state") == "RENDERING"
    assert reported.get("job_id") == job.job_id


def test_other_tools_answer_while_a_render_runs(config):
    """Nothing else may block on Blender either: list_projects, storage."""
    server = _server(config)
    manager = render_jobs()
    gate = threading.Event()
    project_id = _project(server)

    job = manager.submit(
        "finalize", lambda: (gate.wait(timeout=30), {"final": True})[1],
        project_id=project_id, shot_id="SC01_SH001", label="blocking render",
    )
    for _ in range(100):
        if manager.get(job.job_id).state is RenderState.RENDERING:
            break
        time.sleep(0.02)

    began = time.time()
    listing = _call(server, "list_projects", {})
    storage = _call(server, "get_storage_status", {"project_id": project_id})
    elapsed = time.time() - began
    gate.set()

    assert elapsed < 10.0, f"reads took {elapsed:.1f}s during a render"
    assert not listing["_is_error"]
    assert not storage["_is_error"]


def test_completed_job_exposes_its_result_through_get_render(config):
    server = _server(config)
    project_id = _project(server)
    job = render_jobs().submit(
        "finalize", lambda: {"final": True, "video": "/tmp/x.mp4"},
        project_id=project_id, shot_id="SC01_SH001", label="quick",
    )
    for _ in range(100):
        if job.terminal:
            break
        time.sleep(0.02)

    status = _call(server, "get_render", {"project_id": project_id,
                                          "shot_id": "SC01_SH001",
                                          "job_id": job.job_id})
    assert status.get("render_job", {}).get("state") == "COMPLETED"
    assert status.get("result", {}).get("final") is True
    assert status.get("result", {}).get("video") == "/tmp/x.mp4"


def test_cancel_refuses_a_rendering_job_and_explains_why():
    manager = render_jobs()
    gate = threading.Event()
    job = manager.submit("finalize", lambda: (gate.wait(timeout=30), {})[1],
                         label="uncancellable")
    for _ in range(100):
        if manager.get(job.job_id).state is RenderState.RENDERING:
            break
        time.sleep(0.02)

    cancelled, message = manager.cancel(job.job_id)
    gate.set()
    assert cancelled is False
    assert "cannot" in message.lower()


def test_a_failing_job_body_settles_as_FAILED_with_the_error():
    manager = render_jobs()

    def explode() -> dict:
        raise RuntimeError("encoder vanished")

    job = manager.submit("finalize", explode, label="boom")
    for _ in range(100):
        if job.terminal:
            break
        time.sleep(0.02)
    assert job.state is RenderState.FAILED
    assert "encoder vanished" in job.error
    assert job.terminal


# ---------------------------------------------------------------------------
# 2. The Blender session: timeout is not a crash, busy is not a queue
# ---------------------------------------------------------------------------


class _StubSocket:
    """Enough of a socket for the session's timeout bookkeeping."""

    def settimeout(self, value: float) -> None:  # noqa: D102 - stub
        return None


class _FakeSession(BlenderSession):
    """A session whose transport is scripted, so failure modes are testable."""

    def __init__(self, behaviour):
        super().__init__()
        self.behaviour = behaviour
        self.calls: list[str] = []
        self.restarts = 0
        # Pretend the control channel is open so call() proceeds to the
        # scripted transport instead of trying to launch a real Blender.
        self._sock = _StubSocket()
        self._stream = object()

    @property
    def running(self) -> bool:  # type: ignore[override]
        return True

    def restart(self) -> None:  # type: ignore[override]
        self.restarts += 1

    def _send_raw(self, op: str, args: dict) -> dict:  # type: ignore[override]
        self.calls.append(op)
        return self.behaviour(op, len(self.calls))


def test_a_timed_out_call_does_not_restart_blender(monkeypatch):
    """The defect that destroyed a finished render: timeout -> restart -> re-run."""
    import socket

    def behaviour(op: str, attempt: int) -> dict:
        raise socket.timeout("no answer")

    session = _FakeSession(behaviour)
    with pytest.raises(BlenderBusy):
        session.call("render_animation", directory="/tmp/frames")
    assert session.restarts == 0, (
        "a timeout must not restart Blender: the scene and any partial "
        "output belong to the interrupted operation"
    )


def test_a_genuinely_dead_transport_is_still_recovered():
    session = _FakeSession(
        lambda op, attempt: {"id": 1, "ok": True, "result": {"fine": True}}
        if attempt > 1 else (_ for _ in ()).throw(OSError("broken pipe"))
    )
    result = session.call("scene_state")
    assert result == {"fine": True}
    assert session.restarts == 1, "a dead channel should be recovered once"


def test_a_non_render_call_refuses_to_queue_behind_a_render():
    session = _FakeSession(lambda op, attempt: {"id": 1, "ok": True,
                                                "result": {}})
    # Simulate another thread owning the render.
    session._render_lock.acquire()
    session._render_active = True
    session._render_owner = -1
    session._render_label = "render_animation"
    try:
        with pytest.raises(BlenderBusy) as excinfo:
            session.call("scene_state")
        assert "rendering" in str(excinfo.value).lower()
        assert session.calls == [], "the op must not reach the transport"
    finally:
        session._render_active = False
        session._render_owner = None
        session._render_lock.release()


def test_only_one_render_can_own_the_session():
    session = _FakeSession(lambda op, attempt: {"id": 1, "ok": True,
                                                "result": {"ok": True}})
    session._render_lock.acquire()
    try:
        with pytest.raises(BlenderBusy):
            session.call("render_animation", directory="/tmp/x")
    finally:
        session._render_lock.release()


# ---------------------------------------------------------------------------
# 3. correct_shot: zero is a value, not an absence
# ---------------------------------------------------------------------------


def _shot(config, shot_id: str = "SC01_SH001") -> tuple[MCPServer, str]:
    server = _server(config)
    project_id = _project(server)
    _call(server, "create_shot", {
        "project_id": project_id, "shot_id": shot_id, "scene_id": "SC01",
        "description": "a courtyard at dusk", "duration_s": 3.0,
    })
    return server, project_id


def _lighting(server: MCPServer, project_id: str, shot_id: str) -> dict:
    shot = _call(server, "get_shot", {"project_id": project_id,
                                      "shot_id": shot_id})
    return (shot.get("spec") or {}).get("lighting") or {}


def test_exposure_compensation_zero_is_applied(config):
    """The recorded production defect: 0 was dropped by a truthiness check."""
    server, project_id = _shot(config)
    _call(server, "correct_shot", {"project_id": project_id,
                                   "shot_id": "SC01_SH001",
                                   "exposure_compensation": 1.0})
    assert _lighting(server, project_id, "SC01_SH001")["exposure_compensation"] == 1.0

    corrected = _call(server, "correct_shot", {
        "project_id": project_id, "shot_id": "SC01_SH001",
        "exposure_compensation": 0,
    })
    assert not corrected["_is_error"]
    assert corrected["applied"] == ["lighting.exposure_compensation = 0.0"]
    assert _lighting(server, project_id,
                     "SC01_SH001")["exposure_compensation"] == 0.0


def test_zero_float_and_false_are_all_literal_values(config):
    server, project_id = _shot(config)
    _call(server, "correct_shot", {"project_id": project_id,
                                   "shot_id": "SC01_SH001",
                                   "key_energy": 900.0, "fill_energy": 500.0,
                                   "rim_energy": 400.0})
    corrected = _call(server, "correct_shot", {
        "project_id": project_id, "shot_id": "SC01_SH001",
        "key_energy": 0.0,          # switch the key light off
        "fill_energy": 0,           # int zero
        "rim_energy": False,        # boolean zero
    })
    applied = " ".join(corrected["applied"])
    lighting = _lighting(server, project_id, "SC01_SH001")
    assert "key_energy = 0.0" in applied
    assert "fill_energy = 0" in applied
    assert "rim_energy = 0.0" in applied
    assert lighting["key_energy"] == 0.0
    assert lighting["fill_energy"] == 0.0
    assert lighting["rim_energy"] == 0.0


def test_an_omitted_field_is_left_alone(config):
    server, project_id = _shot(config)
    _call(server, "correct_shot", {"project_id": project_id,
                                   "shot_id": "SC01_SH001",
                                   "key_energy": 777.0})
    before = _lighting(server, project_id, "SC01_SH001")["fill_energy"]
    _call(server, "correct_shot", {"project_id": project_id,
                                   "shot_id": "SC01_SH001",
                                   "key_energy": 123.0})
    after = _lighting(server, project_id, "SC01_SH001")
    assert after["fill_energy"] == before, "an absent field must not change"
    assert after["key_energy"] == 123.0


def test_correcting_with_no_fields_is_refused(config):
    server, project_id = _shot(config)
    result = _call(server, "correct_shot", {"project_id": project_id,
                                           "shot_id": "SC01_SH001"})
    assert result["_is_error"]
    assert "no changes" in json.dumps(result).lower()


# ---------------------------------------------------------------------------
# 4. Camera: explicit means explicit, and it is verified before rendering
# ---------------------------------------------------------------------------


def _spec(camera: dict, **overrides) -> ShotSpec:
    payload = {"shot_id": "SC01_SH001", "scene_id": "SC01", "duration_s": 3.0,
               "camera": camera, "subjects": []}
    payload.update(overrides)
    return ShotSpec.from_dict(payload)


def test_a_position_equal_to_the_default_is_still_explicit():
    """The reported defect: (0, -6, 1.6) was treated as 'untouched'."""
    spec = _spec({"shot_size": "extreme_wide", "location": [0.0, -6.0, 1.6]})
    assert spec.camera.location_explicit is True


def test_an_explicit_camera_is_never_re_solved(blender_agent):
    spec = _spec({"shot_size": "extreme_wide",
                  "location": [0.0, -6.0, 1.6],
                  "look_at": [0.0, 0.0, 1.6], "lens_mm": 24.0})
    location, look_at, lens = blender_agent._solve_camera(spec)
    assert location == (0.0, -6.0, 1.6), (
        f"asked for (0, -6, 1.6) and got {location} — the value must win over "
        "the shot-size solve"
    )
    assert look_at == (0.0, 0.0, 1.6)
    assert lens == 24.0


def test_a_shot_specific_position_beats_the_preset(blender_agent):
    spec = _spec({"shot_size": "extreme_wide",
                  "location": [2.5, -14.0, 3.4], "look_at": [-0.5, 2.0, 1.6]})
    location, look_at, _ = blender_agent._solve_camera(spec)
    assert location == (2.5, -14.0, 3.4)
    assert look_at == (-0.5, 2.0, 1.6)


def test_an_unspecified_camera_is_solved_from_the_shot_size(blender_agent):
    wide = _spec({"shot_size": "extreme_wide", "lens_mm": 24.0})
    close = _spec({"shot_size": "close_up", "lens_mm": 85.0})
    wide_location, _, wide_lens = blender_agent._solve_camera(wide)
    close_location, _, close_lens = blender_agent._solve_camera(close)
    assert abs(wide_location[1]) > abs(close_location[1]), (
        "an extreme wide must stand further back than a close-up"
    )
    assert wide_lens == 24.0 and close_lens == 85.0


def test_lens_falls_back_to_the_shot_size_preset(blender_agent):
    """lens_mm=0 is 'auto': the preset applies, and nothing is silently lost."""
    spec = _spec({"shot_size": "close_up", "lens_mm": 0.0})
    assert spec.camera.lens_explicit is True
    _, _, lens = blender_agent._solve_camera(spec)
    assert lens == 85.0, "0 mm is not a lens; the preset for close_up applies"


def test_rendering_is_refused_when_the_camera_was_never_verified(blender_agent):
    spec = _spec({"location": [0.0, -6.0, 1.6]})
    blender_agent._camera_verification = {}
    with pytest.raises(RuntimeError) as excinfo:
        blender_agent._require_camera_verified(spec, "render a preview")
    assert "never verified" in str(excinfo.value)


def test_rendering_is_refused_when_the_camera_verification_failed(blender_agent):
    spec = _spec({"location": [0.0, -6.0, 1.6]})
    blender_agent._camera_verification = {
        "ok": False, "problems": ["location drifted 41.0 m"],
    }
    with pytest.raises(RuntimeError) as excinfo:
        blender_agent._require_camera_verified(spec, "render the final sequence")
    message = str(excinfo.value)
    assert "verification failed" in message
    assert "drifted" in message


def test_a_verified_camera_passes_the_gate(blender_agent):
    spec = _spec({"location": [0.0, -6.0, 1.6]})
    blender_agent._camera_verification = {"ok": True, "problems": []}
    blender_agent._require_camera_verified(spec, "render a preview")


# ---------------------------------------------------------------------------
# 5. Shot ordering: ordinals are unique and monotonic within a scene
# ---------------------------------------------------------------------------


def test_five_shots_get_unique_sequential_ordinals(config):
    server = _server(config)
    project_id = _project(server)
    for index in range(1, 6):
        created = _call(server, "create_shot", {
            "project_id": project_id, "shot_id": f"SC01_SH00{index}",
            "scene_id": "SC01", "description": f"shot {index}",
            "duration_s": 3.0,
        })
        assert not created["_is_error"], str(created)

    shots = _call(server, "list_shots", {"project_id": project_id})["shots"]
    ordinals = [row["ordinal"] for row in shots]
    assert ordinals == [0, 1, 2, 3, 4], f"ordinals were {ordinals}"
    assert len(set(ordinals)) == 5, "ordinals must be unique"


def test_re_creating_a_shot_keeps_its_ordinal(config):
    server = _server(config)
    project_id = _project(server)
    for index in (1, 2, 3):
        _call(server, "create_shot", {
            "project_id": project_id, "shot_id": f"SC01_SH00{index}",
            "scene_id": "SC01", "description": "x", "duration_s": 3.0,
        })
    _call(server, "create_shot", {
        "project_id": project_id, "shot_id": "SC01_SH001", "scene_id": "SC01",
        "description": "revised", "duration_s": 4.0,
    })
    shots = _call(server, "list_shots", {"project_id": project_id})["shots"]
    first = next(s for s in shots if s["shot_id"] == "SC01_SH001")
    assert first["ordinal"] == 0, "re-creating must not renumber the film"
    assert len(shots) == 3


def test_a_duplicate_ordinal_in_a_scene_is_rejected(config):
    server = _server(config)
    project_id = _project(server)
    _call(server, "create_shot", {"project_id": project_id,
                                 "shot_id": "SC01_SH001", "scene_id": "SC01",
                                 "description": "x", "duration_s": 3.0})
    database = server.context.db
    with pytest.raises(ValueError) as excinfo:
        database.upsert_shot(project_id, "SC01_SH009", "SC01", 0, 3.0, {})
    assert "ordinal" in str(excinfo.value).lower()
    # The original shot must survive the refused write.
    assert database.get_shot(project_id, "SC01_SH001") is not None


def test_shots_come_back_in_ordinal_order(config):
    server = _server(config)
    project_id = _project(server)
    for index in (3, 1, 5, 2):
        _call(server, "create_shot", {
            "project_id": project_id, "shot_id": f"SC01_SH00{index}",
            "scene_id": "SC01", "description": "x", "duration_s": 3.0,
        })
    shots = _call(server, "list_shots", {"project_id": project_id})["shots"]
    assert [s["ordinal"] for s in shots] == sorted(s["ordinal"] for s in shots)


# ---------------------------------------------------------------------------
# 6. Preview vs final: approval is about the preview that was looked at
# ---------------------------------------------------------------------------


def _record_preview(server: MCPServer, project_id: str, shot_id: str,
                    version: int, path: str) -> None:
    server.context.db.record_render(project_id, shot_id=shot_id,
                                    version_no=version, kind="preview",
                                    path=path, engine="BLENDER_EEVEE",
                                    metadata={"version": version})


def test_a_shot_can_be_approved_from_a_preview_alone(config):
    server, project_id = _shot(config)
    _record_preview(server, project_id, "SC01_SH001", 1, "/tmp/v1.png")

    approved = _call(server, "approve_shot", {"project_id": project_id,
                                              "shot_id": "SC01_SH001"})
    assert not approved["_is_error"], str(approved)
    assert approved["approved_version"] == 1
    assert approved["basis"] == "preview"
    assert approved["preview_versions"] == [1]

    shot = _call(server, "get_shot", {"project_id": project_id,
                                      "shot_id": "SC01_SH001"})
    assert shot["status"] == "APPROVED"
    assert shot["approved_version"] == 1


def test_the_latest_preview_is_approved_by_default(config):
    server, project_id = _shot(config)
    for version in (1, 2, 3):
        _record_preview(server, project_id, "SC01_SH001", version,
                        f"/tmp/v{version}.png")

    latest = _call(server, "approve_shot", {"project_id": project_id,
                                            "shot_id": "SC01_SH001"})
    assert latest["approved_version"] == 3
    assert latest["preview_versions"] == [1, 2, 3]

    older = _call(server, "approve_shot", {"project_id": project_id,
                                           "shot_id": "SC01_SH001",
                                           "version": 2})
    assert older["approved_version"] == 2

    missing = _call(server, "approve_shot", {"project_id": project_id,
                                             "shot_id": "SC01_SH001",
                                             "version": 9})
    assert missing["_is_error"]
    assert "no preview" in json.dumps(missing).lower()


def test_approving_a_shot_with_no_preview_is_refused(config):
    server, project_id = _shot(config)
    result = _call(server, "approve_shot", {"project_id": project_id,
                                            "shot_id": "SC01_SH001"})
    assert result["_is_error"]
    assert "preview" in json.dumps(result).lower()


def test_the_approved_preview_path_stays_traceable(config):
    server, project_id = _shot(config)
    _record_preview(server, project_id, "SC01_SH001", 2, "/tmp/frame_v2.png")
    approved = _call(server, "approve_shot", {"project_id": project_id,
                                              "shot_id": "SC01_SH001"})
    assert approved["preview_path"] == "/tmp/frame_v2.png"
    versions = server.context.db.list_shot_versions(project_id, "SC01_SH001")
    approved_rows = [v for v in versions if v.approved]
    assert approved_rows and approved_rows[0].render_path == "/tmp/frame_v2.png"


# ---------------------------------------------------------------------------
# 7. Storage is project-scoped: one project's files never appear in another's
# ---------------------------------------------------------------------------


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_project_scoped_measurement_excludes_other_projects(tmp_path):
    workspace = tmp_path / "workspace"
    # Two projects, each with an unpromoted frame sequence (retained, because
    # no verified MP4 has replaced it yet).
    _write(workspace / "proj_a" / "shots" / "SC01_SH001" / "v001" / "render"
           / "frame_0001.png", 4096)
    _write(workspace / "proj_b" / "shots" / "SC01_SH001" / "v001" / "render"
           / "frame_0001.png", 8192)

    governor = StorageGovernor(workspace)
    scoped = governor.measure(project_id="proj_a")

    assert scoped.project_bytes == 4096, "only proj_a's bytes belong here"
    assert scoped.total_bytes >= 4096 + 8192, "the ceiling is workspace-wide"

    project_section = scoped.to_dict()["project"]
    assert project_section["project_id"] == "proj_a"
    assert project_section["bytes"] == 4096
    listed = json.dumps(project_section)
    assert "proj_b" not in listed, "another project leaked into the report"

    retained_paths = [e.path for e in scoped.retained]
    assert retained_paths, "the unpromoted frames should be listed as retained"
    assert all("proj_a" in str(p) for p in retained_paths), (
        f"retained list contains another project's file: {retained_paths}"
    )


def test_a_scoped_report_keeps_the_global_totals(config):
    server = _server(config)
    project_id = _project(server, "Scoped")
    status = _call(server, "get_storage_status", {"project_id": project_id})
    assert not status["_is_error"]
    # Global view against the shared ceiling...
    assert isinstance(status["total_bytes"], int)
    assert "state" in status and "thresholds" in status
    # ...and the project's own accounting, named as such.
    assert status["project"]["project_id"] == project_id
    assert status["project"]["bytes"] <= status["total_bytes"]


def test_an_unscoped_report_has_no_project_section(config):
    server = _server(config)
    status = _call(server, "get_storage_status", {})
    assert not status["_is_error"]
    assert "project" not in status, (
        "a global report must not claim to describe one project"
    )
    assert isinstance(status["total_bytes"], int)


def test_two_projects_are_accounted_separately(tmp_path):
    workspace = tmp_path / "workspace"
    _write(workspace / "proj_a" / "previews" / "a.png", 1024)
    _write(workspace / "proj_b" / "previews" / "b.png", 2048)

    governor = StorageGovernor(workspace)
    a = governor.measure(project_id="proj_a")
    b = governor.measure(project_id="proj_b")
    both = governor.measure()

    assert a.project_bytes == 1024
    assert b.project_bytes == 2048
    assert a.project_bytes + b.project_bytes == both.total_bytes
    assert a.total_bytes == both.total_bytes

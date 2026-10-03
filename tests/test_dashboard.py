"""Dashboard HTTP surface (spec sections 10 and 12).

The dashboard's whole job is to show real state, so these tests assert that it
reads what the database actually contains — and that it refuses to invent
anything when there is nothing to show.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from filmautomator.config import AppConfig
from filmautomator.core.events import EventBus, EventKind
from filmautomator.dashboard import DashboardServer
from filmautomator.runtime import ProductionRunner


@pytest.fixture
def dashboard(tmp_path: Path):
    config = AppConfig()
    config.workspace = tmp_path / "projects"
    events = EventBus()
    runner = ProductionRunner(config, events)
    server = DashboardServer(config, runner, events)
    port = server.start("127.0.0.1", 0)
    yield server, config, events, port
    server.stop()


def get(port: int, path: str) -> tuple[int, bytes, str]:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def get_json(port: int, path: str) -> tuple[int, dict]:
    status, body, _ = get(port, path)
    return status, json.loads(body.decode("utf-8"))


def post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _register_complete_output(db, project_id: str, tmp_path: Path) -> list[str]:
    """Register a full, present set of artifacts — what a good run leaves."""
    from filmautomator.core.integrity import REQUIRED_KINDS

    paths = []
    for kind in REQUIRED_KINDS:
        target = tmp_path / f"{kind}.bin"
        target.write_bytes(b"x" * 128)
        db.register_artifact(project_id, kind, str(target), f"{kind} output")
        paths.append(str(target))
    return paths


# -- the page --------------------------------------------------------------


def test_index_serves_a_self_contained_page(dashboard):
    _, config, _, port = dashboard
    status, body, content_type = get(port, "/")
    assert status == 200
    assert "text/html" in content_type
    html = body.decode("utf-8")
    assert "Filmautomator" in html
    # Must work with no internet: no CDN links, no external fonts.
    assert "http://cdn" not in html and "https://cdn" not in html
    assert "unpkg" not in html and "jsdelivr" not in html


def test_page_has_the_controls_the_spec_requires(dashboard):
    _, _, _, port = dashboard
    html = get(port, "/")[1].decode("utf-8")
    for control in ("btn-pause", "btn-resume", "btn-stop", "btn-retry"):
        assert control in html


# -- state -----------------------------------------------------------------


def test_state_reports_honestly_when_there_are_no_projects(dashboard):
    _, _, _, port = dashboard
    status, payload = get_json(port, "/api/state")
    assert status == 200
    assert payload["project"] is None
    assert "No projects yet" in payload["note"]


def test_state_reflects_a_real_project(dashboard):
    _, config, _, port = dashboard
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("AWAKE_EP01", "a figure in the rain")

    status, payload = get_json(port, "/api/state")
    assert status == 200
    assert payload["project"]["project_id"] == project_id
    assert payload["project"]["name"] == "AWAKE_EP01"
    assert payload["progress"]["shots_total"] == 0
    assert payload["shots"] == []


def test_state_shows_shots_and_task_counts(dashboard):
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    db.upsert_shot(project_id, "SC01_SH01", "SC01", 0, 3.0, {"description": "wide"})
    from filmautomator.core.task import Task, TaskStatus

    task = Task(objective="produce", agent="director", project_id=project_id)
    task.touch(TaskStatus.RUNNING)
    db.save_task(task)

    _, payload = get_json(port=dashboard[3], path="/api/state")
    assert len(payload["shots"]) == 1
    assert payload["shots"][0]["shot_id"] == "SC01_SH01"
    assert payload["tasks"]["by_status"]["RUNNING"] == 1


def test_state_reports_no_preview_until_one_exists(dashboard):
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    db.create_project("Film", "")
    _, payload = get_json(port=dashboard[3], path="/api/state")
    assert payload["preview"] is None
    assert payload["final_movie"] is None


def test_state_exposes_the_latest_preview_artifact(dashboard, tmp_path: Path):
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    image = tmp_path / "preview.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    db.register_artifact(project_id, "preview", str(image), "SC01_SH01",
                         shot_id="SC01_SH01")

    _, payload = get_json(port=dashboard[3], path="/api/state")
    assert payload["preview"]["shot_id"] == "SC01_SH01"
    assert payload["preview"]["artifact_id"]


def test_activity_feed_is_empty_rather_than_fabricated(dashboard):
    server = dashboard[0]
    server._db_handle().create_project("Film", "")  # noqa: SLF001
    _, payload = get_json(port=dashboard[3], path="/api/state")
    # Nothing has happened yet, so the feed must be empty.
    assert payload["events"] == []
    assert payload["current_activity"]["agent"] == ""


def test_events_appear_in_the_feed_once_they_happen(dashboard):
    server, _, events, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    events.bind_db(db)
    events.emit(project_id, EventKind.PREVIEW_READY, "Preview ready for SC01_SH01",
                agent="blender_agent", shot_id="SC01_SH01")

    _, payload = get_json(port, "/api/state")
    assert payload["current_activity"]["agent"] == "blender_agent"
    kinds = [e["kind"] for e in payload["events"]]
    assert "preview.ready" in kinds


# -- artifacts -------------------------------------------------------------


def test_artifact_endpoint_serves_the_registered_file(dashboard, tmp_path: Path):
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nreal-bytes")
    artifact_id = db.register_artifact(project_id, "preview", str(image))

    status, body, content_type = get(dashboard[3], f"/artifact/{artifact_id}")
    assert status == 200
    assert body == b"\x89PNG\r\n\x1a\nreal-bytes"
    assert content_type == "image/png"


def test_artifact_endpoint_rejects_an_unknown_id(dashboard):
    status, payload = get_json(dashboard[3], "/artifact/art_nonexistent")
    assert status == 404


def test_artifact_endpoint_cannot_read_arbitrary_files(dashboard, tmp_path: Path):
    """Only registered artifacts are reachable — there is no path parameter,
    so a request cannot walk the filesystem."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")
    for attempt in ("/artifact/../../etc/passwd",
                    "/artifact/..%2f..%2fetc%2fpasswd",
                    f"/artifact/{secret}"):
        status, _ = get_json(dashboard[3], attempt)
        assert status == 404, attempt


def test_missing_artifact_file_reports_gone(dashboard):
    server = dashboard[0]
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    artifact_id = db.register_artifact(project_id, "final_movie", "/nowhere/x.mp4")
    status, payload = get_json(dashboard[3], f"/artifact/{artifact_id}")
    assert status == 410
    assert "missing" in payload["error"]


# -- controls --------------------------------------------------------------


def test_controls_report_nothing_running(dashboard):
    port = dashboard[3]
    assert post(port, "/api/control/pause", {})[1]["ok"] is False
    assert post(port, "/api/control/resume", {})[1]["ok"] is False
    assert post(port, "/api/control/stop", {"confirm": True})[1]["ok"] is False


def test_stop_requires_confirmation(dashboard):
    """Stopping abandons in-flight work, so it is gated."""
    status, payload = post(dashboard[3], "/api/control/stop", {})
    assert status == 400
    assert "confirm" in payload["error"]


def test_retry_needs_a_project(dashboard):
    status, payload = post(dashboard[3], "/api/control/retry", {})
    assert status == 400
    assert "project_id" in payload["error"]


def test_unknown_endpoint_is_404(dashboard):
    assert get_json(dashboard[3], "/api/nonsense")[0] == 404


# -- cross-process visibility ---------------------------------------------
# The dashboard and the MCP server Claude Code spawns are separate processes,
# each with its own runner. Before run state was persisted, the dashboard sat
# on IDLE with dead controls while a film was being rendered in the other
# process — showing the owner something that was not true.


def test_dashboard_sees_a_run_owned_by_another_process(dashboard):
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "a film being made")
    # Exactly what the other process writes.
    db.upsert_run(project_id, state="RUNNING", objective="a film being made",
                  stage="Rendering preview for SC01_SH01", current_shot="SC01_SH01",
                  shots_total=3, shots_done=1, percent=33.3, pid=99999)

    status, payload = get_json(port, "/api/state")
    assert status == 200
    run = payload["run"]
    assert run["state"] == "RUNNING"
    assert run["active"] is True
    assert run["current_shot"] == "SC01_SH01"
    assert run["percent"] == 33.3
    assert run["managed_elsewhere"] is True


def test_a_completed_run_with_its_output_verifies_as_complete(dashboard, tmp_path):
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    _register_complete_output(db, project_id, tmp_path)
    db.upsert_run(project_id, state="COMPLETED", percent=100.0,
                  finished_at="2026-01-01T00:00:10+00:00",
                  started_at="2026-01-01T00:00:00+00:00")

    _, payload = get_json(port, "/api/state")
    assert payload["run"]["state"] == "COMPLETED"
    assert payload["run"]["active"] is False
    assert payload["run"]["integrity"]["ok"] is True


def test_a_completed_run_whose_output_vanished_reports_failed(dashboard, tmp_path):
    """The exact bug: the database said COMPLETED, the files were gone, and the
    dashboard repeated the claim. Completion now has to survive a filesystem
    check on every read."""
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    paths = _register_complete_output(db, project_id, tmp_path)
    db.upsert_run(project_id, state="COMPLETED", percent=100.0,
                  started_at="2026-01-01T00:00:00+00:00",
                  finished_at="2026-01-01T00:00:10+00:00")

    # Everything looked fine a moment ago.
    assert get_json(port, "/api/state")[1]["run"]["state"] == "COMPLETED"

    for path in paths:
        Path(path).unlink()

    _, payload = get_json(port, "/api/state")
    assert payload["run"]["state"] == "FAILED", payload["run"]
    assert payload["run"]["integrity"]["ok"] is False
    assert "no longer on disk" in payload["run"]["error"]
    # And it must not silently keep claiming success to the next reader.
    assert db.get_run(project_id)["state"] == "FAILED"


def test_control_for_another_process_is_queued_not_dropped(dashboard):
    """The dashboard cannot pause a run it does not own, so it files a request
    the owning process picks up — and says so, rather than pretending."""
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    db.upsert_run(project_id, state="RUNNING", objective="x")

    status, payload = post(port, "/api/control/pause", {"project_id": project_id})
    assert status == 200
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert "safe point" in payload["note"]

    queued = db.pending_controls(project_id)
    assert len(queued) == 1
    assert queued[0]["action"] == "pause"


def test_control_on_a_project_that_is_not_running_does_nothing(dashboard):
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Idle", "")
    status, payload = post(port, "/api/control/pause", {"project_id": project_id})
    assert status == 200
    assert payload["ok"] is False
    assert db.pending_controls(project_id) == []


def test_control_request_is_consumed_by_the_owning_process(dashboard):
    """The handshake: the dashboard files a request, the running production
    takes it and marks it consumed.

    Uses *stop* rather than *pause* because a paused checkpoint blocks by
    design — stop is the one that unblocks the pipeline by raising.
    """
    from filmautomator.runtime import ProductionControl, ProductionStopped

    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    db.upsert_run(project_id, state="RUNNING", objective="x")
    post(port, "/api/control/stop", {"project_id": project_id, "confirm": True})
    assert len(db.pending_controls(project_id)) == 1

    control = ProductionControl(db, project_id)
    try:
        control.checkpoint()          # polls, applies the stop, consumes it
        raise AssertionError("checkpoint should have raised once stopped")
    except ProductionStopped:
        pass
    assert control.is_stopped is True
    assert db.pending_controls(project_id) == []


def test_pause_request_pauses_the_owning_process(dashboard):
    """A pause has to actually take hold, not just be recorded."""
    from filmautomator.runtime import ProductionControl

    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    db.upsert_run(project_id, state="RUNNING", objective="x")
    post(port, "/api/control/pause", {"project_id": project_id})

    control = ProductionControl(db, project_id)
    control._poll_requests()  # noqa: SLF001 - the poll half of checkpoint()
    assert control.is_paused is True
    assert db.pending_controls(project_id) == []

    # And a resume travels the same way.
    post(port, "/api/control/resume", {"project_id": project_id})
    control._poll_requests()  # noqa: SLF001
    assert control.is_paused is False


def test_stop_from_the_dashboard_still_requires_confirmation(dashboard):
    server, _, _, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Elsewhere", "")
    db.upsert_run(project_id, state="RUNNING", objective="x")

    status, payload = post(port, "/api/control/stop", {"project_id": project_id})
    assert status == 400
    assert "confirm" in payload["error"]
    assert db.pending_controls(project_id) == []


# -- SSE -------------------------------------------------------------------


def test_event_stream_sends_server_sent_events(dashboard):
    """The live feed must actually stream, so the preview updates with no
    refresh."""
    server, _, events, port = dashboard
    db = server._db_handle()  # noqa: SLF001
    project_id = db.create_project("Film", "")
    events.bind_db(db)

    received: list[str] = []
    ready = threading.Event()

    def reader() -> None:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/events")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.headers.get("Content-Type") == "text/event-stream"
                ready.set()
                buffer = b""
                while len(received) < 2:
                    chunk = response.read(1)
                    if not chunk:
                        break
                    buffer += chunk
                    if buffer.endswith(b"\n\n"):
                        received.append(buffer.decode("utf-8", "replace"))
                        buffer = b""
        except Exception:  # noqa: BLE001 - the connection is closed by the test
            pass

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    assert ready.wait(timeout=10), "the event stream never connected"

    events.emit(project_id, EventKind.PREVIEW_READY, "Preview ready",
                agent="blender_agent")
    thread.join(timeout=10)

    assert received, "no SSE frames arrived"
    frame = received[0]
    assert "event: production" in frame
    assert "data: " in frame
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["message"] == "Preview ready"


def test_dashboard_binds_to_loopback_only(dashboard):
    """It must not be reachable from another machine."""
    server = dashboard[0]
    assert server._httpd.server_address[0] == "127.0.0.1"  # noqa: SLF001


def test_dashboard_uses_a_free_port_when_asked(dashboard):
    port = dashboard[3]
    assert port > 0
    # Confirm something is genuinely listening.
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass

"""Event bus and artifact storage (spec sections 9 and 11)."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import pytest

from filmautomator.core.events import Event, EventBus, EventKind, default_bus, reset_default_bus
from filmautomator.core.project import ProjectDB
from filmautomator.core.workspace import ASSET_KINDS, AUDIO_KINDS, Workspace, slugify


# -- event bus -------------------------------------------------------------


def test_events_persist_and_keep_a_monotonic_cursor(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    first = bus.emit(project_id, EventKind.PRODUCTION_STARTED, "started")
    second = bus.emit(project_id, EventKind.TASK_STARTED, "working")
    assert second.seq > first.seq

    # The cursor is what lets a reconnecting subscriber avoid missing events.
    after = db.events_after(project_id, first.seq)
    assert len(after) == 1
    assert after[0]["message"] == "working"
    assert db.latest_event_seq(project_id) == second.seq


def test_event_carries_enough_context_to_explain_itself(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    bus.emit(project_id, EventKind.SHOT_REJECTED, "too dark",
             agent="vision_agent", task_id="task_1", shot_id="SC01_SH02",
             scene_id="SC01", payload={"score": 0.3})

    event = db.list_events(project_id)[0]
    assert event["agent"] == "vision_agent"
    assert event["shot_id"] == "SC01_SH02"
    assert event["payload"]["score"] == 0.3
    assert event["kind"] == "shot.rejected"


def test_subscribers_receive_events_live(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    q = bus.subscribe()
    try:
        bus.emit(project_id, EventKind.PREVIEW_READY, "preview ready")
        received = q.get(timeout=2)
        assert isinstance(received, Event)
        assert received.message == "preview ready"
        assert bus.subscriber_count == 1
    finally:
        bus.unsubscribe(q)
    assert bus.subscriber_count == 0


def test_multiple_subscribers_all_receive(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    queues = [bus.subscribe() for _ in range(3)]
    try:
        bus.emit(project_id, EventKind.RENDER_STARTED, "rendering")
        for q in queues:
            assert q.get(timeout=2).message == "rendering"
    finally:
        for q in queues:
            bus.unsubscribe(q)


def test_a_slow_subscriber_is_dropped_not_blocking(db: ProjectDB):
    """A stalled dashboard must never be able to stall production."""
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    q = bus.subscribe(maxsize=2)
    for i in range(10):
        bus.emit(project_id, EventKind.RENDER_PROGRESS, f"frame {i}")
    # The queue filled and the subscriber was dropped; the events are still
    # durable in the database.
    assert bus.subscriber_count == 0
    progress = [e for e in db.events_after(project_id, 0)
                if e["kind"] == EventKind.RENDER_PROGRESS]
    assert len(progress) == 10
    assert q.qsize() <= 2


def test_emitting_never_raises_even_with_a_broken_database():
    class Exploding:
        def log_event(self, *a, **k):
            raise RuntimeError("disk on fire")

    bus = EventBus(Exploding())
    event = bus.emit("p", EventKind.ERROR, "still works")
    assert event.message == "still works"  # observability must not break production


def test_current_activity_reflects_the_latest_event(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    bus.emit(project_id, EventKind.TASK_STARTED, "planning shots",
             agent="director", shot_id="SC01_SH01")
    activity = bus.current_activity(project_id)
    assert activity["agent"] == "director"
    assert activity["message"] == "planning shots"
    assert activity["shot_id"] == "SC01_SH01"


def test_history_replays_in_order(db: ProjectDB):
    bus = EventBus(db)
    project_id = db.create_project("P", "")
    for i in range(5):
        bus.emit(project_id, EventKind.TASK_STARTED, f"step {i}")
    history = bus.history(project_id, limit=10)
    steps = [e.message for e in history if e.kind == EventKind.TASK_STARTED]
    assert steps == [f"step {i}" for i in range(5)]


def test_concurrent_emits_are_serialised(db: ProjectDB):
    """Agents run on worker threads; the bus must not corrupt under them."""
    bus = EventBus(db)
    project_id = db.create_project("P", "")

    def worker(n: int) -> None:
        for i in range(20):
            bus.emit(project_id, EventKind.RENDER_PROGRESS, f"w{n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = db.events_after(project_id, 0, limit=1000)
    progress = [e for e in events if e["kind"] == EventKind.RENDER_PROGRESS]
    assert len(progress) == 80
    # Sequence numbers must be unique — that is what makes the cursor reliable.
    assert len({e["seq"] for e in events}) == len(events)


def test_default_bus_is_a_singleton():
    reset_default_bus()
    try:
        assert default_bus() is default_bus()
    finally:
        reset_default_bus()


# -- workspace layout ------------------------------------------------------


def test_workspace_creates_the_specified_tree(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "proj_1", "AWAKE_EP01")
    for directory in ("story", "assets", "scenes", "shots", "previews",
                      "renders", "audio", "editorial", "qa", "final", "logs"):
        assert (ws.root / directory).is_dir(), f"missing {directory}/"
    for kind in ASSET_KINDS:
        assert ws.asset_dir(kind).is_dir(), f"missing assets/{kind}/"
    for kind in AUDIO_KINDS:
        assert ws.audio_dir_for(kind).is_dir(), f"missing audio/{kind}/"


def test_final_movie_is_named_after_the_project(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "proj_1", "AWAKE_EP01")
    assert ws.final_video().name == "AWAKE_EP01_FINAL.mp4"
    assert ws.final_video().parent.name == "final"


def test_final_movie_name_is_filesystem_safe(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "proj_1", "AWAKE Ep 1 / draft!")
    assert ws.final_video().name == "AWAKE_Ep_1_draft_FINAL.mp4"
    assert "/" not in ws.final_video().name


def test_slugify_falls_back_rather_than_producing_an_empty_name():
    assert slugify("!!!") == "PROJECT"
    assert slugify("") == "PROJECT"
    assert slugify("Ep 01") == "Ep_01"


def test_shot_versions_live_in_their_own_directories(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "p", "Film")
    ws.prepare_shot_version("SC01_SH01", 1)
    ws.prepare_shot_version("SC01_SH01", 2)
    assert ws.shot_blend("SC01_SH01", 1).parent.name == "v001"
    assert ws.shot_blend("SC01_SH01", 2).parent.name == "v002"
    assert ws.shot_render_dir("SC01_SH01", 2).is_dir()


def test_mirror_preview_puts_a_copy_in_the_previews_directory(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "p", "Film")
    ws.prepare_shot_version("SC01_SH01", 3)
    source = ws.shot_preview("SC01_SH01", 3)
    source.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    mirrored = ws.mirror_preview("SC01_SH01", 3)
    assert mirrored is not None
    assert mirrored.parent == ws.previews_dir
    assert mirrored.name == "SC01_SH01_v003.png"
    assert mirrored.read_bytes() == source.read_bytes()


def test_mirror_preview_returns_none_when_there_is_nothing_to_copy(tmp_path: Path):
    ws = Workspace.create(tmp_path / "projects", "p", "Film")
    assert ws.mirror_preview("SC01_SH99", 1) is None


# -- artifact registry -----------------------------------------------------


def test_artifacts_are_registered_and_queryable(db: ProjectDB, tmp_path: Path):
    project_id = db.create_project("P", "")
    image = tmp_path / "preview.png"
    image.write_bytes(b"x" * 64)

    artifact_id = db.register_artifact(
        project_id, "preview", str(image), "SC01_SH01 preview",
        shot_id="SC01_SH01", metadata={"version": 1},
    )
    found = db.get_artifact(artifact_id)
    assert found["kind"] == "preview"
    assert found["bytes"] == 64
    assert found["metadata"]["version"] == 1
    assert found["exists"] is True
    assert db.latest_artifact(project_id, "preview", "SC01_SH01")["artifact_id"] == artifact_id


def test_missing_artifact_file_is_recorded_honestly(db: ProjectDB):
    """An artifact row pointing at nothing is more useful than an exception —
    but it must report that the file is gone."""
    project_id = db.create_project("P", "")
    artifact_id = db.register_artifact(project_id, "final_movie", "/nope/x.mp4")
    found = db.get_artifact(artifact_id)
    assert found["bytes"] == 0
    assert found["exists"] is False


def test_latest_artifact_picks_the_newest(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.register_artifact(project_id, "preview", "/a.png", shot_id="S1")
    second = db.register_artifact(project_id, "preview", "/b.png", shot_id="S1")
    assert db.latest_artifact(project_id, "preview", "S1")["artifact_id"] == second


def test_artifacts_are_scoped_per_project(db: ProjectDB):
    one = db.create_project("One", "")
    two = db.create_project("Two", "")
    db.register_artifact(one, "preview", "/one.png")
    db.register_artifact(two, "preview", "/two.png")
    assert len(db.list_artifacts(one)) == 1
    assert db.list_artifacts(one)[0]["path"] == "/one.png"
    assert db.list_artifacts(two)[0]["path"] == "/two.png"


def test_list_artifacts_can_filter_by_kind(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.register_artifact(project_id, "preview", "/a.png")
    db.register_artifact(project_id, "final_movie", "/f.mp4")
    assert len(db.list_artifacts(project_id, "preview")) == 1
    assert len(db.list_artifacts(project_id, "final_movie")) == 1
    assert len(db.list_artifacts(project_id)) == 2

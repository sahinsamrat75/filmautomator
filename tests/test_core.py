"""Project memory, versioning, and the specs agents exchange."""

from __future__ import annotations

from filmautomator.core.project import ProjectDB
from filmautomator.core.spec import (
    AudioRequirement,
    CameraSpec,
    LightingSpec,
    QualityReport,
    SceneSpec,
    ShotSpec,
    SubjectSpec,
)
from filmautomator.core.task import Artifact, QAStatus, Task, TaskStatus


# -- tasks -----------------------------------------------------------------


def test_task_roundtrip_preserves_everything():
    task = Task(objective="build the shot", agent="blender_agent",
                project_id="p1", inputs={"a": 1})
    task.add_artifact("blend", "/tmp/x.blend", "scene")
    task.note("started")
    restored = Task.from_dict(task.to_dict())
    assert restored.objective == task.objective
    assert restored.inputs == {"a": 1}
    assert len(restored.artifacts) == 1
    assert restored.artifacts[0].kind == "blend"
    assert restored.notes == task.notes


def test_task_status_terminality():
    assert TaskStatus.COMPLETED.is_terminal
    assert TaskStatus.FAILED.is_terminal
    assert not TaskStatus.RUNNING.is_terminal
    assert TaskStatus.APPROVED.is_successful


def test_task_retry_budget():
    task = Task(objective="x", agent="a", project_id="p")
    assert task.can_retry(3)
    task.retry_count = 3
    assert not task.can_retry(3)


# -- spec roundtrip --------------------------------------------------------


def test_shotspec_roundtrip_preserves_nested_specs():
    spec = ShotSpec(
        shot_id="SC01_SH01", scene_id="SC01", ordinal=0, duration_s=6.2,
        description="Akira slowly raises his head.",
        camera=CameraSpec(shot_size="medium_close", angle="low", lens_mm=65.0,
                          location=(0.0, -3.0, 1.2), look_at=(0.0, 0.0, 1.5)),
        lighting=LightingSpec(mood="moonlight", key_energy=520.0),
        subjects=[SubjectSpec(name="Akira", height_m=1.75,
                              location=(0.0, 0.0, 0.0), action="raises head")],
        environment="rain-soaked street",
        audio=AudioRequirement(ambience="rain", sfx=["distant thunder"]),
        quality_criteria=["Akira's face is readable."],
    )
    restored = ShotSpec.from_dict(spec.to_dict())
    assert restored.shot_id == spec.shot_id
    assert restored.duration_s == spec.duration_s
    assert restored.camera.shot_size == "medium_close"
    assert restored.camera.lens_mm == 65.0
    # JSON has no tuples; vectors must come back as tuples for mathutils.
    assert restored.camera.location == (0.0, -3.0, 1.2)
    assert isinstance(restored.lighting.key_color, tuple)
    assert restored.subjects[0].name == "Akira"
    assert restored.audio.sfx == ["distant thunder"]


def test_shotspec_survives_a_json_roundtrip():
    import json

    spec = ShotSpec(shot_id="S1", subjects=[SubjectSpec(name="A")])
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = ShotSpec.from_dict(payload)
    assert restored.subjects[0].location == (0.0, 0.0, 0.0)


def test_frame_count_from_duration():
    spec = ShotSpec(shot_id="S1", duration_s=2.0)
    assert spec.frame_count_at(24) == 48
    assert spec.frame_count_at(30) == 60
    # Never zero, even for a very short shot.
    assert ShotSpec(shot_id="S2", duration_s=0.01).frame_count_at(24) == 1


def test_scene_roundtrip():
    scene = SceneSpec(scene_id="SC01", title="Rooftop", shots=[
        ShotSpec(shot_id="SC01_SH01"), ShotSpec(shot_id="SC01_SH02"),
    ])
    restored = SceneSpec.from_dict(scene.to_dict())
    assert [s.shot_id for s in restored.shots] == ["SC01_SH01", "SC01_SH02"]


def test_quality_report_blocking_findings():
    from filmautomator.core.spec import QualityFinding

    report = QualityReport(
        passed=False,
        findings=[
            QualityFinding(category="lighting", severity="minor", description="a"),
            QualityFinding(category="framing", severity="major", description="b"),
            QualityFinding(category="technical", severity="critical", description="c"),
        ],
    )
    assert len(report.blocking_findings) == 2
    assert QualityReport.from_dict(report.to_dict()).passed is False


# -- database --------------------------------------------------------------


def test_project_lifecycle(db: ProjectDB):
    project_id = db.create_project("Awake Ep1", "adapt the story bible")
    project = db.get_project(project_id)
    assert project is not None
    assert project["name"] == "Awake Ep1"
    db.update_project(project_id, status="PAUSED")
    assert db.get_project(project_id)["status"] == "PAUSED"


def test_task_persistence_and_status_filter(db: ProjectDB):
    project_id = db.create_project("P", "")
    task = Task(objective="plan", agent="director", project_id=project_id)
    db.save_task(task)
    task.touch(TaskStatus.RUNNING)
    db.save_task(task)

    assert len(db.list_tasks(project_id, TaskStatus.RUNNING)) == 1
    assert len(db.list_tasks(project_id, TaskStatus.COMPLETED)) == 0
    restored = db.get_task(task.task_id)
    assert restored is not None and restored.status is TaskStatus.RUNNING


def test_shot_versions_are_additive_and_approvable(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.upsert_shot(project_id, "SC01_SH01", "SC01", 0, 6.0, {"a": 1})

    assert db.next_version_no(project_id, "SC01_SH01") == 1
    first = db.add_shot_version(project_id, "SC01_SH01", blend_path="v1.blend",
                                qa={"score": 0.4})
    second = db.add_shot_version(project_id, "SC01_SH01", blend_path="v2.blend",
                                 qa={"score": 0.9})
    assert first.version_no == 1 and second.version_no == 2

    db.approve_shot_version(project_id, "SC01_SH01", 2)
    versions = db.list_shot_versions(project_id, "SC01_SH01")
    # Approving v2 must not remove v1 — spec section 16.
    assert len(versions) == 2
    assert [v.version_no for v in versions] == [1, 2]
    assert [v.approved for v in versions] == [False, True]

    latest = db.latest_approved_version(project_id, "SC01_SH01")
    assert latest is not None and latest.version_no == 2
    assert db.get_shot(project_id, "SC01_SH01")["approved_version"] == 2


def test_reverting_to_an_earlier_approved_version(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.upsert_shot(project_id, "S1")
    db.add_shot_version(project_id, "S1", blend_path="v1.blend")
    db.add_shot_version(project_id, "S1", blend_path="v2.blend")
    db.approve_shot_version(project_id, "S1", 1)
    db.approve_shot_version(project_id, "S1", 2)
    # Approve an older version again; the newest approved is still what wins.
    assert db.latest_approved_version(project_id, "S1").version_no == 2


# -- project isolation -----------------------------------------------------
# Every production numbers its shots SC01_SH01 upward. When shots were keyed by
# shot_id alone, a second film silently overwrote the first film's rows: the
# newest project ended up with zero shots and older projects pointed at another
# film's data. These two tests exist because that actually happened.


def test_two_projects_can_use_the_same_shot_ids(db: ProjectDB):
    first = db.create_project("Film One", "")
    second = db.create_project("Film Two", "")

    db.upsert_shot(first, "SC01_SH01", "SC01", 0, 3.0, {"who": "one"})
    db.upsert_shot(second, "SC01_SH01", "SC01", 0, 5.0, {"who": "two"})

    one = db.get_shot(first, "SC01_SH01")
    two = db.get_shot(second, "SC01_SH01")
    assert one["spec"]["who"] == "one" and one["duration_s"] == 3.0
    assert two["spec"]["who"] == "two" and two["duration_s"] == 5.0

    assert len(db.list_shots(first)) == 1
    assert len(db.list_shots(second)) == 1


def test_approving_a_shot_in_one_project_does_not_touch_the_other(db: ProjectDB):
    first = db.create_project("Film One", "")
    second = db.create_project("Film Two", "")
    for project in (first, second):
        db.upsert_shot(project, "SC01_SH01", "SC01", 0, 3.0)
        db.add_shot_version(project, "SC01_SH01", blend_path="v1.blend")

    db.approve_shot_version(first, "SC01_SH01", 1)

    assert db.get_shot(first, "SC01_SH01")["status"] == "APPROVED"
    assert db.get_shot(second, "SC01_SH01")["status"] == "PENDING"
    # Version numbering is per project too, so the second film still starts at 1.
    assert db.next_version_no(second, "SC01_SH01") == 2
    assert db.latest_approved_version(second, "SC01_SH01") is None


def test_qa_records_are_scoped_to_a_project(db: ProjectDB):
    first = db.create_project("Film One", "")
    second = db.create_project("Film Two", "")
    db.record_qa(first, "shot", "SC01_SH01", "preview", QAStatus.PASSED, score=0.9)
    db.record_qa(second, "shot", "SC01_SH01", "preview",
                 QAStatus.NEEDS_REVISION, score=0.2)

    assert db.latest_qa(first, "SC01_SH01", "preview")["score"] == 0.9
    assert db.latest_qa(second, "SC01_SH01", "preview")["score"] == 0.2


def test_final_artifacts_land_on_the_round_they_belong_to(db: ProjectDB):
    """A round's version row is created at review time and completed at encode
    time — it must not spawn a second row, or the approved version ends up
    pointing at the preview while the finished video sits under a version
    number with no directory behind it."""
    project_id = db.create_project("P", "")
    db.upsert_shot(project_id, "S1")

    # Round 1: preview reviewed.
    db.add_shot_version(project_id, "S1", blend_path="v001/S1.blend",
                        render_path="v001/preview.png", qa={"score": 0.5})
    # Same round, later: final render and encode.
    db.update_shot_version(project_id, "S1", 1,
                           render_path="v001/render",
                           video_path="v001/S1.mp4",
                           qa={"score": 0.9}, notes="final")

    versions = db.list_shot_versions(project_id, "S1")
    assert len(versions) == 1
    assert versions[0].video_path == "v001/S1.mp4"
    assert versions[0].render_path == "v001/render"
    assert versions[0].qa["score"] == 0.9

    db.approve_shot_version(project_id, "S1", 1)
    approved = db.latest_approved_version(project_id, "S1")
    assert approved is not None
    assert approved.video_path == "v001/S1.mp4", (
        "the approved version must be the one holding the finished video"
    )


def test_update_shot_version_ignores_unknown_fields(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.upsert_shot(project_id, "S1")
    db.add_shot_version(project_id, "S1")
    db.update_shot_version(project_id, "S1", 1, nonsense="x", render_path="r")
    assert db.list_shot_versions(project_id, "S1")[0].render_path == "r"


def test_asset_reuse_lookup(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.upsert_asset(project_id, "character", "Akira", "/a/v1.blend",
                    version=1, canonical=True)
    db.upsert_asset(project_id, "character", "Akira", "/a/v2.blend",
                    version=2, canonical=True)
    db.upsert_asset(project_id, "prop", "Umbrella", "/p/v1.blend", version=1)

    found = db.find_assets(project_id, kind="character", name="Akira")
    assert len(found) == 2
    assert found[0]["version"] == 2  # newest first
    assert len(db.find_assets(project_id, canonical_only=True)) == 2
    assert len(db.find_assets(project_id, kind="prop")) == 1


def test_character_canonical_and_continuity_merge(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.upsert_character(project_id, "Akira",
                        canonical={"hair": "black, shoulder length"},
                        continuity={"location": "rooftop"})
    db.upsert_character(project_id, "Akira",
                        canonical={"eyes": "grey"},
                        continuity={"injury": "cut on left cheek"})

    character = db.get_character(project_id, "Akira")
    # Both writes survive; a later update must not erase the canonical record.
    assert character["canonical"]["hair"] == "black, shoulder length"
    assert character["canonical"]["eyes"] == "grey"
    assert character["continuity"]["location"] == "rooftop"
    assert character["continuity"]["injury"] == "cut on left cheek"


def test_owner_decision_queue(db: ProjectDB):
    project_id = db.create_project("P", "")
    decision_id = db.request_decision(
        project_id, "Make the battlefield darker?",
        ["Yes", "No"], urgency="high",
    )
    pending = db.pending_decisions(project_id)
    assert len(pending) == 1
    assert pending[0]["options"] == ["Yes", "No"]

    db.resolve_decision(decision_id, "Yes")
    assert db.pending_decisions(project_id) == []
    assert db.get_decision(decision_id)["answer"] == "Yes"


def test_qa_records_and_latest_lookup(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.record_qa(project_id, "shot", "S1", "preview", QAStatus.NEEDS_REVISION,
                 score=0.3, findings=[{"severity": "major"}])
    db.record_qa(project_id, "shot", "S1", "preview", QAStatus.PASSED, score=0.9)

    latest = db.latest_qa(project_id, "S1", "preview")
    assert latest["status"] == "PASSED"
    assert latest["score"] == 0.9


def test_progress_rollup(db: ProjectDB):
    project_id = db.create_project("P", "")
    for index in range(4):
        shot_id = f"S{index}"
        db.upsert_shot(project_id, shot_id, "SC01", index, 2.0)
        if index < 1:
            db.set_shot_status(project_id, shot_id, "APPROVED")

    progress = db.progress(project_id)
    assert progress["shots_total"] == 4
    assert progress["shots_approved"] == 1
    assert progress["shots_percent"] == 25.0


def test_event_log_is_ordered_newest_first(db: ProjectDB):
    project_id = db.create_project("P", "")
    db.log_event(project_id, "a", "first")
    db.log_event(project_id, "b", "second")
    events = db.list_events(project_id)
    assert events[0]["message"] == "second"


def test_artifact_roundtrip():
    artifact = Artifact(kind="image", path="/x.png", label="preview",
                        metadata={"width": 480})
    restored = Artifact.from_dict(artifact.to_dict())
    assert restored.metadata == {"width": 480}

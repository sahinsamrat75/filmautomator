"""Vision Agent judgement and the Blender Agent's correction logic.

These are the two halves of the revision loop, and they must agree: every
parameter path the Vision Agent emits has to be one the Blender Agent can
actually apply. The cross-check at the bottom enforces that.
"""

from __future__ import annotations

import pytest

from conftest import metrics

from filmautomator.agents import BlenderAgent, VisionAgent
from filmautomator.agents.blender_agent import _DEFAULT_LENS, _FILL_RATIO
from filmautomator.core.spec import (
    CameraSpec,
    LightingSpec,
    QualityFinding,
    ShotSpec,
    SubjectSpec,
)


@pytest.fixture
def subject_shot() -> ShotSpec:
    return ShotSpec(
        shot_id="SC01_SH01", duration_s=4.0,
        description="A figure stands in the rain.",
        camera=CameraSpec(shot_size="medium", lens_mm=50.0),
        lighting=LightingSpec(mood="natural", key_energy=1400.0,
                              fill_energy=500.0, world_strength=1.0),
        subjects=[SubjectSpec(name="Figure", height_m=1.8)],
    )


# -- measurement-driven findings ------------------------------------------


def test_well_exposed_frame_produces_no_findings(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(metrics(), subject_shot)
    assert findings == []


def test_black_frame_is_critical_and_raises_exposure(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(
        metrics(luma_mean=0.02, luma_std=0.01, foreground_coverage=0.0),
        subject_shot,
    )
    lighting = [f for f in findings if f.category == "lighting"]
    assert lighting and lighting[0].severity == "critical"
    # The suggested fix must actually raise energy, not just describe it.
    assert lighting[0].parameters["lighting.key_energy"] > subject_shot.lighting.key_energy


def test_underlit_frame_is_major(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(metrics(luma_mean=0.10), subject_shot)
    lighting = [f for f in findings if f.category == "lighting"]
    assert lighting and lighting[0].severity == "major"


def test_overexposed_frame_lowers_energy(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(metrics(luma_mean=0.95), subject_shot)
    lighting = [f for f in findings if f.category == "lighting"]
    assert lighting
    assert lighting[0].parameters["lighting.key_energy"] < subject_shot.lighting.key_energy


def test_clipped_highlights_detected(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(
        metrics(clipped_highlights=0.4), subject_shot
    )
    assert any("clipped" in f.description.lower() for f in findings)


def test_flat_frame_detected(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(metrics(luma_std=0.01), subject_shot)
    assert any(f.category == "lighting" and "flat" in f.description for f in findings)


def test_empty_frame_is_critical_and_tightens(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.001), subject_shot
    )
    framing = [f for f in findings if f.category == "framing"]
    assert framing and framing[0].severity == "critical"
    assert framing[0].parameters["camera.shot_size"] != "medium"


def test_subject_too_small_tightens_framing(vision: VisionAgent, subject_shot):
    # A medium shot wants roughly 12% coverage; 3% is far too small.
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.03), subject_shot
    )
    framing = [f for f in findings if f.category == "framing"]
    assert framing
    assert framing[0].parameters["camera.shot_size"] in {"medium_close", "close_up"}


def test_subject_too_large_loosens_framing(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.75), subject_shot
    )
    framing = [f for f in findings if f.category == "framing"]
    assert framing
    assert framing[0].parameters["camera.shot_size"] == "medium_wide"


def test_dead_space_detected(vision: VisionAgent, subject_shot):
    findings = vision._findings_from_metrics(
        metrics(dark_row_fraction=0.5), subject_shot
    )
    assert any(f.category == "composition" for f in findings)


def test_a_wide_shot_with_a_distant_figure_is_not_flagged(vision: VisionAgent):
    """A wide establishing shot legitimately puts a figure at ~0.4% of frame.
    The old floor of 2% expected coverage called that an empty frame."""
    spec = ShotSpec(shot_id="s", camera=CameraSpec(shot_size="wide"),
                    subjects=[SubjectSpec(name="F", height_m=1.75)])
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.004, aspect=16 / 9), spec
    )
    assert not findings, f"a correct wide shot was flagged: {findings}"


def test_a_well_framed_medium_shot_is_not_flagged(vision: VisionAgent):
    spec = ShotSpec(shot_id="s", camera=CameraSpec(shot_size="medium"),
                    subjects=[SubjectSpec(name="F", height_m=1.75)])
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.098, aspect=16 / 9), spec
    )
    assert not [f for f in findings if f.category == "framing"]


def test_a_truly_empty_frame_is_still_caught(vision: VisionAgent):
    spec = ShotSpec(shot_id="s", camera=CameraSpec(shot_size="medium"),
                    subjects=[SubjectSpec(name="F", height_m=1.75)])
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.0005, aspect=16 / 9), spec
    )
    framing = [f for f in findings if f.category == "framing"]
    assert framing and framing[0].severity == "critical"


def test_expected_coverage_shrinks_as_the_shot_widens(vision: VisionAgent):
    aspect = 16 / 9
    wide = VisionAgent._expected_coverage(
        ShotSpec(shot_id="a", camera=CameraSpec(shot_size="wide")), aspect
    )
    close = VisionAgent._expected_coverage(
        ShotSpec(shot_id="b", camera=CameraSpec(shot_size="close_up")), aspect
    )
    assert wide < close


def test_expected_coverage_is_always_reachable(vision: VisionAgent):
    """No shot size may demand coverage a real frame cannot deliver.

    The previous model asked an extreme close-up for 85% coverage, which is
    unreachable — the revision loop tightened the shot forever and never
    cleared the finding.
    """
    from filmautomator.agents.vision_agent import _MAX_EXPECTED_COVERAGE

    for size in _FILL_RATIO:
        expected = VisionAgent._expected_coverage(
            ShotSpec(shot_id="s", camera=CameraSpec(shot_size=size)), 16 / 9
        )
        assert 0.0 < expected <= _MAX_EXPECTED_COVERAGE, f"{size} -> {expected}"


def test_a_well_framed_close_up_is_not_flagged(vision: VisionAgent):
    """27.5% is what a close-up on a standing figure actually measures."""
    spec = ShotSpec(shot_id="s", camera=CameraSpec(shot_size="close_up"),
                    subjects=[SubjectSpec(name="F", height_m=1.75)])
    findings = vision._findings_from_metrics(
        metrics(foreground_coverage=0.275, aspect=16 / 9), spec
    )
    assert not [f for f in findings if f.category == "framing"], (
        "a correctly framed close-up must not be reported as a framing defect"
    )


def test_tightening_at_the_tightest_size_is_a_no_op(blender_agent: BlenderAgent):
    """The precondition for the Director's no-op guard.

    If the reviewer asks to tighten a shot that is already extreme_close_up,
    the correction must leave the spec unchanged so the loop can notice and
    stop rather than burning rounds on identical rebuilds.
    """
    spec = ShotSpec(shot_id="s", camera=CameraSpec(shot_size="extreme_close_up"))
    findings = [
        QualityFinding(category="framing", severity="major",
                       description="Subject occupies 27.5% but 55.0% was intended.",
                       parameters={"camera.shot_size": "extreme_close_up"}),
    ]
    revised, _ = blender_agent.apply_corrections(spec, findings)
    assert revised.to_dict() == spec.to_dict()


def test_score_falls_with_severity(vision: VisionAgent):
    clean = vision._score(metrics(), [])
    minor = vision._score(metrics(), [
        QualityFinding(category="technical", severity="minor", description="x")
    ])
    critical = vision._score(metrics(), [
        QualityFinding(category="lighting", severity="critical", description="x")
    ])
    assert clean > minor > critical
    assert 0.0 <= critical <= 1.0


def test_inspection_without_a_vision_model_is_flagged_heuristic(
    vision: VisionAgent, subject_shot, tmp_path
):
    """The gateway in tests has no reachable model, so this exercises the
    measurement-only path — and that path must announce itself."""
    import json

    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeSession:
        def call(self, op, **kwargs):
            assert op == "analyze_image"
            return metrics()

    vision.ctx.session = FakeSession()  # type: ignore[assignment]
    report = vision.inspect(
        _task(), subject_shot, str(image), iteration=1
    )
    assert report.heuristic_only is True
    assert report.passed is True
    # And the QA record must have landed in the database.
    assert vision.ctx.db.latest_qa("proj_test", "SC01_SH01", "preview") is not None


def test_inspection_fails_a_dark_frame(vision: VisionAgent, subject_shot, tmp_path):
    image = tmp_path / "dark.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeSession:
        def call(self, op, **kwargs):
            return metrics(luma_mean=0.01, luma_std=0.005, foreground_coverage=0.0)

    vision.ctx.session = FakeSession()  # type: ignore[assignment]
    report = vision.inspect(_task(), subject_shot, str(image), iteration=1)
    assert report.passed is False
    assert report.blocking_findings


def _task():
    from filmautomator.core.task import Task

    return Task(objective="inspect", agent="vision_agent", project_id="proj_test")


# -- corrections -----------------------------------------------------------


def test_apply_corrections_sets_dotted_paths(blender_agent: BlenderAgent, subject_shot):
    findings = [
        QualityFinding(category="lighting", severity="major", description="dark",
                       parameters={"lighting.key_energy": 2600.0,
                                   "lighting.fill_energy": 800.0}),
        QualityFinding(category="framing", severity="major", description="small",
                       parameters={"camera.shot_size": "close_up",
                                   "camera.lens_mm": 85.0}),
    ]
    revised, applied = blender_agent.apply_corrections(subject_shot, findings)
    assert revised.lighting.key_energy == 2600.0
    assert revised.camera.shot_size == "close_up"
    assert revised.camera.lens_mm == 85.0
    # The original must be untouched — versions depend on this.
    assert subject_shot.lighting.key_energy == 1400.0
    assert subject_shot.camera.shot_size == "medium"
    assert len(applied) == 4


def test_apply_corrections_can_target_an_indexed_subject(
    blender_agent: BlenderAgent, subject_shot
):
    findings = [
        QualityFinding(category="composition", severity="minor", description="move",
                       parameters={"subjects[0].location": [1.0, 0.0, 0.0]}),
    ]
    revised, _ = blender_agent.apply_corrections(subject_shot, findings)
    assert tuple(revised.subjects[0].location) == (1.0, 0.0, 0.0)


def test_unknown_parameter_is_reported_not_silently_dropped(
    blender_agent: BlenderAgent, subject_shot
):
    findings = [
        QualityFinding(category="lighting", severity="major", description="x",
                       parameters={"lighting.nonexistent_knob": 5}),
    ]
    _, applied = blender_agent.apply_corrections(subject_shot, findings)
    assert any("SKIPPED" in line for line in applied)


def test_heuristic_corrections_when_no_parameters_supplied(
    blender_agent: BlenderAgent, subject_shot
):
    findings = [
        QualityFinding(category="lighting", severity="major",
                       description="Frame is underlit."),
    ]
    revised, applied = blender_agent.apply_corrections(subject_shot, findings)
    assert revised.lighting.key_energy > subject_shot.lighting.key_energy
    assert applied


def test_out_of_range_subject_index_is_rejected_safely(
    blender_agent: BlenderAgent, subject_shot
):
    findings = [
        QualityFinding(category="composition", severity="minor", description="x",
                       parameters={"subjects[7].location": [0, 0, 0]}),
    ]
    _, applied = blender_agent.apply_corrections(subject_shot, findings)
    assert any("SKIPPED" in line for line in applied)


# -- the contract between the two agents ----------------------------------


def test_every_framing_correction_names_a_real_shot_size(
    vision: VisionAgent, blender_agent: BlenderAgent, subject_shot
):
    """The Vision Agent must never emit a shot size the Blender Agent cannot
    look up, or the correction silently becomes a no-op."""
    for coverage in (0.0, 0.005, 0.01, 0.02, 0.05, 0.3, 0.9, 1.0):
        findings = vision._findings_from_metrics(
            metrics(foreground_coverage=coverage), subject_shot
        )
        for finding in findings:
            size = finding.parameters.get("camera.shot_size")
            if size is not None:
                assert size in _FILL_RATIO, f"{size} has no framing entry"
                assert size in _DEFAULT_LENS, f"{size} has no default lens"


def test_every_lighting_correction_targets_a_real_field(
    vision: VisionAgent, blender_agent: BlenderAgent, subject_shot
):
    for luma in (0.0, 0.05, 0.1, 0.5, 0.9, 1.0):
        findings = vision._findings_from_metrics(
            metrics(luma_mean=luma), subject_shot
        )
        revised, applied = blender_agent.apply_corrections(subject_shot, findings)
        assert not any("SKIPPED" in line for line in applied), (
            f"luma={luma} produced an unroutable correction: {applied}"
        )


def test_repeated_corrections_do_not_explode_the_values(
    blender_agent: BlenderAgent, subject_shot
):
    """Four rounds of "still too dark" must not reach physically absurd
    energies — the loop is capped, but each step should still be sane."""
    spec = subject_shot
    finding = QualityFinding(category="lighting", severity="major",
                             description="underlit", parameters={})
    for _ in range(4):
        spec, _ = blender_agent.apply_corrections(spec, [finding])
    assert spec.lighting.key_energy < 100_000

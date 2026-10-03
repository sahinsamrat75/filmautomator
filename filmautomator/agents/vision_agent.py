"""Vision Agent — inspects a rendered frame and judges it (spec sections 6, 15).

Two independent passes, deliberately:

1. **Measurement.** Objective statistics computed from the pixels by the
   Blender-side ``analyze_image`` op. This always runs, needs no model, and
   cannot hallucinate. It reliably catches a dark frame, a flat frame, a blown
   frame, and a subject that is too small or missing.

2. **Judgement.** If a vision-capable model is available, it is asked to assess
   the frame against the shot's own quality criteria and return findings with
   concrete parameter corrections.

The report records which passes contributed. A report built only from
measurement is flagged ``heuristic_only`` so the owner is never told a model
approved something it never saw.
"""

from __future__ import annotations

from typing import Any

from ..core.spec import QualityFinding, QualityReport, ShotSpec
from ..core.task import Task
from ..gateway import Capability, ImageInput
from .base import Agent

#: A standing figure is roughly this wide relative to its height. Used to turn
#: "the subject should fill X of frame height" into a fraction of frame *area*.
_SUBJECT_ASPECT = 0.26

#: fill-ratio per shot size, mirroring the Blender Agent's framing table.
_EXPECTED_FILL = {
    "extreme_wide": 0.06, "wide": 0.16, "full": 0.55, "medium_wide": 0.66,
    "medium": 0.80, "medium_close": 1.10, "close_up": 1.90,
    "extreme_close_up": 3.50,
}

#: No shot size should ever be expected to cover more than this much of frame.
#: A close-up frames a head and shoulders, not the entire rectangle.
_MAX_EXPECTED_COVERAGE = 0.55

_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["framing", "lighting", "composition",
                                 "continuity", "technical"],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "major", "critical"],
                    },
                    "description": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Dotted spec paths to change, e.g. "
                            '{"camera.shot_size": "close_up", '
                            '"lighting.key_energy": 2400}'
                        ),
                    },
                },
                "required": ["category", "severity", "description"],
            },
        },
    },
    "required": ["passed", "score", "findings"],
}


class VisionAgent(Agent):
    name = "vision_agent"
    role = "Inspects rendered frames and reports defects the Director can act on."

    def inspect(self, task: Task, spec: ShotSpec, image_path: str,
                *, background_path: str | None = None,
                iteration: int = 1) -> QualityReport:
        """Judge one rendered frame against the shot's intent."""
        self.announce(f"inspecting {spec.shot_id} preview (pass {iteration})")

        metrics = self._measure(image_path, background_path)
        findings = self._findings_from_metrics(metrics, spec)

        model_report = self._model_judgement(spec, image_path, metrics)
        if model_report is not None:
            findings.extend(model_report.findings)
            summary = model_report.summary
            heuristic_only = False
            model_passed = model_report.passed
        else:
            summary = self._heuristic_summary(metrics, findings)
            heuristic_only = True
            model_passed = None

        blocking = [f for f in findings if f.severity in {"major", "critical"}]
        # Measurement always has veto power: a model saying "looks great" does
        # not override a frame that is measurably black.
        passed = not blocking and (model_passed is not False)

        report = QualityReport(
            passed=passed,
            score=self._score(metrics, findings),
            findings=findings,
            summary=summary,
            heuristic_only=heuristic_only,
        )

        self.ctx.db.record_qa(
            self.ctx.project_id, "shot", spec.shot_id, "preview",
            "PASSED" if passed else "NEEDS_REVISION",
            score=report.score,
            findings=[f.to_dict() for f in findings],
        )
        for finding in blocking:
            self.log.info("  [%s/%s] %s", finding.severity, finding.category,
                          finding.description)
        return report

    # -- pass 1: measurement ----------------------------------------------

    def _measure(self, image_path: str, background_path: str | None = None) -> dict:
        if self.ctx.session is None:
            raise RuntimeError("Vision Agent needs a Blender session to measure pixels")
        args: dict[str, Any] = {"filepath": image_path}
        if background_path:
            args["background_filepath"] = background_path
        metrics = self.ctx.session.call("analyze_image", **args)
        method = metrics.get("coverage_method", "unknown")
        if method != "background_plate":
            self.log.warning(
                "subject coverage measured by %s rather than a background plate; "
                "on a lit stage this is a rough estimate — %s",
                method, metrics.get("plate_note") or "no plate supplied",
            )
        return metrics

    def _findings_from_metrics(self, metrics: dict, spec: ShotSpec) -> list[QualityFinding]:
        findings: list[QualityFinding] = []

        luma_mean = metrics["luma_mean"]
        luma_std = metrics["luma_std"]
        coverage = metrics["foreground_coverage"]

        # -- exposure --
        if luma_mean < 0.06:
            findings.append(QualityFinding(
                category="lighting", severity="critical",
                description=f"Frame is essentially black (mean luminance {luma_mean:.3f}).",
                suggested_fix="Raise key and fill energy substantially.",
                parameters={
                    "lighting.key_energy": round(spec.lighting.key_energy * 3.0, 1),
                    "lighting.fill_energy": round(spec.lighting.fill_energy * 3.0, 1),
                    "lighting.world_strength": round(spec.lighting.world_strength * 2.5, 2),
                },
            ))
        elif luma_mean < 0.14:
            findings.append(QualityFinding(
                category="lighting", severity="major",
                description=f"Frame is underlit (mean luminance {luma_mean:.3f}).",
                suggested_fix="Raise key and fill energy.",
                parameters={
                    "lighting.key_energy": round(spec.lighting.key_energy * 1.8, 1),
                    "lighting.fill_energy": round(spec.lighting.fill_energy * 1.7, 1),
                },
            ))
        elif luma_mean > 0.82:
            findings.append(QualityFinding(
                category="lighting", severity="major",
                description=f"Frame is overexposed (mean luminance {luma_mean:.3f}).",
                suggested_fix="Lower key and fill energy.",
                parameters={
                    "lighting.key_energy": round(spec.lighting.key_energy * 0.5, 1),
                    "lighting.fill_energy": round(spec.lighting.fill_energy * 0.6, 1),
                },
            ))

        if metrics["clipped_highlights"] > 0.20:
            findings.append(QualityFinding(
                category="lighting", severity="major",
                description=(
                    f"{metrics['clipped_highlights']:.0%} of the frame is clipped to "
                    "white; highlight detail is lost."
                ),
                suggested_fix="Reduce key energy.",
                parameters={"lighting.key_energy": round(spec.lighting.key_energy * 0.6, 1)},
            ))

        # -- contrast --
        if luma_std < 0.035 and luma_mean > 0.06:
            findings.append(QualityFinding(
                category="lighting", severity="major",
                description=(
                    f"Frame is flat (luminance std {luma_std:.3f}); there is very "
                    "little separation between light and shadow."
                ),
                suggested_fix="Increase the key-to-fill ratio for more shape.",
                parameters={
                    "lighting.key_energy": round(spec.lighting.key_energy * 1.6, 1),
                    "lighting.fill_energy": round(spec.lighting.fill_energy * 0.55, 1),
                },
            ))

        # -- subject presence and scale --
        expected = self._expected_coverage(spec, metrics["aspect"])
        # Absolute emptiness. Deliberately a very low bar: a wide establishing
        # shot puts a figure at a few tenths of a percent of frame, and calling
        # that "empty" would reject correct work. Scale errors are the relative
        # test below, not this one.
        if coverage < 0.002:
            findings.append(QualityFinding(
                category="framing", severity="critical",
                description=(
                    f"No subject reads in frame (foreground coverage {coverage:.1%}); "
                    "the shot is empty or the subject is out of view."
                ),
                suggested_fix="Tighten framing and confirm the subject is inside the frustum.",
                parameters={"camera.shot_size": self._tighten(spec.camera.shot_size)},
            ))
        elif expected > 0 and coverage < expected * 0.35:
            findings.append(QualityFinding(
                category="framing", severity="major",
                description=(
                    f"Subject occupies {coverage:.1%} of frame but roughly "
                    f"{expected:.1%} was intended for a {spec.camera.shot_size} shot."
                ),
                suggested_fix="Tighten the shot size or shorten the lens.",
                parameters={"camera.shot_size": self._tighten(spec.camera.shot_size)},
            ))
        elif expected > 0 and coverage > min(0.92, expected * 2.4):
            findings.append(QualityFinding(
                category="framing", severity="minor",
                description=(
                    f"Subject fills {coverage:.1%} of frame, more than the "
                    f"{expected:.1%} intended for a {spec.camera.shot_size} shot."
                ),
                suggested_fix="Loosen the shot size.",
                parameters={"camera.shot_size": self._loosen(spec.camera.shot_size)},
            ))

        # -- dead space --
        if metrics["dark_row_fraction"] > 0.35 or metrics["dark_col_fraction"] > 0.35:
            findings.append(QualityFinding(
                category="composition", severity="minor",
                description=(
                    f"Large dead regions: {metrics['dark_row_fraction']:.0%} of rows "
                    f"and {metrics['dark_col_fraction']:.0%} of columns are near-black."
                ),
                suggested_fix="Recompose or reframe to remove empty space.",
            ))

        # -- detail --
        if metrics["edge_density"] < 0.004 and coverage > 0.02:
            findings.append(QualityFinding(
                category="technical", severity="minor",
                description=(
                    f"Very little detail in frame (edge density "
                    f"{metrics['edge_density']:.4f}); the render may be out of focus, "
                    "untextured, or too low resolution to judge."
                ),
                suggested_fix="Check focus, materials and preview resolution.",
            ))

        return findings

    @staticmethod
    def _expected_coverage(spec: ShotSpec, aspect: float) -> float:
        """Rough fraction of frame area a standing figure should occupy.

        Geometric, not a fitted constant. The figure fills `fill` of the frame
        height when it fits; past that it is cropped and its height contribution
        saturates at 1.0 while its width contribution keeps growing.

        The naive version of this (`aspect_ratio * fill**2 / frame_aspect`,
        clamped to 0.85) demanded 85% coverage from an extreme close-up. No
        close-up can reach that, so the revision loop kept tightening the shot
        and the finding never cleared — three rounds spent chasing a target that
        does not exist. Capping at a realistic maximum and modelling the
        saturation removes the oscillation.
        """
        fill = _EXPECTED_FILL.get(spec.camera.shot_size, 0.80)
        if aspect <= 0:
            return 0.0

        height_fraction = min(1.0, fill)
        width_fraction = min(1.0, (_SUBJECT_ASPECT * fill) / aspect)
        expected = height_fraction * width_fraction
        # The floor only exists to keep the ratio test meaningful; set it badly
        # and a legitimate wide shot — where a distant figure genuinely covers
        # 0.4% of frame — reads as a failure.
        return max(0.002, min(_MAX_EXPECTED_COVERAGE, expected))

    @staticmethod
    def _tighten(shot_size: str) -> str:
        order = ["extreme_wide", "wide", "full", "medium_wide", "medium",
                 "medium_close", "close_up", "extreme_close_up"]
        idx = order.index(shot_size) if shot_size in order else 4
        return order[min(len(order) - 1, idx + 1)]

    @staticmethod
    def _loosen(shot_size: str) -> str:
        order = ["extreme_wide", "wide", "full", "medium_wide", "medium",
                 "medium_close", "close_up", "extreme_close_up"]
        idx = order.index(shot_size) if shot_size in order else 4
        return order[max(0, idx - 1)]

    @staticmethod
    def _heuristic_summary(metrics: dict, findings: list[QualityFinding]) -> str:
        if not findings:
            return (
                f"Measurement pass found no defects: mean luminance "
                f"{metrics['luma_mean']:.3f}, contrast {metrics['luma_std']:.3f}, "
                f"subject coverage {metrics['foreground_coverage']:.1%}."
            )
        worst = max(
            findings,
            key=lambda f: {"minor": 0, "major": 1, "critical": 2}[f.severity],
        )
        return (
            f"Measurement pass found {len(findings)} issue(s); most severe is "
            f"{worst.severity} ({worst.category}): {worst.description}"
        )

    @staticmethod
    def _score(metrics: dict, findings: list[QualityFinding]) -> float:
        """A 0-1 quality score that penalises by severity and exposure error."""
        penalty = {"minor": 0.07, "major": 0.20, "critical": 0.42}
        score = 1.0
        for finding in findings:
            score -= penalty.get(finding.severity, 0.1)
        # Penalise distance from a well-exposed mean.
        exposure_error = abs(metrics["luma_mean"] - 0.42)
        score -= min(0.3, exposure_error * 0.6)
        return round(max(0.0, min(1.0, score)), 3)

    # -- pass 2: model judgement ------------------------------------------

    def _model_judgement(self, spec: ShotSpec, image_path: str,
                         metrics: dict) -> QualityReport | None:
        """Ask a vision model. Returns None when no vision model is available."""
        if not self.ctx.gateway.supports(Capability.VISION):
            self.log.info(
                "no vision-capable model available; using measurement pass only"
            )
            return None

        criteria = spec.quality_criteria or [
            "The subject is clearly readable and well framed.",
            "Lighting is motivated and the frame is correctly exposed.",
            "The composition is balanced and cinematic.",
        ]
        prompt = "\n".join([
            "You are a film director's quality-control reviewer looking at a",
            "previsualisation frame rendered from a 3D scene.",
            "",
            f"Shot: {spec.shot_id}",
            f"Intent: {spec.description or 'unspecified'}",
            f"Shot size: {spec.camera.shot_size}, angle: {spec.camera.angle}, "
            f"lens: {spec.camera.lens_mm}mm",
            f"Lighting mood: {spec.lighting.mood}",
            f"Environment: {spec.environment or 'unspecified'}",
            "",
            "Acceptance criteria:",
            *[f"  - {c}" for c in criteria],
            "",
            "Measured statistics for this exact frame:",
            f"  mean luminance {metrics['luma_mean']:.3f} (0=black, 1=white)",
            f"  contrast (std) {metrics['luma_std']:.3f}",
            f"  subject coverage {metrics['foreground_coverage']:.1%} of frame",
            f"  clipped highlights {metrics['clipped_highlights']:.1%}",
            "",
            "Note: the scene uses placeholder proxy geometry for characters,",
            "so judge framing, scale, lighting and composition — not whether",
            "the figure looks like a finished character.",
            "",
            "Report only defects that genuinely matter. If the frame satisfies",
            "the criteria, return passed=true with an empty findings array.",
            "When you do report a finding, include `parameters` giving the",
            "exact spec fields to change, using paths like",
            "camera.shot_size, camera.lens_mm, camera.angle,",
            "lighting.key_energy, lighting.fill_energy, lighting.mood.",
        ])

        try:
            response = self.think(
                Capability.VISION,
                prompt,
                system=(
                    "You are a precise, unsentimental film quality reviewer. "
                    "You respond only with JSON."
                ),
                json_schema=_VISION_SCHEMA,
                images=[ImageInput(path=image_path, label=f"{spec.shot_id} preview")],
                max_tokens=1200,
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to measurement
            self.log.warning("vision model call failed (%s); measurement only", exc)
            return None

        if response.synthetic or response.data is None:
            self.log.info("vision model gave no usable JSON; measurement only")
            return None

        return QualityReport.from_dict(response.data)

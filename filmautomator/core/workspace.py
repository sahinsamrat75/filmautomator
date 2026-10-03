"""On-disk layout for a production.

Everything a project produces lands under a single workspace directory, and
nothing is ever overwritten in place — revisions get a new version directory
(spec section 16).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Workspace:
    """Directory structure for one project."""

    root: Path

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, base: Path, project_id: str, name: str) -> "Workspace":
        root = Path(base) / project_id
        ws = cls(root=root)
        for directory in (
            ws.scenes_dir, ws.shots_dir, ws.assets_dir,
            ws.audio_dir, ws.final_dir, ws.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        ws.write_project_json({"project_id": project_id, "name": name})
        return ws

    # -- static paths ------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.root / "project.db"

    @property
    def project_json(self) -> Path:
        return self.root / "project.json"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # -- shot paths --------------------------------------------------------

    def shot_dir(self, shot_id: str, version: int | None = None) -> Path:
        base = self.shots_dir / shot_id
        return base / f"v{version:03d}" if version is not None else base

    def shot_blend(self, shot_id: str, version: int) -> Path:
        return self.shot_dir(shot_id, version) / f"{shot_id}.blend"

    def shot_preview(self, shot_id: str, version: int) -> Path:
        return self.shot_dir(shot_id, version) / "preview.png"

    def shot_preview_plate(self, shot_id: str, version: int) -> Path:
        """The same preview frame rendered with the subjects hidden.

        Differencing the two gives an exact subject mask, which is what the
        Vision Agent's coverage measurement needs on a lit stage.
        """
        return self.shot_dir(shot_id, version) / "preview_background.png"

    def shot_render_dir(self, shot_id: str, version: int) -> Path:
        return self.shot_dir(shot_id, version) / "render"

    def shot_video(self, shot_id: str, version: int) -> Path:
        return self.shot_dir(shot_id, version) / f"{shot_id}.mp4"

    def shot_spec_path(self, shot_id: str, version: int) -> Path:
        return self.shot_dir(shot_id, version) / "spec.json"

    def prepare_shot_version(self, shot_id: str, version: int) -> Path:
        directory = self.shot_dir(shot_id, version)
        directory.mkdir(parents=True, exist_ok=True)
        self.shot_render_dir(shot_id, version).mkdir(parents=True, exist_ok=True)
        return directory

    # -- final outputs -----------------------------------------------------

    def final_video(self, name: str = "final.mp4") -> Path:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        return self.final_dir / name

    def report(self, name: str) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return self.reports_dir / name

    # -- helpers -----------------------------------------------------------

    def write_project_json(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def write_json(self, relative: str | Path, data: Any) -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return target

    def relative(self, path: str | Path) -> str:
        """Path relative to the workspace root, for storing in the database."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def resolve(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate

    def disk_usage_mb(self) -> float:
        total = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)

    def clear_shot_version(self, shot_id: str, version: int) -> None:
        """Remove a single version directory. Earlier versions are untouched."""
        directory = self.shot_dir(shot_id, version)
        if directory.exists():
            shutil.rmtree(directory)

"""On-disk layout for a production.

Everything a project produces lives under one workspace directory, and nothing
is ever overwritten in place — revisions get a new version directory
(spec section 16).

Layout (spec section 9)::

    projects/<project_id>/
        project.db
        story/
        assets/{characters,environments,props,materials,animations}/
        scenes/
        shots/
        previews/
        renders/
        audio/{dialogue,music,sfx}/
        editorial/
        qa/
        final/
        logs/

The final movie is always written to ``final/`` and named after the project, so
the owner is never left guessing where it went.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Asset categories, each of which gets its own directory (spec section 9).
ASSET_KINDS = ("characters", "environments", "props", "materials", "animations")

#: Audio categories.
AUDIO_KINDS = ("dialogue", "music", "sfx")


def slugify(text: str, *, fallback: str = "PROJECT") -> str:
    """Turn a project name into something safe for a filename.

    Kept deliberately conservative — this value ends up in a path the owner
    will type.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return cleaned[:64] or fallback


@dataclass
class Workspace:
    """Directory structure for one project."""

    root: Path
    #: Used to name the final movie, e.g. AWAKE_EP01 -> AWAKE_EP01_FINAL.mp4
    project_slug: str = "PROJECT"

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, base: Path, project_id: str, name: str) -> "Workspace":
        root = Path(base) / project_id
        ws = cls(root=root, project_slug=slugify(name) or slugify(project_id))
        ws.ensure_dirs()
        ws.write_project_json({"project_id": project_id, "name": name})
        return ws

    def ensure_dirs(self) -> None:
        """Create the full directory tree. Safe to call repeatedly."""
        for directory in (
            self.story_dir, self.scenes_dir, self.shots_dir, self.previews_dir,
            self.renders_dir, self.editorial_dir, self.qa_dir, self.final_dir,
            self.logs_dir,
            *(self.asset_dir(kind) for kind in ASSET_KINDS),
            *(self.audio_dir_for(kind) for kind in AUDIO_KINDS),
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- static paths ------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.root / "project.db"

    @property
    def project_json(self) -> Path:
        return self.root / "project.json"

    @property
    def story_dir(self) -> Path:
        return self.root / "story"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    @property
    def previews_dir(self) -> Path:
        return self.root / "previews"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def editorial_dir(self) -> Path:
        return self.root / "editorial"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def reports_dir(self) -> Path:
        """The existing production report lives here."""
        return self.root / "reports"

    def asset_dir(self, kind: str) -> Path:
        return self.assets_dir / kind

    def audio_dir_for(self, kind: str) -> Path:
        return self.audio_dir / kind

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
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        return directory

    def mirror_preview(self, shot_id: str, version: int) -> Path | None:
        """Copy the shot preview into previews/ under a stable name.

        The dashboard and the owner both benefit from one obvious place where
        the latest preview of a shot can be found without knowing the version
        numbering.
        """
        source = self.shot_preview(shot_id, version)
        if not source.is_file():
            return None
        target = self.previews_dir / f"{shot_id}_v{version:03d}.png"
        shutil.copy2(source, target)
        return target

    # -- final outputs -----------------------------------------------------

    def final_movie_name(self) -> str:
        return f"{self.project_slug}_FINAL.mp4"

    def final_video(self, name: str | None = None) -> Path:
        self.final_dir.mkdir(parents=True, exist_ok=True)
        return self.final_dir / (name or self.final_movie_name())

    def report(self, name: str) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return self.reports_dir / name

    def qa_report(self, name: str = "qa_report.json") -> Path:
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        return self.qa_dir / name

    def log_file(self, name: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir / name

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

    def tree(self, limit: int = 200) -> list[str]:
        """Relative paths of everything under the workspace, for reporting."""
        out: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if len(out) >= limit:
                break
            out.append(str(path.relative_to(self.root)))
        return out

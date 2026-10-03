"""Storage governor — the disk-space budget production runs inside.

A feature film rendered shot-by-shot produces far more data than it keeps. The
frames of a shot are the largest thing this system ever writes, and once a shot
has been encoded and verified they are superseded: the MP4 replaces them. Left
alone, a long production fills the disk and then fails mid-render.

This module makes that explicit rather than incidental. It answers three
questions, in this order:

    1. How much working storage is Filmautomator using right now?
    2. Which files are disposable, and which are load-bearing?
    3. Is it safe to delete this one, *right now*?

The governing rule, and the reason this is a separate module rather than a few
``shutil.rmtree`` calls:

    NEVER DELETE AN INTERMEDIATE UNTIL ITS REPLACEMENT HAS BEEN SUCCESSFULLY
    PROMOTED AND VERIFIED.

So rendered frames are *candidates*, not garbage. A frame directory only becomes
deletable once its shot's MP4 physically exists, is non-zero, and probes as real
video. A failed render keeps its frames, because those frames are the only copy
of the work. A failed encode keeps them too. That is the whole point: cleanup is
only ever allowed to reclaim space that a verified deliverable already accounts
for.

Storage is measured over the workspace root — every project — because the disk
constraint belongs to the machine, not to one film. Thresholds default to a 35 GB
ceiling because that is what leaves the owner's own files room to exist:

    NORMAL             < 25 GB
    WARNING            25-30 GB
    AGGRESSIVE_CLEANUP 30-35 GB
    HARD_LIMIT         >= 35 GB

At the hard limit the answer is never "delete something and hope". The governor
returns a refusal, names the candidates it would have needed, and lets the
caller stop safely.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from ..post.ffmpeg import FFmpegError, VideoEncoder

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

#: Binary gigabyte. Disk sizes are reported in these, and the ceiling is about
#: real free space rather than a decimal marketing figure.
GB = 1024 ** 3
MB = 1024 ** 2


class StorageState(str, Enum):
    """How much room is left, in terms production can act on."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    AGGRESSIVE_CLEANUP = "AGGRESSIVE_CLEANUP"
    HARD_LIMIT = "HARD_LIMIT"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


@dataclass(frozen=True)
class StorageThresholds:
    """Where the states change. Configurable because machines differ."""

    warning_bytes: int = 25 * GB
    aggressive_bytes: int = 30 * GB
    hard_limit_bytes: int = 35 * GB

    def state_for(self, used_bytes: int) -> StorageState:
        if used_bytes >= self.hard_limit_bytes:
            return StorageState.HARD_LIMIT
        if used_bytes >= self.aggressive_bytes:
            return StorageState.AGGRESSIVE_CLEANUP
        if used_bytes >= self.warning_bytes:
            return StorageState.WARNING
        return StorageState.NORMAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_gb": round(self.warning_bytes / GB, 2),
            "aggressive_gb": round(self.aggressive_bytes / GB, 2),
            "hard_limit_gb": round(self.hard_limit_bytes / GB, 2),
        }
# -- artifact categories ----------------------------------------------------

CAT_FRAMES = "render_frames"
CAT_PREVIEW = "preview"
CAT_INTERMEDIATE = "intermediate"
CAT_TEMP = "temporary"
CAT_FAILED = "failed_render_leftover"
CAT_FINAL = "final_deliverable"
CAT_SHOT_VIDEO = "shot_video"
CAT_REPORT = "report"
CAT_SCENE = "scene"
CAT_AUDIO = "audio"
CAT_DATABASE = "database"
CAT_OTHER = "other"

#: Categories that hold work. Nothing here is ever a cleanup candidate, whatever
#: the disk says — losing a final movie to make room for frames is not a trade,
#: it is data loss.
PROTECTED_CATEGORIES: frozenset[str] = frozenset({
    CAT_FINAL, CAT_SHOT_VIDEO, CAT_REPORT, CAT_SCENE, CAT_AUDIO, CAT_DATABASE,
})

#: Categories cleanup may consider, in the order it should consider them. Frames
#: first because they dominate; previews are small but numerous.
DISPOSABLE_CATEGORIES: tuple[str, ...] = (
    CAT_FAILED, CAT_FRAMES, CAT_TEMP, CAT_INTERMEDIATE, CAT_PREVIEW,
)

CATEGORY_LABELS: dict[str, str] = {
    CAT_FRAMES: "Rendered frames",
    CAT_PREVIEW: "Preview images",
    CAT_INTERMEDIATE: "Intermediate renders and timelines",
    CAT_TEMP: "Temporary files",
    CAT_FAILED: "Failed-render leftovers",
    CAT_FINAL: "Final movies",
    CAT_SHOT_VIDEO: "Shot videos",
    CAT_REPORT: "QA and production reports",
    CAT_SCENE: "Blender scenes",
    CAT_AUDIO: "Audio",
    CAT_DATABASE: "Project databases",
    CAT_OTHER: "Other files",
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class StorageEntry:
    """One path on disk, with what it is and what may be done to it."""

    path: Path
    category: str
    bytes: int
    shot_id: str = ""
    version: int = 0
    #: True when this specific file may be deleted right now.
    disposable: bool = False
    #: Why it may not be. Empty when ``disposable`` is True.
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "category": self.category,
            "label": CATEGORY_LABELS.get(self.category, self.category),
            "bytes": self.bytes,
            "size_mb": round(self.bytes / MB, 2),
            "shot_id": self.shot_id,
            "version": self.version,
            "disposable": self.disposable,
            "reason": self.reason,
        }


@dataclass
class StorageUsage:
    """A point-in-time measurement of working storage."""

    root: str
    total_bytes: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    disposable_bytes: int = 0
    protected_bytes: int = 0
    state: StorageState = StorageState.NORMAL
    thresholds: StorageThresholds = field(default_factory=StorageThresholds)
    free_disk_bytes: int = 0
    #: Entries that look reclaimable but are being held back, with the reason.
    retained: list[StorageEntry] = field(default_factory=list)
    error: str = ""

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    @property
    def percent_of_limit(self) -> float:
        limit = self.thresholds.hard_limit_bytes
        return round(100.0 * self.total_bytes / limit, 1) if limit else 0.0

    @property
    def can_start_large_render(self) -> bool:
        """Whether a big render may begin.

        False at the hard limit when there is nothing safe to delete — "we are
        full" is only actionable if something can actually be freed.
        """
        if self.state is StorageState.HARD_LIMIT:
            return self.disposable_bytes > 0
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "state": self.state.value,
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_gb, 3),
            "percent_of_limit": self.percent_of_limit,
            "disposable_bytes": self.disposable_bytes,
            "disposable_gb": round(self.disposable_bytes / GB, 3),
            "protected_bytes": self.protected_bytes,
            "free_disk_gb": round(self.free_disk_bytes / GB, 2),
            "thresholds": self.thresholds.to_dict(),
            "by_category": {
                category: {
                    "bytes": size,
                    "gb": round(size / GB, 3),
                    "label": CATEGORY_LABELS.get(category, category),
                }
                for category, size in sorted(
                    self.by_category.items(), key=lambda kv: -kv[1]
                )
            },
            "retained_count": len(self.retained),
            "retained": [e.to_dict() for e in self.retained[:50]],
            "can_start_large_render": self.can_start_large_render,
            "error": self.error,
        }

    def render(self) -> str:
        lines = [
            f"Working storage: {self.total_gb:.2f} GB of "
            f"{self.thresholds.hard_limit_bytes / GB:.0f} GB "
            f"({self.percent_of_limit}%) — {self.state.value}",
            f"  reclaimable now:  {self.disposable_bytes / GB:.2f} GB",
            f"  protected:         {self.protected_bytes / GB:.2f} GB",
        ]
        for category, size in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
            if size <= 0:
                continue
            lines.append(f"    {CATEGORY_LABELS.get(category, category):<36} "
                         f"{size / GB:7.3f} GB")
# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def parse_shot_location(relative: Path) -> tuple[str, int]:
    """Recover ``(shot_id, version)`` from a path under a project's ``shots/``.

    Layout is ``<project>/shots/<shot_id>/v<NNN>/<...>``, so this is a pure
    function of the path and needs no database round-trip to classify a file.
    """
    parts = relative.parts
    if "shots" not in parts:
        return "", 0
    index = parts.index("shots")
    if index + 2 >= len(parts):
        return "", 0
    shot_id = parts[index + 1]
    version_part = parts[index + 2]
    if version_part.startswith("v") and version_part[1:].isdigit():
        return shot_id, int(version_part[1:])
    return shot_id, 0


def classify(relative: Path) -> str:
    """Name the category of one file, from its position in the workspace.

    Classification is structural rather than registry-driven on purpose: a file
    that exists but was never registered still has to be accounted for, or it
    becomes invisible space the governor can never reclaim.
    """
    parts = relative.parts
    name = relative.name
    suffix = relative.suffix.lower()

    if suffix in {".db", ".sqlite", ".sqlite3"} or ".db-" in name:
        return CAT_DATABASE
    if suffix in {".blend", ".blend1"}:
        return CAT_SCENE

    # A frame sequence is the big one: shots/<id>/v<NNN>/render/*.
    if "render" in parts and suffix in {".png", ".jpg", ".jpeg", ".exr", ".tga"}:
        return CAT_FRAMES

    # Failed or superseded renders leave frames behind outside a live version
    # directory. Same cost, no deliverable, so worth reclaiming first.
    if "render" in parts:
        return CAT_FAILED

    if "final" in parts and suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return CAT_FINAL
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return CAT_SHOT_VIDEO

    if "previews" in parts or name.startswith("preview"):
        return CAT_PREVIEW
    if "editorial" in parts:
        return CAT_INTERMEDIATE
    if "qa" in parts or "reports" in parts:
        return CAT_REPORT
    if "audio" in parts:
        return CAT_AUDIO
    if suffix in {".tmp", ".part", ".temp"} or name.startswith("."):
        return CAT_TEMP
    if "logs" in parts:
        return CAT_INTERMEDIATE
    return CAT_OTHER


def directory_bytes(path: Path) -> tuple[int, int]:
    """``(total_bytes, file_count)`` for a file or a whole directory tree."""
    if path.is_file():
        try:
            return path.stat().st_size, 1
        except OSError:
            return 0, 0
    if not path.is_dir():
        return 0, 0
    total = 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
                count += 1
            except OSError:  # a file vanished mid-walk; it costs nothing
                continue
    return total, count


# ---------------------------------------------------------------------------
# The governor
# ---------------------------------------------------------------------------


class StorageRefused(RuntimeError):
    """Raised when a large operation cannot proceed for lack of safe space.

    This is a *stop* signal, not a failure to swallow. The caller is expected to
    halt at a checkpoint and tell the owner what is holding the disk, rather
    than deleting something to force progress.
    """

    def __init__(self, message: str, usage: StorageUsage) -> None:
        super().__init__(message)
        self.usage = usage


class StorageGovernor:
    """Measures working storage and reclaims only what is provably disposable.

    The safety rule lives in :meth:`_may_release`, which is the single place in
    the system permitted to conclude that rendered frames may go. It answers one
    question — *has this shot's replacement been promoted and verified?* — and
    everything else follows from that answer.
    """

    def __init__(self, root: str | Path, *,
                 thresholds: StorageThresholds | None = None,
                 db: Any = None,
                 encoder: VideoEncoder | None = None,
                 events: Any = None,
                 ) -> None:
        self.root = Path(root)
        self.thresholds = thresholds or StorageThresholds()
        self.db = db
        self.encoder = encoder or VideoEncoder()
        self.events = events
        #: Every deletion this governor performs, for the audit trail and the
        #: dashboard. Deleting is the one irreversible act here, so it is logged
        #: rather than merely counted.
        self.audit_log: list[dict[str, Any]] = []

    # -- measurement -------------------------------------------------------

    def measure(self) -> StorageUsage:
        """Walk the workspace and classify everything currently on disk."""
        usage = StorageUsage(root=str(self.root), thresholds=self.thresholds)

        try:
            usage.free_disk_bytes = shutil.disk_usage(self.root).free
        except OSError:
            usage.free_disk_bytes = 0

        if not self.root.is_dir():
            usage.state = self.thresholds.state_for(0)
            return usage

        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(self.root)
            except ValueError:  # pragma: no cover - rglob guarantees this
                continue
            size = path.stat().st_size
            category = classify(relative)
            shot_id, version = parse_shot_location(relative)

            usage.total_bytes += size
            usage.by_category[category] = usage.by_category.get(category, 0) + size

            entry = StorageEntry(path=path, category=category, bytes=size,
                                 shot_id=shot_id, version=version)

            if category in PROTECTED_CATEGORIES or category not in DISPOSABLE_CATEGORIES:
                # Accounted for in the total, but never a deletion candidate.
                usage.protected_bytes += size
                continue

            if self._may_release(entry):
                usage.disposable_bytes += size
            else:
                usage.protected_bytes += size
                entry.reason = self._retention_reason(entry)
                usage.retained.append(entry)

        # State is computed from the measured total, never assumed up front.
        usage.state = self.thresholds.state_for(usage.total_bytes)
        return usage

    def render(self) -> str:
        lines = [
            f"Working storage: {self.total_gb:.2f} GB of "
            f"{self.thresholds.hard_limit_bytes / GB:.0f} GB "
            f"({self.percent_of_limit}%) — {self.state.value}",
            f"  reclaimable now:  {self.disposable_bytes / GB:.2f} GB",
            f"  protected:         {self.protected_bytes / GB:.2f} GB",
        ]
        for category, size in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
            if size <= 0:
                continue
            label = CATEGORY_LABELS.get(category, category)
            lines.append(f"    {label:<36} {size / GB:7.3f} GB")
        if self.retained:
            lines.append(
                f"  held back (replacement not verified): {len(self.retained)} item(s)"
            )
        if self.error:
            lines.append(f"  measurement error: {self.error}")
        return "\n".join(lines)


    # -- the safety rule ---------------------------------------------------

    def _may_release(self, entry: StorageEntry) -> bool:
        """Is this specific file safe to delete at this moment?

        Protected categories are never released. Everything else is judged by
        whether the thing it was an intermediate *for* has been verified.
        """
        if entry.category in PROTECTED_CATEGORIES:
            return False
        if entry.category in {CAT_FRAMES, CAT_FAILED}:
            return self._frames_are_releasable(entry)
        if entry.category in {CAT_TEMP, CAT_INTERMEDIATE, CAT_PREVIEW}:
            return self._intermediate_is_releasable(entry)
        return False

    def _frames_are_releasable(self, entry: StorageEntry) -> bool:
        """Frames may go only once their shot's MP4 is promoted and verified.

        The sequence this encodes, and it is not negotiable:

            render frames -> encode MP4 -> MP4 physically exists -> ffprobe
            passes -> artifact registry updated -> shot FINAL -> frames released

        Any break in that chain means the frames are still the only copy of the
        work, so they are retained. A failed render retains them. A failed
        encode retains them. A missing MP4 retains them.
        """
        if not entry.shot_id:
            return False
        project_id = self._project_of(entry.path)
        if not project_id:
            return False
        video = self._verified_shot_video(project_id, entry.shot_id, entry.version)
        if video is None:
            return False
        # The registry must also point at the file that is actually there. A row
        # claiming a video that has been moved or deleted is exactly the
        # stale-state failure this system exists to prevent.
        return self._registry_matches(project_id, video)

    def _intermediate_is_releasable(self, entry: StorageEntry) -> bool:
        """Intermediates and temporaries are disposable once the final exists.

        The timeline is the subtle case. It is an *input* to the final movie, not
        a leftover from it, so releasing it before the final has been built would
        delete the very file assembly still needs. The rule is therefore
        successor-based rather than category-based: an intermediate goes only
        once the deliverable it feeds is verified on disk.

        A preview is kept while its shot is still in review — it is what the AI
        is looking at — and released only once that shot has a verified video.
        """
        project_id = self._project_of(entry.path)
        if project_id is None:
            # Scratch outside any project: clear only what is genuinely
            # temporary, never anything that resembles a deliverable.
            return entry.category == CAT_TEMP
        if self._has_verified_final(project_id):
            return True
        if entry.category == CAT_TEMP:
            # Temporary files are always reclaimable; nothing depends on them.
            return True
        if entry.category == CAT_PREVIEW and entry.shot_id:
            return self._frames_are_releasable(entry)
        return False

    def _verified_shot_video(self, project_id: str, shot_id: str,
                             version: int) -> Path | None:
        """The shot's MP4, if it exists, is non-empty, and probes as video."""
        candidates: list[Path] = []
        if self.db is not None:
            try:
                for artifact in self.db.list_artifacts(project_id, "shot_video",
                                                       limit=5000):
                    if artifact.get("shot_id") != shot_id:
                        continue
                    metadata = artifact.get("metadata") or {}
                    if version and metadata.get("version") not in (None, version):
                        continue
                    candidates.append(Path(artifact["path"]))
            except Exception:  # noqa: BLE001 - an unreadable registry proves nothing
                log.debug("registry read failed for %s", project_id, exc_info=True)

        if not candidates:
            # Fall back to the canonical on-disk location so cleanup still works
            # against a workspace whose registry was lost.
            candidates = sorted(
                (self.root / project_id / "shots" / shot_id).glob(
                    "v*/" + f"{shot_id}.mp4")
            )

        for path in candidates:
            if self._is_validated_video(path):
                return path
        return None

    def _is_validated_video(self, path: Path) -> bool:
        """Exists, non-zero, and ffprobe agrees it is playable video."""
        if not path.is_file():
            return False
        try:
            if path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if not self.encoder.available or self.encoder.ffprobe is None:
            # Without ffprobe we cannot verify, so we do not release. Defaulting
            # to "delete anyway" is precisely the failure this rule prevents.
            return False
        try:
            info = self.encoder.probe(path)
        except (FFmpegError, OSError):
            return False
        return bool(info.has_video and info.width > 0 and info.height > 0
                    and info.duration_s > 0)

    def _registry_matches(self, project_id: str, video: Path) -> bool:
        """Does the artifact registry point at this exact, present file?"""
        if self.db is None:
            return True
        try:
            artifacts = self.db.list_artifacts(project_id, "shot_video", limit=5000)
        except Exception:  # noqa: BLE001
            return False
        target = str(video)
        return any(str(a.get("path")) == target and bool(a.get("exists"))
                   for a in artifacts)

    def _has_verified_final(self, project_id: str) -> bool:
        if self.db is None:
            return False
        try:
            artifacts = self.db.list_artifacts(project_id, "final_movie", limit=100)
        except Exception:  # noqa: BLE001
            return False
        return any(a.get("exists") and self._is_validated_video(Path(a["path"]))
                   for a in artifacts)

    def _retention_reason(self, entry: StorageEntry) -> str:
        """Plain-language reason a disposable-looking file is being kept."""
        if entry.category in {CAT_FRAMES, CAT_FAILED}:
            return ("its shot has no verified, registered MP4 yet — the frames "
                    "are still the only copy")
        if entry.category == CAT_PREVIEW:
            return "its shot is still under review"
        if entry.category == CAT_INTERMEDIATE:
            return "the final movie has not been verified yet"
        return "no verified replacement exists"

    def _project_of(self, path: Path) -> str | None:
        """The project id a path belongs to, from its first path segment."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return None
        return relative.parts[0] if relative.parts else None
# -- cleanup -----------------------------------------------------------

    def plan_cleanup(self, usage: StorageUsage | None = None,
                     target_bytes: int = 0) -> list[StorageEntry]:
        """Disposable entries, biggest first, in the order they should go.

        Ordered by size so a modest reclaim clears the most space, and grouped
        so whole frame sets disappear together rather than one file at a time.
        """
        usage = usage or self.measure()
        candidates: list[StorageEntry] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(self.root)
            except ValueError:  # pragma: no cover
                continue
            category = classify(relative)
            if category not in DISPOSABLE_CATEGORIES:
                continue
            shot_id, version = parse_shot_location(relative)
            entry = StorageEntry(path=path, category=category,
                                 bytes=path.stat().st_size,
                                 shot_id=shot_id, version=version)
            if self._may_release(entry):
                candidates.append(entry)
        candidates.sort(key=lambda e: -e.bytes)
        return candidates

    def cleanup(self, *, target_bytes: int = 0,
                project_id: str = "",
                dry_run: bool = False) -> dict[str, Any]:
        """Delete only what is provably disposable, then re-measure.

        ``target_bytes`` is how much the caller wants back; zero means "clear
        everything currently safe to remove". Nothing protected and nothing
        unpromoted is ever touched, regardless of how much space is needed.
        """
        before = self.measure()
        candidates = self.plan_cleanup(before)
        if project_id:
            prefix = self.root / project_id
            candidates = [c for c in candidates
                          if c.path == prefix or str(c.path).startswith(str(prefix) + "/")]

        freed = 0
        removed: list[dict[str, Any]] = []
        for entry in candidates:
            if target_bytes and freed >= target_bytes:
                break
            record = {"path": str(entry.path), "category": entry.category,
                      "bytes": entry.bytes, "shot_id": entry.shot_id,
                      "version": entry.version}
            if dry_run:
                record["dry_run"] = True
                removed.append(record)
                freed += entry.bytes
                continue
            try:
                entry.path.unlink()
            except OSError as exc:
                record["error"] = str(exc)
                removed.append(record)
                continue
            freed += entry.bytes
            record["deleted"] = True
            removed.append(record)
            self.audit_log.append(dict(record, at=_now()))

        # Empty directories left behind are not data; removing them keeps the
        # owner's project tree readable after a cleanup.
        if not dry_run:
            self._prune_empty_dirs()

        after = self.measure()
        result = {
            "before": before.to_dict(),
            "after": after.to_dict(),
            "freed_bytes": freed,
            "freed_gb": round(freed / GB, 4),
            "removed": removed,
            "removed_count": len(removed),
            "dry_run": dry_run,
            "state_before": before.state.value,
            "state_after": after.state.value,
            "production_continues": after.state is not StorageState.HARD_LIMIT,
        }
        if project_id:
            result["project_id"] = project_id
        return result

    def _prune_empty_dirs(self) -> None:
        """Remove directories left empty by cleanup, keeping the workspace shape.

        A project's top-level folders are part of the documented on-disk layout
        and are created once by ``Workspace.ensure_dirs``. Deleting one — and
        ``editorial/`` is empty until a timeline is assembled, so it always is —
        breaks code that later writes into it without recreating it first.

        So pruning is restricted to directories that are unambiguously scratch:
        below a project's top level, and not one of the structural names.
        """
        if not self.root.is_dir():
            return
        structural = {
            "story", "assets", "scenes", "shots", "previews", "renders",
            "editorial", "qa", "final", "logs", "audio", "reports",
            "characters", "environments", "props", "materials", "animations",
            "dialogue", "music", "sfx",
        }
        for path in sorted(self.root.rglob("*"), key=lambda p: -len(p.parts)):
            if not path.is_dir() or path.is_symlink():
                continue
            if path.name in structural:
                continue
            # Only prune below a project's top level, never a project directory
            # and never anything directly beneath the workspace root.
            if len(path.parts) - len(self.root.parts) < 3:
                continue
            try:
                next(path.iterdir())
            except StopIteration:
                try:
                    path.rmdir()
                except OSError:
                    pass
            except OSError:
                continue

    # -- capacity guard ----------------------------------------------------

    def ensure_capacity(self, *, needed_bytes: int = 0,
                        usage: StorageUsage | None = None) -> StorageUsage:
        """Refuse to start a large operation that cannot fit.

        Below the hard limit this is a no-op. At the hard limit it attempts a
        safe cleanup first; if that frees enough, production continues. If it
        cannot, :class:`StorageRefused` is raised so the caller stops at a
        checkpoint rather than filling the disk or deleting something precious.
        """
        usage = usage or self.measure()
        required = usage.total_bytes + max(0, needed_bytes)
        if required <= self.thresholds.hard_limit_bytes:
            return usage

        cleanup = self.cleanup()
        after = StorageUsage(**{k: v for k, v in cleanup["after"].items()
                                if k in StorageUsage.__dataclass_fields__})
        if after.total_bytes + max(0, needed_bytes) <= self.thresholds.hard_limit_bytes:
            return after

        message = (
            f"working storage is {after.total_gb:.2f} GB, at or above the "
            f"{self.thresholds.hard_limit_bytes / GB:.0f} GB limit, and no safe "
            f"cleanup could free enough space "
            f"({cleanup['freed_gb']:.2f} GB reclaimed). Refusing to start another "
            f"large render. Free space by hand or raise the limit in "
            f"filmautomator.toml."
        )
        raise StorageRefused(message, after)

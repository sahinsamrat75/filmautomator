"""Project database — the persistent movie memory of spec section 7.

SQLite, because the first milestone must run on one machine and a movie's
metadata is small; the schema and the access layer are the parts that matter if
this later moves to Postgres or a document store.

Holds: story, characters (canonical + continuity), locations, props, visual
style, scenes, shots and their versions, dialogue, music, SFX, assets, render
status, QA status, plus an event log and the owner-decision queue.

Identity
--------
Shots and scenes are keyed by ``(project_id, shot_id)``, not by ``shot_id``
alone. Every production numbers its shots SC01_SH01 upward, so a global key
meant the second film silently overwrote the first film's shot rows — the
newest project ended up with no shots at all and older projects were left
pointing at another film's data. Shot identity is only meaningful inside a
project, and the schema now says so.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task import Artifact, QAStatus, Task, TaskStatus, new_id

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    objective    TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'ACTIVE',
    workspace    TEXT NOT NULL DEFAULT '',
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id        TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    parent_task_id TEXT,
    agent          TEXT NOT NULL,
    objective      TEXT NOT NULL,
    status         TEXT NOT NULL,
    priority       INTEGER NOT NULL DEFAULT 100,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    payload        TEXT NOT NULL DEFAULT '{}',
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, status);

CREATE TABLE IF NOT EXISTS scenes (
    project_id TEXT NOT NULL,
    scene_id   TEXT NOT NULL,
    ordinal    INTEGER NOT NULL DEFAULT 0,
    title      TEXT NOT NULL DEFAULT '',
    spec       TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'PENDING',
    PRIMARY KEY (project_id, scene_id)
);

CREATE TABLE IF NOT EXISTS shots (
    project_id       TEXT NOT NULL,
    shot_id          TEXT NOT NULL,
    scene_id         TEXT NOT NULL DEFAULT '',
    ordinal          INTEGER NOT NULL DEFAULT 0,
    duration_s       REAL NOT NULL DEFAULT 0,
    spec             TEXT NOT NULL DEFAULT '{}',
    status           TEXT NOT NULL DEFAULT 'PENDING',
    approved_version INTEGER,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (project_id, shot_id)
);
CREATE INDEX IF NOT EXISTS idx_shots_project ON shots(project_id, scene_id, ordinal);

CREATE TABLE IF NOT EXISTS shot_versions (
    version_id  TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    shot_id     TEXT NOT NULL,
    version_no  INTEGER NOT NULL,
    blend_path  TEXT NOT NULL DEFAULT '',
    render_path TEXT NOT NULL DEFAULT '',
    video_path  TEXT NOT NULL DEFAULT '',
    approved    INTEGER NOT NULL DEFAULT 0,
    qa          TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    UNIQUE(project_id, shot_id, version_no)
);
CREATE INDEX IF NOT EXISTS idx_versions_shot ON shot_versions(project_id, shot_id);

CREATE TABLE IF NOT EXISTS assets (
    asset_id   TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    path       TEXT NOT NULL DEFAULT '',
    canonical  INTEGER NOT NULL DEFAULT 0,
    preview    TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, kind, name, version)
);
CREATE INDEX IF NOT EXISTS idx_assets_lookup ON assets(project_id, kind, name);

CREATE TABLE IF NOT EXISTS characters (
    character_id   TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    name           TEXT NOT NULL,
    canonical      TEXT NOT NULL DEFAULT '{}',
    continuity     TEXT NOT NULL DEFAULT '{}',
    asset_id       TEXT NOT NULL DEFAULT '',
    updated_at     TEXT NOT NULL,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS renders (
    render_id  TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    shot_id    TEXT NOT NULL DEFAULT '',
    version_no INTEGER NOT NULL DEFAULT 0,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'PENDING',
    engine     TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_renders_shot ON renders(project_id, shot_id, kind);

CREATE TABLE IF NOT EXISTS qa_results (
    qa_id      TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    subject    TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    scope      TEXT NOT NULL,
    status     TEXT NOT NULL,
    score      REAL,
    findings   TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_subject ON qa_results(project_id, subject_id, scope);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    task_id     TEXT NOT NULL DEFAULT '',
    urgency     TEXT NOT NULL DEFAULT 'normal',
    question    TEXT NOT NULL,
    options     TEXT NOT NULL DEFAULT '[]',
    context     TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'PENDING',
    answer      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    resolved_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(project_id, status);

CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL UNIQUE,
    project_id TEXT NOT NULL,
    task_id    TEXT NOT NULL DEFAULT '',
    agent      TEXT NOT NULL DEFAULT '',
    scene_id   TEXT NOT NULL DEFAULT '',
    shot_id    TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL,
    message    TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id, seq);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    path        TEXT NOT NULL,
    shot_id     TEXT NOT NULL DEFAULT '',
    scene_id    TEXT NOT NULL DEFAULT '',
    bytes       INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_lookup ON artifacts(project_id, kind, created_at);

-- Live production state, written by whichever process is running the
-- production. The CLI, the MCP server and the dashboard are separate
-- processes, so run state cannot live only in memory: a dashboard that read
-- its own empty runner would cheerfully report IDLE while a film was being
-- rendered in another process.
CREATE TABLE IF NOT EXISTS production_runs (
    project_id   TEXT PRIMARY KEY,
    state        TEXT NOT NULL,
    objective    TEXT NOT NULL DEFAULT '',
    stage        TEXT NOT NULL DEFAULT '',
    current_shot TEXT NOT NULL DEFAULT '',
    shots_total  INTEGER NOT NULL DEFAULT 0,
    shots_done   INTEGER NOT NULL DEFAULT 0,
    percent      REAL NOT NULL DEFAULT 0,
    error        TEXT NOT NULL DEFAULT '',
    pid          INTEGER NOT NULL DEFAULT 0,
    started_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL DEFAULT ''
);

-- Control requests travel the same way: the dashboard asks for a pause by
-- writing a row, and the process actually running the production picks it up
-- at its next safe point.
CREATE TABLE IF NOT EXISTS control_requests (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    action      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING',
    created_at  TEXT NOT NULL,
    consumed_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_controls_pending ON control_requests(project_id, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ShotVersion:
    version_id: str
    shot_id: str
    version_no: int
    blend_path: str = ""
    render_path: str = ""
    video_path: str = ""
    approved: bool = False
    qa: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""


class ProjectDB:
    """All persistent project state. Safe to share across threads."""

    def __init__(self, path: str | Path, *, auto_migrate: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if auto_migrate:
            self._set_aside_stale_database()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    def _set_aside_stale_database(self) -> None:
        """Move an older-schema database aside rather than mangling it.

        The v1 schema keyed shots by ``shot_id`` alone, so its shot rows cannot
        be reliably re-attributed to the project that produced them. Rather
        than guess, the file is renamed alongside the new one and rebuilt. No
        rendered artifact is touched — everything under the workspace stays
        exactly where it is, and the production reports on disk remain the
        authoritative record of what was made.
        """
        if not self.path.is_file():
            return
        try:
            probe = sqlite3.connect(str(self.path))
            try:
                row = probe.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()
            finally:
                probe.close()
        except sqlite3.Error:
            return  # unreadable or not ours; let sqlite handle it

        if row is None:
            return  # brand-new or pre-versioning file with no meta table

        try:
            found = int(row[0])
        except (TypeError, ValueError):
            return

        if found == SCHEMA_VERSION:
            return

        backup = self.path.with_suffix(f".v{found}.bak")
        index = 1
        while backup.exists():
            backup = self.path.with_suffix(f".v{found}.{index}.bak")
            index += 1

        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                target = Path(str(backup) + suffix)
                candidate.rename(target)
        log.warning(
            "project database schema v%d is incompatible with v%d; moved it to "
            "%s and started a fresh index. Rendered artifacts and production "
            "reports on disk were not touched.",
            found, SCHEMA_VERSION, backup,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ProjectDB":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- low-level helpers -------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def _query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # -- projects ----------------------------------------------------------

    def create_project(
        self,
        name: str,
        objective: str = "",
        workspace: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        project_id = new_id("proj")
        now = _now()
        self._execute(
            "INSERT INTO projects(project_id, name, objective, workspace, metadata,"
            " created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
            (project_id, name, objective, workspace,
             json.dumps(metadata or {}), now, now),
        )
        # Deliberately no event here. Creation is announced by the caller
        # through the event bus, which both persists it and pushes it to live
        # subscribers; logging it here as well produced a duplicate row for
        # every project.
        return project_id

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM projects WHERE project_id=?", (project_id,))
        return self._project_row(row) if row else None

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM projects ORDER BY created_at DESC, rowid DESC")
        return [self._project_row(r) for r in rows]

    @staticmethod
    def _project_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        return data

    def update_project(self, project_id: str, **fields: Any) -> None:
        allowed = {"name", "objective", "status", "workspace"}
        sets, params = [], []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key}=?")
                params.append(value)
        if not sets:
            return
        sets.append("updated_at=?")
        params.extend([_now(), project_id])
        self._execute(f"UPDATE projects SET {', '.join(sets)} WHERE project_id=?", tuple(params))

    # -- tasks -------------------------------------------------------------

    def save_task(self, task: Task) -> None:
        self._execute(
            "INSERT OR REPLACE INTO tasks(task_id, project_id, parent_task_id, agent,"
            " objective, status, priority, retry_count, payload, updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                task.task_id, task.project_id, task.parent_task_id, task.agent,
                task.objective, task.status.value, task.priority, task.retry_count,
                json.dumps(task.to_dict()), task.updated_at,
            ),
        )

    def get_task(self, task_id: str) -> Task | None:
        row = self._query_one("SELECT payload FROM tasks WHERE task_id=?", (task_id,))
        return Task.from_dict(json.loads(row["payload"])) if row else None

    def list_tasks(
        self, project_id: str, status: TaskStatus | None = None
    ) -> list[Task]:
        if status is None:
            rows = self._query(
                "SELECT payload FROM tasks WHERE project_id=? ORDER BY priority, updated_at",
                (project_id,),
            )
        else:
            rows = self._query(
                "SELECT payload FROM tasks WHERE project_id=? AND status=?"
                " ORDER BY priority, updated_at",
                (project_id, status.value),
            )
        return [Task.from_dict(json.loads(r["payload"])) for r in rows]

    # -- scenes and shots --------------------------------------------------

    def upsert_scene(self, project_id: str, scene_id: str, ordinal: int,
                     title: str = "", spec: dict[str, Any] | None = None,
                     status: str = "PENDING") -> str:
        self._execute(
            "INSERT OR REPLACE INTO scenes(project_id, scene_id, ordinal, title, spec, status)"
            " VALUES(?,?,?,?,?,?)",
            (project_id, scene_id, ordinal, title, json.dumps(spec or {}), status),
        )
        return scene_id

    def list_scenes(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM scenes WHERE project_id=? ORDER BY ordinal", (project_id,)
        )
        out = []
        for row in rows:
            data = dict(row)
            data["spec"] = json.loads(data.get("spec") or "{}")
            out.append(data)
        return out

    def upsert_shot(self, project_id: str, shot_id: str, scene_id: str = "",
                    ordinal: int = 0, duration_s: float = 0.0,
                    spec: dict[str, Any] | None = None,
                    status: str = "PENDING") -> str:
        now = _now()
        existing = self._query_one(
            "SELECT shot_id FROM shots WHERE project_id=? AND shot_id=?",
            (project_id, shot_id),
        )
        if existing:
            self._execute(
                "UPDATE shots SET scene_id=?, ordinal=?, duration_s=?, spec=?, status=?,"
                " updated_at=? WHERE project_id=? AND shot_id=?",
                (scene_id, ordinal, duration_s, json.dumps(spec or {}), status, now,
                 project_id, shot_id),
            )
        else:
            self._execute(
                "INSERT INTO shots(project_id, shot_id, scene_id, ordinal, duration_s,"
                " spec, status, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (project_id, shot_id, scene_id, ordinal, duration_s,
                 json.dumps(spec or {}), status, now, now),
            )
        return shot_id

    def get_shot(self, project_id: str, shot_id: str) -> dict[str, Any] | None:
        row = self._query_one(
            "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
        )
        if not row:
            return None
        data = dict(row)
        data["spec"] = json.loads(data.get("spec") or "{}")
        return data

    def list_shots(self, project_id: str, scene_id: str | None = None) -> list[dict[str, Any]]:
        if scene_id is None:
            rows = self._query(
                "SELECT * FROM shots WHERE project_id=? ORDER BY scene_id, ordinal",
                (project_id,),
            )
        else:
            rows = self._query(
                "SELECT * FROM shots WHERE project_id=? AND scene_id=? ORDER BY ordinal",
                (project_id, scene_id),
            )
        out = []
        for row in rows:
            data = dict(row)
            data["spec"] = json.loads(data.get("spec") or "{}")
            out.append(data)
        return out

    def set_shot_status(self, project_id: str, shot_id: str, status: str) -> None:
        self._execute(
            "UPDATE shots SET status=?, updated_at=? WHERE project_id=? AND shot_id=?",
            (status, _now(), project_id, shot_id),
        )

    # -- versioning (spec section 16) -------------------------------------

    def next_version_no(self, project_id: str, shot_id: str) -> int:
        row = self._query_one(
            "SELECT COALESCE(MAX(version_no), 0) AS n FROM shot_versions"
            " WHERE project_id=? AND shot_id=?",
            (project_id, shot_id),
        )
        return int(row["n"]) + 1 if row else 1

    def add_shot_version(self, project_id: str, shot_id: str, *,
                         blend_path: str = "", render_path: str = "",
                         video_path: str = "", qa: dict[str, Any] | None = None,
                         notes: str = "") -> ShotVersion:
        version_no = self.next_version_no(project_id, shot_id)
        version_id = new_id("ver")
        created = _now()
        self._execute(
            "INSERT INTO shot_versions(version_id, project_id, shot_id, version_no,"
            " blend_path, render_path, video_path, approved, qa, notes, created_at)"
            " VALUES(?,?,?,?,?,?,?,0,?,?,?)",
            (version_id, project_id, shot_id, version_no, blend_path, render_path,
             video_path, json.dumps(qa or {}), notes, created),
        )
        return ShotVersion(version_id, shot_id, version_no, blend_path,
                           render_path, video_path, False, qa or {}, notes, created)

    def update_shot_version(self, project_id: str, shot_id: str, version_no: int,
                            **fields: Any) -> None:
        """Fill in a version row once its later artifacts exist.

        A round's version row is created as soon as its preview has been
        reviewed, and the final render and encode land on that same row
        afterwards. Creating a second row for the final render would leave the
        *approved* version pointing at the preview and the finished video
        recorded under a version number no directory corresponds to.
        """
        allowed = {"blend_path", "render_path", "video_path", "qa", "notes"}
        sets, params = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            sets.append(f"{key}=?")
            params.append(json.dumps(value) if key == "qa" else value)
        if not sets:
            return
        params.extend([project_id, shot_id, version_no])
        self._execute(
            f"UPDATE shot_versions SET {', '.join(sets)}"
            " WHERE project_id=? AND shot_id=? AND version_no=?",
            tuple(params),
        )

    def list_shot_versions(self, project_id: str, shot_id: str) -> list[ShotVersion]:
        rows = self._query(
            "SELECT * FROM shot_versions WHERE project_id=? AND shot_id=?"
            " ORDER BY version_no",
            (project_id, shot_id),
        )
        return [
            ShotVersion(
                version_id=r["version_id"], shot_id=r["shot_id"],
                version_no=r["version_no"], blend_path=r["blend_path"],
                render_path=r["render_path"], video_path=r["video_path"],
                approved=bool(r["approved"]), qa=json.loads(r["qa"] or "{}"),
                notes=r["notes"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def approve_shot_version(self, project_id: str, shot_id: str, version_no: int) -> None:
        """Mark a version approved without destroying any earlier one."""
        self._execute(
            "UPDATE shot_versions SET approved=1 WHERE project_id=? AND shot_id=?"
            " AND version_no=?",
            (project_id, shot_id, version_no),
        )
        self._execute(
            "UPDATE shots SET approved_version=?, status='APPROVED', updated_at=?"
            " WHERE project_id=? AND shot_id=?",
            (version_no, _now(), project_id, shot_id),
        )

    def latest_approved_version(self, project_id: str, shot_id: str) -> ShotVersion | None:
        """The version the Director reverts to when a revision goes wrong."""
        row = self._query_one(
            "SELECT * FROM shot_versions WHERE project_id=? AND shot_id=? AND approved=1"
            " ORDER BY version_no DESC LIMIT 1",
            (project_id, shot_id),
        )
        if not row:
            return None
        return ShotVersion(
            version_id=row["version_id"], shot_id=row["shot_id"],
            version_no=row["version_no"], blend_path=row["blend_path"],
            render_path=row["render_path"], video_path=row["video_path"],
            approved=True, qa=json.loads(row["qa"] or "{}"),
            notes=row["notes"], created_at=row["created_at"],
        )

    # -- assets (spec section 10) -----------------------------------------

    def upsert_asset(self, project_id: str, kind: str, name: str, path: str,
                     *, version: int = 1, canonical: bool = False,
                     preview: str = "", metadata: dict[str, Any] | None = None) -> str:
        now = _now()
        row = self._query_one(
            "SELECT asset_id FROM assets WHERE project_id=? AND kind=? AND name=? AND version=?",
            (project_id, kind, name, version),
        )
        if row:
            self._execute(
                "UPDATE assets SET path=?, canonical=?, preview=?, metadata=?, updated_at=?"
                " WHERE asset_id=?",
                (path, int(canonical), preview, json.dumps(metadata or {}), now,
                 row["asset_id"]),
            )
            return row["asset_id"]
        asset_id = new_id("asset")
        self._execute(
            "INSERT INTO assets(asset_id, project_id, kind, name, version, path,"
            " canonical, preview, metadata, created_at, updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (asset_id, project_id, kind, name, version, path, int(canonical),
             preview, json.dumps(metadata or {}), now, now),
        )
        return asset_id

    def find_assets(self, project_id: str, kind: str | None = None,
                    name: str | None = None,
                    canonical_only: bool = False) -> list[dict[str, Any]]:
        """Asset reuse lookup. Spec section 10: prefer reuse over recreation."""
        clauses = ["project_id=?"]
        params: list[Any] = [project_id]
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        if name:
            clauses.append("name=?")
            params.append(name)
        if canonical_only:
            clauses.append("canonical=1")
        rows = self._query(
            f"SELECT * FROM assets WHERE {' AND '.join(clauses)}"
            " ORDER BY version DESC",
            tuple(params),
        )
        out = []
        for row in rows:
            data = dict(row)
            data["canonical"] = bool(data["canonical"])
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            out.append(data)
        return out

    # -- characters (spec sections 7 and 11) ------------------------------

    def upsert_character(self, project_id: str, name: str,
                         canonical: dict[str, Any] | None = None,
                         continuity: dict[str, Any] | None = None,
                         asset_id: str = "") -> str:
        existing = self._query_one(
            "SELECT * FROM characters WHERE project_id=? AND name=?", (project_id, name)
        )
        now = _now()
        if existing:
            merged_canonical = json.loads(existing["canonical"] or "{}")
            merged_canonical.update(canonical or {})
            merged_continuity = json.loads(existing["continuity"] or "{}")
            merged_continuity.update(continuity or {})
            self._execute(
                "UPDATE characters SET canonical=?, continuity=?, asset_id=?, updated_at=?"
                " WHERE character_id=?",
                (json.dumps(merged_canonical), json.dumps(merged_continuity),
                 asset_id or existing["asset_id"], now, existing["character_id"]),
            )
            return existing["character_id"]
        character_id = new_id("char")
        self._execute(
            "INSERT INTO characters(character_id, project_id, name, canonical,"
            " continuity, asset_id, updated_at) VALUES(?,?,?,?,?,?,?)",
            (character_id, project_id, name, json.dumps(canonical or {}),
             json.dumps(continuity or {}), asset_id, now),
        )
        return character_id

    def get_character(self, project_id: str, name: str) -> dict[str, Any] | None:
        row = self._query_one(
            "SELECT * FROM characters WHERE project_id=? AND name=?", (project_id, name)
        )
        if not row:
            return None
        data = dict(row)
        data["canonical"] = json.loads(data.get("canonical") or "{}")
        data["continuity"] = json.loads(data.get("continuity") or "{}")
        return data

    def list_characters(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM characters WHERE project_id=? ORDER BY name", (project_id,)
        )
        out = []
        for row in rows:
            data = dict(row)
            data["canonical"] = json.loads(data.get("canonical") or "{}")
            data["continuity"] = json.loads(data.get("continuity") or "{}")
            out.append(data)
        return out

    # -- renders -----------------------------------------------------------

    def record_render(self, project_id: str, *, shot_id: str = "", version_no: int = 0,
                      kind: str = "preview", path: str = "", status: str = "COMPLETED",
                      engine: str = "", metadata: dict[str, Any] | None = None) -> str:
        render_id = new_id("render")
        self._execute(
            "INSERT INTO renders(render_id, project_id, shot_id, version_no, kind, path,"
            " status, engine, metadata, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (render_id, project_id, shot_id, version_no, kind, path, status, engine,
             json.dumps(metadata or {}), _now()),
        )
        return render_id

    def list_renders(self, project_id: str, shot_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM renders WHERE project_id=? AND shot_id=?"
            " ORDER BY created_at, rowid",
            (project_id, shot_id),
        )
        out = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data.get("metadata") or "{}")
            out.append(data)
        return out

    # -- QA ----------------------------------------------------------------

    def record_qa(self, project_id: str, subject: str, subject_id: str, scope: str,
                  status: QAStatus | str, score: float | None = None,
                  findings: list[dict[str, Any]] | None = None) -> str:
        qa_id = new_id("qa")
        status_value = status.value if isinstance(status, QAStatus) else str(status)
        self._execute(
            "INSERT INTO qa_results(qa_id, project_id, subject, subject_id, scope,"
            " status, score, findings, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (qa_id, project_id, subject, subject_id, scope, status_value, score,
             json.dumps(findings or []), _now()),
        )
        return qa_id

    def latest_qa(self, project_id: str, subject_id: str,
                  scope: str | None = None) -> dict[str, Any] | None:
        # created_at only has second resolution, so several QA passes in one
        # second would otherwise tie. rowid is monotonic and breaks the tie in
        # true insertion order.
        if scope:
            row = self._query_one(
                "SELECT * FROM qa_results WHERE project_id=? AND subject_id=?"
                " AND scope=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (project_id, subject_id, scope),
            )
        else:
            row = self._query_one(
                "SELECT * FROM qa_results WHERE project_id=? AND subject_id=?"
                " ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (project_id, subject_id),
            )
        if not row:
            return None
        data = dict(row)
        data["findings"] = json.loads(data.get("findings") or "[]")
        return data

    # -- owner decisions (spec sections 17 and 18) ------------------------

    def request_decision(self, project_id: str, question: str,
                         options: list[str] | None = None, *, task_id: str = "",
                         urgency: str = "normal",
                         context: dict[str, Any] | None = None) -> str:
        """Queue something that genuinely needs human authority."""
        decision_id = new_id("dec")
        self._execute(
            "INSERT INTO decisions(decision_id, project_id, task_id, urgency, question,"
            " options, context, status, created_at) VALUES(?,?,?,?,?,?,?, 'PENDING', ?)",
            (decision_id, project_id, task_id, urgency, question,
             json.dumps(options or []), json.dumps(context or {}), _now()),
        )
        self.log_event(project_id, "decision_requested", question,
                       {"decision_id": decision_id, "urgency": urgency})
        return decision_id

    def pending_decisions(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM decisions WHERE project_id=? AND status='PENDING'"
            " ORDER BY created_at",
            (project_id,),
        )
        out = []
        for row in rows:
            data = dict(row)
            data["options"] = json.loads(data.get("options") or "[]")
            data["context"] = json.loads(data.get("context") or "{}")
            out.append(data)
        return out

    def resolve_decision(self, decision_id: str, answer: str) -> None:
        self._execute(
            "UPDATE decisions SET status='RESOLVED', answer=?, resolved_at=?"
            " WHERE decision_id=?",
            (answer, _now(), decision_id),
        )

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self._query_one("SELECT * FROM decisions WHERE decision_id=?", (decision_id,))
        if not row:
            return None
        data = dict(row)
        data["options"] = json.loads(data.get("options") or "[]")
        data["context"] = json.loads(data.get("context") or "{}")
        return data

    # -- event log ---------------------------------------------------------

    def log_event(self, project_id: str, kind: str, message: str,
                  payload: dict[str, Any] | None = None, task_id: str = "",
                  *, agent: str = "", scene_id: str = "",
                  shot_id: str = "") -> int:
        """Append to the production event log and return its sequence number.

        The sequence is monotonic, which is what lets a live subscriber resume
        from a cursor instead of missing events that fired while it was
        reconnecting.
        """
        event_id = new_id("evt")
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO events(event_id, project_id, task_id, agent, scene_id,"
                " shot_id, kind, message, payload, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (event_id, project_id, task_id, agent, scene_id, shot_id, kind,
                 message, json.dumps(payload or {}), _now()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def list_events(self, project_id: str, limit: int = 200,
                    *, since_seq: int = 0) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM events WHERE project_id=? AND seq>?"
            " ORDER BY seq DESC LIMIT ?",
            (project_id, since_seq, limit),
        )
        out = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.get("payload") or "{}")
            out.append(data)
        return out

    def events_after(self, project_id: str, since_seq: int,
                     limit: int = 500) -> list[dict[str, Any]]:
        """Events in ascending order after a cursor — the live-feed path."""
        rows = self._query(
            "SELECT * FROM events WHERE project_id=? AND seq>?"
            " ORDER BY seq ASC LIMIT ?",
            (project_id, since_seq, limit),
        )
        out = []
        for row in rows:
            data = dict(row)
            data["payload"] = json.loads(data.get("payload") or "{}")
            out.append(data)
        return out

    def latest_event_seq(self, project_id: str) -> int:
        row = self._query_one(
            "SELECT COALESCE(MAX(seq), 0) AS n FROM events WHERE project_id=?",
            (project_id,),
        )
        return int(row["n"]) if row else 0

    # -- artifacts (spec section 9) ---------------------------------------

    def register_artifact(self, project_id: str, kind: str, path: str,
                          label: str = "", *, shot_id: str = "",
                          scene_id: str = "",
                          metadata: dict[str, Any] | None = None) -> str:
        """Record a produced file so the owner never has to go looking for it.

        Sizes are captured at registration time; a missing file is recorded as
        zero bytes rather than raising, because an artifact row that exists and
        points at nothing is more useful than an exception mid-production.
        """
        artifact_id = new_id("art")
        candidate = Path(path)
        size = candidate.stat().st_size if candidate.is_file() else 0
        self._execute(
            "INSERT INTO artifacts(artifact_id, project_id, kind, label, path,"
            " shot_id, scene_id, bytes, metadata, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (artifact_id, project_id, kind, label, str(path), shot_id, scene_id,
             size, json.dumps(metadata or {}), _now()),
        )
        return artifact_id

    @staticmethod
    def _artifact_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.get("metadata") or "{}")
        # An artifact may be a directory — a rendered frame sequence is one.
        # Testing is_file() alone reported every directory artifact as missing,
        # which is both wrong and exactly the kind of false alarm that makes a
        # real one easy to miss.
        path = Path(data["path"])
        if path.is_file():
            data["exists"] = True
            data["artifact_type"] = "file"
        elif path.is_dir():
            data["exists"] = True
            data["artifact_type"] = "directory"
        else:
            data["exists"] = False
            data["artifact_type"] = "missing"
        return data

    def list_artifacts(self, project_id: str, kind: str | None = None,
                       limit: int = 500) -> list[dict[str, Any]]:
        if kind:
            rows = self._query(
                "SELECT * FROM artifacts WHERE project_id=? AND kind=?"
                " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (project_id, kind, limit),
            )
        else:
            rows = self._query(
                "SELECT * FROM artifacts WHERE project_id=?"
                " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (project_id, limit),
            )
        return [self._artifact_row(r) for r in rows]

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._query_one(
            "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
        )
        return self._artifact_row(row) if row else None

    def latest_artifact(self, project_id: str, kind: str,
                        shot_id: str | None = None) -> dict[str, Any] | None:
        if shot_id:
            row = self._query_one(
                "SELECT * FROM artifacts WHERE project_id=? AND kind=? AND shot_id=?"
                " ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (project_id, kind, shot_id),
            )
        else:
            row = self._query_one(
                "SELECT * FROM artifacts WHERE project_id=? AND kind=?"
                " ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (project_id, kind),
            )
        return self._artifact_row(row) if row else None

    # -- live production state (cross-process) -----------------------------

    def upsert_run(self, project_id: str, **fields: Any) -> None:
        """Record the state of a production run so other processes can see it."""
        allowed = {"state", "objective", "stage", "current_shot", "shots_total",
                   "shots_done", "percent", "error", "pid", "started_at",
                   "finished_at"}
        now = _now()
        existing = self._query_one(
            "SELECT project_id FROM production_runs WHERE project_id=?",
            (project_id,),
        )
        if existing:
            sets = [f"{k}=?" for k in fields if k in allowed]
            params = [v for k, v in fields.items() if k in allowed]
            sets.append("updated_at=?")
            params.extend([now, project_id])
            self._execute(
                f"UPDATE production_runs SET {', '.join(sets)} WHERE project_id=?",
                tuple(params),
            )
            return

        values = {k: v for k, v in fields.items() if k in allowed}
        values.setdefault("state", "IDLE")
        values.setdefault("started_at", now)
        columns = list(values) + ["project_id", "updated_at"]
        placeholders = ",".join("?" for _ in columns)
        self._execute(
            f"INSERT INTO production_runs({', '.join(columns)})"
            f" VALUES({placeholders})",
            tuple(list(values.values()) + [project_id, now]),
        )

    def get_run(self, project_id: str) -> dict[str, Any] | None:
        row = self._query_one(
            "SELECT * FROM production_runs WHERE project_id=?", (project_id,)
        )
        return dict(row) if row else None

    def latest_run(self) -> dict[str, Any] | None:
        row = self._query_one(
            "SELECT * FROM production_runs ORDER BY updated_at DESC, rowid DESC LIMIT 1"
        )
        return dict(row) if row else None

    def request_control(self, project_id: str, action: str) -> int:
        """Ask whichever process is running this production to pause/stop."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO control_requests(project_id, action, created_at)"
                " VALUES(?,?,?)",
                (project_id, action, _now()),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def pending_controls(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM control_requests WHERE project_id=? AND status='PENDING'"
            " ORDER BY seq",
            (project_id,),
        )
        return [dict(r) for r in rows]

    def consume_control(self, seq: int) -> None:
        self._execute(
            "UPDATE control_requests SET status='CONSUMED', consumed_at=?"
            " WHERE seq=?",
            (_now(), seq),
        )

    # -- progress ----------------------------------------------------------

    def progress(self, project_id: str) -> dict[str, Any]:
        """Roll-up the owner interface renders as the production dashboard."""
        shots = self.list_shots(project_id)
        tasks = self.list_tasks(project_id)
        by_status: dict[str, int] = {}
        for task in tasks:
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1

        approved = sum(1 for s in shots if s["status"] == "APPROVED")
        total = len(shots) or 0
        return {
            "shots_total": total,
            "shots_approved": approved,
            "shots_percent": round(100.0 * approved / total, 1) if total else 0.0,
            "tasks_total": len(tasks),
            "tasks_by_status": by_status,
            "pending_decisions": len(self.pending_decisions(project_id)),
            "characters": len(self.list_characters(project_id)),
            "assets": len(self.find_assets(project_id)),
        }

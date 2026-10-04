"""Live production dashboard (spec sections 10, 12, 13).

A local web view of what the production engine is actually doing. Every value
on the page is read from the same SQLite database and event bus the agents
write to, and the preview image is served from the artifact registry — the
identical file the Vision Agent judged. There is no mock data and no simulated
activity anywhere in this module.

Live updates use Server-Sent Events, so the preview and the activity feed
change without the browser polling or the owner refreshing.

Binds to loopback only. Nothing here is reachable from another machine.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .config import AppConfig, load_config
from .core.events import EventBus, EventKind
from .core.storage import StorageGovernor
from .runtime import ProductionRunner, RunState

log = logging.getLogger(__name__)

#: How long an SSE connection waits for an event before sending a keep-alive.
SSE_HEARTBEAT_S = 20.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed(started_at: str, finished_at: str = "") -> str:
    """Human elapsed time between two ISO timestamps."""
    if not started_at:
        return ""
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(timezone.utc)
    except ValueError:
        return ""
    seconds = max(0.0, (end - start).total_seconds())
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class DashboardState:
    """Reads the live state the page renders."""

    def __init__(self, config: AppConfig, runner: ProductionRunner,
                 events: EventBus, db_factory) -> None:
        self.config = config
        self.runner = runner
        self.events = events
        self._db_factory = db_factory

    @property
    def db(self):
        return self._db_factory()

    def _active_project_id(self, requested: str = "") -> str:
        if requested:
            return requested
        run = self.runner.run
        if run and run.project_id:
            return run.project_id
        # The production may be running in another process (the MCP server the
        # AI client spawned), in which case the only record of it is in the
        # database.
        try:
            latest = self.db.latest_run()
            if latest and latest.get("state") in {"RUNNING", "PAUSED", "STOPPING"}:
                return latest["project_id"]
        except Exception:  # noqa: BLE001
            pass
        projects = self.db.list_projects()
        return projects[0]["project_id"] if projects else ""

    def run_state(self, project_id: str) -> dict[str, Any]:
        """The authoritative run state, from whichever process owns it.

        Delegates to the runner, which reconciles a stored COMPLETED against the
        filesystem. The dashboard and the MCP server are separate processes, so
        a run started by an AI client is invisible to this process's runner —
        reading and verifying the persisted row is what stops the dashboard
        reporting success for a film that is no longer on disk.
        """
        state = self.runner.status(project_id)
        if state.get("state") == "IDLE" and not state.get("note"):
            in_process = self.runner.status()
            if in_process.get("active"):
                return in_process
        state.setdefault("managed_elsewhere", True)
        return state

    def snapshot(self, project_id: str = "") -> dict[str, Any]:
        """Everything the dashboard needs in one response."""
        db = self.db
        pid = self._active_project_id(project_id)
        run_status = self.run_state(pid) if pid else self.runner.status()

        if not pid:
            return {
                "server": {"version": __version__, "time": _now_iso()},
                "project": None,
                "run": run_status,
                "projects": [],
                "note": "No projects yet. Start one from the CLI or an MCP client.",
            }

        project = db.get_project(pid)
        if project is None:
            return {"project": None, "run": run_status, "projects": [],
                    "note": f"unknown project {pid}"}

        progress = db.progress(pid)
        shots = db.list_shots(pid)
        tasks = db.list_tasks(pid)
        events = db.list_events(pid, limit=60)
        projects = db.list_projects()

        latest_preview = db.latest_artifact(pid, "preview")
        latest_final = db.latest_artifact(pid, "final_movie")
        pending = db.pending_decisions(pid)

        # The dashboard must only advertise artifacts that still exist on disk.
        # A preview the storage governor reclaimed under pressure is a registry
        # row with no file behind it; serving it would 410.
        preview_present = bool(
            latest_preview and latest_preview.get("exists")
            and Path(latest_preview["path"]).is_file()
        )
        final_present = bool(
            latest_final and latest_final.get("exists")
            and Path(latest_final["path"]).is_file()
        )

        by_status: dict[str, int] = {}
        for task in tasks:
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1

        return {
            "server": {"version": __version__, "time": _now_iso()},
            "project": {
                "project_id": pid,
                "name": project["name"],
                "objective": project["objective"],
                "status": project["status"],
                "created_at": project["created_at"],
                "elapsed": _elapsed(project["created_at"]),
            },
            "run": {
                **run_status,
                "elapsed": _elapsed(run_status.get("started_at", ""),
                                    run_status.get("finished_at", "")),
            },
            "projects": [
                {"project_id": p["project_id"], "name": p["name"],
                 "status": p["status"]}
                for p in projects
            ],
            "current_activity": self.events.current_activity(pid),
            "progress": progress,
            "shots": [
                {"shot_id": s["shot_id"], "status": s["status"],
                 "duration_s": s["duration_s"],
                 "description": (s.get("spec") or {}).get("description", "")}
                for s in shots
            ],
            "tasks": {
                "by_status": by_status,
                "total": len(tasks),
                "recent": [
                    {"task_id": t.task_id, "agent": t.agent,
                     "objective": t.objective, "status": t.status.value,
                     "updated_at": t.updated_at}
                    for t in sorted(tasks, key=lambda t: t.updated_at, reverse=True)[:12]
                ],
            },
            "events": [
                {"seq": e["seq"], "kind": e["kind"], "agent": e["agent"],
                 "message": e["message"], "shot_id": e["shot_id"],
                 "created_at": e["created_at"]}
                for e in events
            ],
            "preview": {
                "artifact_id": latest_preview["artifact_id"],
                "shot_id": latest_preview["shot_id"],
                "created_at": latest_preview["created_at"],
            } if preview_present else None,
            "final_movie": {
                "artifact_id": latest_final["artifact_id"],
                "path": latest_final["path"],
                "bytes": latest_final["bytes"],
                "created_at": latest_final["created_at"],
            } if final_present else None,
            # The same storage measurement the MCP tool reports, so the
            # dashboard can never show a different disk story than the AI sees.
            "storage": self.storage_state(),
            "pending_decisions": pending,
        }

    def storage_state(self) -> dict[str, Any]:
        """Live working-storage measurement for the dashboard.

        Measured, never estimated: the governor walks the workspace and applies
        the same safe-cleanup rule production uses.
        """
        try:
            usage = StorageGovernor(
                self.config.workspace,
                thresholds=self.config.storage.thresholds(),
            ).measure()
        except Exception as exc:  # noqa: BLE001 - the dashboard must still render
            return {"state": "UNKNOWN", "error": str(exc)}
        return usage.to_dict()


class DashboardServer:
    """HTTP server exposing the dashboard and its live event stream."""

    def __init__(self, config: AppConfig | None = None,
                 runner: ProductionRunner | None = None,
                 events: EventBus | None = None) -> None:
        self.config = config or load_config()
        self.events = events or EventBus()
        self.runner = runner or ProductionRunner(self.config, self.events)
        self._db = None
        self._lock = threading.RLock()
        self.state = DashboardState(
            self.config, self.runner, self.events, self._db_handle
        )
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _db_handle(self):
        with self._lock:
            if self._db is None:
                from .core.project import ProjectDB

                self._db = ProjectDB(Path(self.config.workspace) / "registry.db")
                self.events.bind_db(self._db)
            return self._db

    # -- lifecycle ---------------------------------------------------------

    def start(self, host: str = "127.0.0.1", port: int | None = None) -> int:
        """Start serving and return the bound port."""
        port = self.config.dashboard_port if port is None else port
        handler = make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        bound = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="filmautomator-dashboard",
            daemon=True,
        )
        self._thread.start()
        log.info("dashboard on http://%s:%d", host, bound)
        return bound

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:  # noqa: BLE001
                    pass
                self._db = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1] if self._httpd else 0

    # -- request handling --------------------------------------------------

    def handle_api(self, path: str, query: dict[str, list[str]],
                   body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        """Route an API call. Returns (status, payload)."""
        if path == "/api/state":
            project_id = (query.get("project_id") or [""])[0]
            return 200, self.state.snapshot(project_id)

        if path == "/api/events":
            project_id = (query.get("project_id") or [""])[0]
            since = int((query.get("since") or ["0"])[0] or 0)
            pid = self.state._active_project_id(project_id)  # noqa: SLF001
            events = self._db_handle().events_after(pid, since, 500)
            return 200, {"events": events, "count": len(events)}

        if path.startswith("/api/control/"):
            return self._control(path.rsplit("/", 1)[-1], body or {})

        return 404, {"error": f"unknown endpoint: {path}"}

    def _control(self, action: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Owner controls.

        When this process owns the run, the action is applied directly. When
        the production is running somewhere else — the usual case, since an AI
        client spawns the MCP server as its own process — the request is filed
        in the database and picked up at the next safe point.
        """
        project_id = (body.get("project_id") or "").strip()
        owned = bool(self.runner.run and self.runner.is_busy)

        if action == "stop" and not body.get("confirm"):
            # Stopping abandons in-flight work, so it asks first.
            return 400, {"ok": False, "error": "stop requires confirm=true"}

        if action in {"pause", "resume", "stop"}:
            if owned:
                applied = {
                    "pause": self.runner.pause,
                    "resume": self.runner.resume,
                    "stop": self.runner.stop,
                }[action]()
                return 200, {"ok": applied, **self.runner.status()}

            target = project_id or self.state._active_project_id()  # noqa: SLF001
            if not target:
                return 400, {"ok": False, "error": "no project to control"}

            stored = self._db_handle().get_run(target)
            if not stored or stored.get("state") not in {"RUNNING", "PAUSED", "STOPPING"}:
                return 200, {"ok": False,
                             "state": (stored or {}).get("state", "IDLE"),
                             "note": "nothing is running for that project"}

            request_id = self._db_handle().request_control(target, action)
            self.events.emit(
                target, EventKind.PRODUCTION_PAUSED if action == "pause"
                else EventKind.PRODUCTION_RESUMED if action == "resume"
                else EventKind.PRODUCTION_STOPPED,
                f"{action} requested from the dashboard", agent="owner",
                payload={"action": action, "request_id": request_id},
            )
            return 200, {
                "ok": True,
                "queued": True,
                "request_id": request_id,
                "note": "The production picks this up at its next safe point "
                        "(between shots or revision rounds, never mid-render).",
                **self.state.run_state(target),
            }

        if action == "retry":
            if self.runner.is_busy:
                return 409, {"ok": False, "error": "a production is already running"}
            target = project_id or self.state._active_project_id()  # noqa: SLF001
            if not target:
                return 400, {"ok": False, "error": "project_id is required"}
            from .producer import ProductionRequest

            request = ProductionRequest(objective="", project_id=target)
            try:
                run = self.runner.start(request, project_id=target)
            except RuntimeError as exc:
                return 409, {"ok": False, "error": str(exc)}
            return 200, {"ok": True, "state": run.state.value}

        return 404, {"error": f"unknown control: {action}"}


def make_handler(dashboard: DashboardServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"Filmautomator/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            log.debug("dashboard: " + fmt, *args)

        # -- helpers --

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Previews change as production revises a shot; never cache them.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # -- routes --

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._send_bytes(INDEX_HTML.encode("utf-8"),
                                 "text/html; charset=utf-8")
                return
            if path == "/events":
                self._serve_events(query)
                return
            if path.startswith("/artifact/"):
                self._serve_artifact(path.rsplit("/", 1)[-1])
                return
            if path.startswith("/api/"):
                status, payload = dashboard.handle_api(path, query, None)
                self._send_json(payload, status)
                return
            self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            status, payload = dashboard.handle_api(
                parsed.path, parse_qs(parsed.query), body
            )
            self._send_json(payload, status)

        # -- streaming --

        def _serve_events(self, query: dict[str, list[str]]) -> None:
            """Server-Sent Events: the live production feed."""
            project_id = (query.get("project_id") or [""])[0]
            pid = dashboard.state._active_project_id(project_id)  # noqa: SLF001

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            queue = dashboard.events.subscribe()
            try:
                # Replay what the client missed, so a browser that reloads
                # mid-production is not blank until the next event fires.
                for event in dashboard.events.history(pid, limit=40):
                    self._write_event(event.to_dict())
                self.wfile.flush()

                while True:
                    try:
                        event = queue.get(timeout=SSE_HEARTBEAT_S)
                    except Exception:  # noqa: BLE001 - queue.Empty and friends
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    if pid and event.project_id != pid:
                        continue
                    self._write_event(event.to_dict())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # browser navigated away
            finally:
                dashboard.events.unsubscribe(queue)

        def _write_event(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, default=str)
            self.wfile.write(f"event: production\ndata: {data}\n\n".encode("utf-8"))

        # -- artifact files --

        def _serve_artifact(self, artifact_id: str) -> None:
            """Serve a registered artifact by id.

            Only files the database knows about are reachable. There is no
            path parameter, so a request cannot walk outside the workspace.
            """
            record = dashboard._db_handle().get_artifact(artifact_id)  # noqa: SLF001
            if record is None:
                self._send_json({"error": "no such artifact"}, 404)
                return
            path = Path(record["path"])
            if not path.is_file():
                self._send_json({"error": "artifact file is missing"}, 410)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), content_type)

    return Handler


def serve_dashboard(config: AppConfig | None = None, *,
                    host: str = "127.0.0.1", port: int | None = None,
                    open_browser: bool = False) -> DashboardServer:
    """Start the dashboard and return the running server."""
    dashboard = DashboardServer(config)
    dashboard.start(host, port)
    if open_browser:
        import webbrowser

        webbrowser.open(f"http://{host}:{dashboard.port}")
    return dashboard


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Filmautomator — live production</title>
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --panel-2:#1c2230; --line:#2a3140;
    --fg:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950;
    --warn:#d29922; --bad:#f85149; --run:#58a6ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { padding:16px 22px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; letter-spacing:.14em; text-transform:uppercase; }
  h2 { font-size:11px; margin:0 0 10px; color:var(--muted);
    letter-spacing:.12em; text-transform:uppercase; font-weight:600; }
  .sub { color:var(--muted); font-size:12px; }
  .grow { flex:1; }
  .badge { padding:2px 10px; border-radius:99px; font-size:11px;
    border:1px solid var(--line); letter-spacing:.06em; }
  .badge.RUNNING{color:var(--run);border-color:var(--run)}
  .badge.PAUSED{color:var(--warn);border-color:var(--warn)}
  .badge.COMPLETED{color:var(--ok);border-color:var(--ok)}
  .badge.FAILED,.badge.STOPPED{color:var(--bad);border-color:var(--bad)}
  .badge.IDLE{color:var(--muted)}
  main { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);
    gap:16px; padding:16px 22px 40px; align-items:start; }
  @media (max-width:1000px){ main{grid-template-columns:1fr} }
  .panel { background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px 16px; margin-bottom:16px; }
  .stage { background:#000; border:1px solid var(--line); border-radius:8px;
    overflow:hidden; position:relative; min-height:220px;
    display:flex; align-items:center; justify-content:center; }
  .stage img { width:100%; display:block; }
  .stage .empty { color:var(--muted); font-size:12px; padding:60px 20px;
    text-align:center; }
  .stage .tag { position:absolute; top:8px; left:10px; font-size:11px;
    color:var(--muted); background:rgba(0,0,0,.65); padding:2px 8px;
    border-radius:4px; }
  .bar { height:6px; background:var(--panel-2); border-radius:99px;
    overflow:hidden; margin:8px 0 4px; }
  .bar > i { display:block; height:100%; background:var(--accent);
    width:0%; transition:width .4s ease; }
  .row { display:flex; justify-content:space-between; gap:12px;
    padding:4px 0; font-size:12px; }
  .row span:first-child { color:var(--muted); }
  .feed { max-height:340px; overflow-y:auto; font-size:12px; }
  .feed div { padding:3px 0; border-bottom:1px solid rgba(42,49,64,.5);
    display:flex; gap:10px; }
  .feed time { color:var(--muted); flex:0 0 58px; }
  .feed .agent { color:var(--accent); flex:0 0 108px; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }
  .feed .msg { flex:1; }
  .counts { display:flex; gap:8px; flex-wrap:wrap; }
  .count { background:var(--panel-2); border-radius:6px; padding:6px 10px;
    font-size:11px; min-width:74px; }
  .count b { display:block; font-size:16px; font-weight:600; }
  button { background:var(--panel-2); color:var(--fg);
    border:1px solid var(--line); border-radius:6px; padding:7px 14px;
    font:inherit; font-size:12px; cursor:pointer; }
  button:hover:not(:disabled){ border-color:var(--accent); color:var(--accent); }
  button:disabled { opacity:.4; cursor:not-allowed; }
  button.danger:hover:not(:disabled){ border-color:var(--bad); color:var(--bad); }
  .controls { display:flex; gap:8px; flex-wrap:wrap; }
  ul { list-style:none; margin:0; padding:0; font-size:12px; }
  li { padding:4px 0; border-bottom:1px solid rgba(42,49,64,.5);
    display:flex; justify-content:space-between; gap:10px; }
  li .muted { color:var(--muted); }
  a { color:var(--accent); }
  .flash { animation:flash 1s ease; }
  @keyframes flash { from { background:rgba(88,166,255,.18); } to {} }
</style>
</head>
<body>
<header>
  <h1>Filmautomator</h1>
  <span id="pname" class="sub">—</span>
  <span id="state" class="badge IDLE">IDLE</span>
  <span class="grow"></span>
  <span id="clock" class="sub"></span>
</header>

<main>
  <section>
    <div class="panel">
      <h2>Current activity</h2>
      <div class="row"><span>agent</span><b id="c-agent">—</b></div>
      <div class="row"><span>task</span><b id="c-task">—</b></div>
      <div class="row"><span>shot</span><b id="c-shot">—</b></div>
      <div class="row"><span>stage</span><b id="c-stage">—</b></div>
      <div class="row"><span>elapsed</span><b id="c-elapsed">—</b></div>
    </div>

    <div class="panel">
      <h2>Latest Blender preview <span id="pv-shot" class="sub"></span></h2>
      <div class="stage" id="stage">
        <div class="empty" id="stage-empty">
          No preview yet. One appears the moment Blender renders a frame.
        </div>
      </div>
      <div class="sub" style="margin-top:8px" id="pv-note">
        This is the same frame the Vision Agent judged.
      </div>
    </div>

    <div class="panel">
      <h2>Working storage <span id="st-state" class="sub"></span></h2>
      <div class="bar"><div class="fill" id="st-fill"></div></div>
      <div class="sub" id="st-detail" style="margin-top:8px"></div>
      <ul id="st-cats" style="margin-top:8px"></ul>
    </div>

    <div class="panel">
      <h2>Artifacts</h2>
      <ul id="artifacts"><li class="muted">none yet</li></ul>
    </div>
  </section>

  <section>
    <div class="panel">
      <h2>Progress</h2>
      <div class="bar"><i id="pbar"></i></div>
      <div class="row"><span id="pct">0%</span><span id="pshots">0 / 0 shots</span></div>
      <div class="counts" id="counts" style="margin-top:10px"></div>
    </div>

    <div class="panel">
      <h2>Controls</h2>
      <div class="controls">
        <button id="btn-pause">PAUSE</button>
        <button id="btn-resume">RESUME</button>
        <button id="btn-stop" class="danger">STOP</button>
        <button id="btn-retry">RETRY</button>
      </div>
      <div class="sub" style="margin-top:8px" id="ctl-note"></div>
    </div>

    <div class="panel">
      <h2>Shots</h2>
      <ul id="shots"><li class="muted">none yet</li></ul>
    </div>

    <div class="panel">
      <h2>Agent activity</h2>
      <div class="feed" id="feed"><div class="muted">waiting…</div></div>
    </div>

    <div class="panel" id="decisions-panel" style="display:none">
      <h2>Needs your decision</h2>
      <ul id="decisions"></ul>
    </div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);

function esc(s){ return String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

async function post(path, body){
  const r = await fetch(path, {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body||{})});
  return r.json();
}

function setControls(run){
  const active = run && run.active;
  const paused = run && run.state === "PAUSED";
  $("btn-pause").disabled  = !active || paused;
  $("btn-resume").disabled = !paused;
  $("btn-stop").disabled   = !active;
  $("btn-retry").disabled  = !!active;
}

function render(s){
  if (!s) return;
  const proj = s.project;
  const run  = s.run || {};

  $("pname").textContent = proj ? proj.name : (s.note || "no project");
  const badge = $("state");
  badge.textContent = run.state || "IDLE";
  badge.className = "badge " + (run.state || "IDLE");

  const act = s.current_activity || {};
  $("c-agent").textContent = act.agent || "—";
  $("c-task").textContent  = act.message || "—";
  $("c-shot").textContent  = act.shot_id || run.current_shot || "—";
  $("c-stage").textContent = run.current_stage || "—";
  $("c-elapsed").textContent = run.elapsed || (proj ? proj.elapsed : "—");

  // When the production runs in another process (the MCP server an AI client
  // spawned), controls travel through storage and take effect at the next safe
  // point. Say that plainly instead of looking broken.
  $("ctl-note").dataset.remote = run.managed_elsewhere ? "1" : "";
  if (run.managed_elsewhere && run.active) {
    $("c-agent").title = "Controlled from another process; requests are applied "
                       + "at the next safe point.";
  }

  const p = s.progress || {shots_total:0, shots_approved:0};
  const total = p.shots_total || 0;
  const done  = p.shots_approved || 0;
  const pct   = total ? Math.round(100*done/total) : (run.percent || 0);
  $("pbar").style.width = pct + "%";
  $("pct").textContent = pct + "%";
  $("pshots").textContent = done + " / " + total + " shots approved";

  const counts = (s.tasks && s.tasks.by_status) || {};
  $("counts").innerHTML = Object.keys(counts).length
    ? Object.entries(counts).map(([k,v]) =>
        `<div class="count">${esc(k)}<b>${v}</b></div>`).join("")
    : '<div class="count muted">no tasks<b>0</b></div>';

  // Preview: only touch the <img> when the artifact changes, so it does not
  // flicker on every state refresh.
  const pv = s.preview;
  const stage = $("stage");
  if (pv) {
    let img = stage.querySelector("img");
    const src = "/artifact/" + pv.artifact_id;
    if (!img) {
      stage.innerHTML = '<div class="tag"></div>';
      img = document.createElement("img");
      stage.appendChild(img);
    }
    if (img.dataset.artifact !== pv.artifact_id) {
      img.dataset.artifact = pv.artifact_id;
      img.src = src;
      stage.querySelector(".tag").textContent =
        pv.shot_id + " · " + pv.created_at;
      stage.classList.add("flash");
      setTimeout(() => stage.classList.remove("flash"), 1000);
    }
    $("pv-shot").textContent = pv.shot_id || "";
  }

  // Working storage — the same measurement the MCP tools report.
  const st = s.storage;
  if (st) {
    $("st-state").textContent = st.state || "";
    const pct = Math.max(0, Math.min(100, st.percent_of_limit || 0));
    const fill = $("st-fill");
    fill.style.width = pct + "%";
    fill.style.background =
      st.state === "HARD_LIMIT" ? "#e5484d"
      : st.state === "AGGRESSIVE_CLEANUP" ? "#f5a524"
      : st.state === "WARNING" ? "#f5d90a" : "#30a46c";
    $("st-detail").textContent =
      (st.total_gb || 0).toFixed(2) + " GB of " +
      ((st.thresholds && st.thresholds.hard_limit_gb) || 35) + " GB (" +
      pct + "%) · reclaimable now " + (st.disposable_gb || 0).toFixed(2) +
      " GB · protected " + ((st.protected_bytes || 0) / 1073741824).toFixed(2) +
      " GB";
    const cats = Object.entries(st.by_category || {})
      .filter(([, v]) => v.bytes > 0)
      .sort((a, b) => b[1].bytes - a[1].bytes);
    $("st-cats").innerHTML = cats.length
      ? cats.map(([k, v]) =>
          `<li><span>${esc(v.label || k)}</span>` +
          `<span class="muted">${(v.gb || 0).toFixed(3)} GB</span></li>`).join("")
      : '<li class="muted">nothing on disk yet</li>';
  }

  // Artifacts
  const arts = [];
  if (s.preview) arts.push(["preview", s.preview.shot_id, "/artifact/" + s.preview.artifact_id]);
  if (s.final_movie) arts.push(["final movie", (s.final_movie.bytes/1e6).toFixed(2)+" MB",
                                "/artifact/" + s.final_movie.artifact_id]);
  $("artifacts").innerHTML = arts.length
    ? arts.map(([k,v,href]) =>
        `<li><span>${esc(k)} <span class="muted">${esc(v)}</span></span>` +
        `<a href="${href}" target="_blank">open</a></li>`).join("")
    : '<li class="muted">none yet</li>';

  // Shots
  const shots = s.shots || [];
  $("shots").innerHTML = shots.length
    ? shots.map(sh => `<li><span>${esc(sh.shot_id)}</span>` +
        `<span class="muted">${esc(sh.status)} · ${sh.duration_s}s</span></li>`).join("")
    : '<li class="muted">none yet</li>';

  // Agent feed
  const evts = (s.events || []).slice(0, 60);
  const feed = $("feed");
  const atBottom = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 24;
  feed.innerHTML = evts.length
    ? evts.map(e => `<div><time>${esc((e.created_at||"").slice(11,19))}</time>` +
        `<span class="agent">${esc(e.agent || e.kind)}</span>` +
        `<span class="msg">${esc(e.message)}</span></div>`).join("")
    : '<div class="muted">no activity yet</div>';
  if (atBottom) feed.scrollTop = feed.scrollHeight;

  // Decisions
  const dec = s.pending_decisions || [];
  $("decisions-panel").style.display = dec.length ? "block" : "none";
  $("decisions").innerHTML = dec.map(d =>
    `<li><span>${esc(d.question)}</span><span class="muted">${esc(d.urgency)}</span></li>`
  ).join("");

  setControls(run);
}

let currentProject = "";

async function control(action, body){
  const result = await post("/api/control/" + action, body || {});
  const note = $("ctl-note");
  if (result.queued) {
    // The production is running in another process, so the request has to
    // travel through storage. Say so rather than looking unresponsive.
    note.textContent = "Requested. " + (result.note || "");
  } else if (result.error) {
    note.textContent = result.error;
  } else if (result.ok) {
    note.textContent = action + " applied.";
  } else {
    note.textContent = result.note || "Nothing to " + action + ".";
  }
  setTimeout(() => { note.textContent = ""; }, 8000);
  refresh();
}

async function refresh(){
  try {
    const url = currentProject
      ? "/api/state?project_id=" + encodeURIComponent(currentProject)
      : "/api/state";
    const r = await fetch(url);
    const s = await r.json();
    render(s);
    if (!currentProject && s.project) currentProject = s.project.project_id;
  } catch (e) { /* server restarting; the next tick will catch up */ }
}

// Live push. Any production event triggers an immediate re-render, so the
// preview appears the instant Blender finishes writing it.
function connect(){
  const es = new EventSource("/events");
  es.addEventListener("production", () => refresh());
  es.onerror = () => { es.close(); setTimeout(connect, 2500); };
}

$("btn-pause").onclick  = () => control("pause",  {project_id: currentProject});
$("btn-resume").onclick = () => control("resume", {project_id: currentProject});
$("btn-retry").onclick  = () => control("retry",  {project_id: currentProject});
$("btn-stop").onclick = () => {
  if (!confirm("Stop the production? Work already completed is kept, but the "
             + "shot being produced right now will be abandoned.")) return;
  control("stop", {confirm: true, project_id: currentProject});
};

setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString(); }, 1000);
refresh();
connect();
setInterval(refresh, 5000);   // safety net if the event stream drops
</script>
</body>
</html>
"""

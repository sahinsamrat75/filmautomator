"""Verify the dashboard reflects a real production, live.

Section 14 of the spec requires that the preview the owner sees is the same
artifact the Vision Agent judged, and section 12 requires the dashboard to show
real state with no mock data. This script proves both by running an actual
production and watching the dashboard's own API while it happens.

    python3 scripts/check_dashboard_live.py

Exits non-zero if the dashboard ever fails to reflect what really happened.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filmautomator.config import AppConfig  # noqa: E402
from filmautomator.core.events import EventBus  # noqa: E402
from filmautomator.dashboard import DashboardServer  # noqa: E402
from filmautomator.mcp.server import MCPServer  # noqa: E402
from filmautomator.mcp.tools import ToolContext  # noqa: E402
from filmautomator.runtime import ProductionRunner  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  {detail}")


def fetch(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="fa_dash_live_"))
    print(f"workspace: {workspace}\n")

    config = AppConfig()
    config.workspace = workspace
    config.render.final_engine = "BLENDER_EEVEE"
    config.render.final_width = 320
    config.render.final_height = 180
    config.render.final_samples = 8
    config.render.preview_width = 240
    config.render.preview_height = 135

    # One event bus shared by the dashboard and the MCP server, exactly as a
    # real deployment wires them.
    # Deliberately two independent runners on the same workspace: this mirrors
    # the real deployment, where `filmautomator dashboard` and the MCP server
    # Claude Code spawns are separate processes that share only storage.
    events = EventBus()
    dashboard_runner = ProductionRunner(config, events)
    dashboard = DashboardServer(config, dashboard_runner, events)
    port = dashboard.start("127.0.0.1", 0)
    server = MCPServer(context=ToolContext.create(config), config=config)

    def call(tool: str, args: dict | None = None) -> dict:
        response = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": args or {}},
        })
        return json.loads(response["result"]["content"][0]["text"])

    try:
        print("before production")
        state = fetch(port, "/api/state")
        check("dashboard answers with no projects", state["project"] is None,
              str(state)[:160])

        print("starting a real production via MCP")
        project = call("create_project", {
            "name": "DASH_LIVE", "objective": "a figure in the rain",
            "duration_s": 1.0,
        })
        project_id = project["project_id"]
        call("start_production", {
            "project_id": project_id, "objective": "a figure in the rain",
            "duration_s": 1.0, "offline": True, "engine": "BLENDER_EEVEE",
            "width": 320, "height": 180,
        })

        # Watch the dashboard while it runs.
        saw_running = False
        saw_preview = False
        saw_event = False
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            state = fetch(port, "/api/state")
            run = state.get("run") or {}
            if run.get("active"):
                saw_running = True
            if state.get("preview"):
                saw_preview = True
            if state.get("events"):
                saw_event = True
            if run.get("state") in {"COMPLETED", "FAILED", "STOPPED"}:
                break
            time.sleep(1.0)

        check("dashboard showed the production running", saw_running)
        check("dashboard showed a live preview", saw_preview)
        check("dashboard showed the agent activity feed", saw_event)

        state = fetch(port, "/api/state")
        run = state.get("run") or {}
        check("production completed", run.get("state") == "COMPLETED",
              str(run)[:200])

        # The preview the dashboard serves must be the artifact the system
        # recorded — not a copy, not a placeholder.
        preview = state.get("preview")
        check("dashboard reports a preview artifact", bool(preview))
        if preview:
            artifact_id = preview["artifact_id"]
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/artifact/{artifact_id}", timeout=15
            ) as response:
                served = response.read()
                mime = response.headers.get("Content-Type")
            check("dashboard serves the preview as an image",
                  mime == "image/png", str(mime))
            check("served preview is a real PNG (not a stub)",
                  served.startswith(b"\x89PNG") and len(served) > 1000,
                  f"{len(served)} bytes")

            db = dashboard._db_handle()  # noqa: SLF001
            record = db.get_artifact(artifact_id)
            on_disk = Path(record["path"]).read_bytes()
            check("served bytes are byte-identical to the recorded artifact",
                  served == on_disk)

        # The final movie, visible from the dashboard.
        movie = state.get("final_movie")
        check("dashboard reports the final movie", bool(movie))
        if movie:
            check("final movie exists on disk", Path(movie["path"]).is_file(),
                  movie["path"])
            check("final movie is named after the project",
                  Path(movie["path"]).name == "DASH_LIVE_FINAL.mp4",
                  Path(movie["path"]).name)

        # Every event the dashboard shows must exist in the database. This is
        # the check that rules out invented activity.
        db = dashboard._db_handle()  # noqa: SLF001
        real = {e["seq"] for e in db.events_after(project_id, 0, 500)}
        shown = {e["seq"] for e in state.get("events", [])}
        check("every displayed event exists in the database",
              shown <= real, f"phantom events: {sorted(shown - real)}")
        check("dashboard is showing a meaningful amount of activity",
              len(shown) >= 5, f"{len(shown)} events")

    finally:
        server.close()
        dashboard.stop()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

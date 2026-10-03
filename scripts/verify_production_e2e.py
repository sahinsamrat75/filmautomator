#!/usr/bin/env python3
"""End-to-end production verification over the real MCP stdio transport.

Claude Desktop does not call the server in-process: it spawns
`python3 -m filmautomator mcp` and speaks JSON-RPC over stdin/stdout. This
script does exactly that — same command, same transport, same tool surface — so
what it verifies is the path the extension actually uses.

It then checks the invariants that the original bug violated:

  * every registered artifact physically exists
  * the final movie exists, is non-zero, and probes as valid H.264 video
  * the production report exists
  * status only says COMPLETED once all of the above is true
  * QA's verdict matches the files on disk

    python3 scripts/verify_production_e2e.py [--duration 5] [--width 960]

Exits non-zero if any invariant fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


class StdioMCP:
    """A minimal MCP client over a real subprocess' stdio."""

    def __init__(self, workspace: Path) -> None:
        env = dict(os.environ)
        env["FA_WORKSPACE"] = str(workspace)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        self.workspace = workspace
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "filmautomator", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(ROOT), env=env,
        )
        self._id = 0

    def _send(self, method: str, params: dict | None = None,
              notify: bool = False) -> dict | None:
        message = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            message["id"] = self._id
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed the connection")
        return json.loads(line)

    def initialize(self) -> dict:
        result = self._send("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "verify-production", "version": "1.0"},
        })
        self._send("notifications/initialized", notify=True)
        return result["result"]

    def call(self, tool: str, args: dict | None = None) -> dict:
        response = self._send("tools/call",
                              {"name": tool, "arguments": args or {}})
        result = response.get("result") or {}
        block = (result.get("content") or [{}])[0]
        text = block.get("text", "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"_raw": text}
        payload["_is_error"] = bool(result.get("isError"))
        return payload

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    workspace = ROOT / "projects"
    print(f"workspace : {workspace}")
    print(f"transport : stdio, via `python3 -m filmautomator mcp`")
    print(f"production: {args.duration}s at {args.width}x{args.height}")
    print()

    client = StdioMCP(workspace)
    try:
        init = client.initialize()
        info = init.get("serverInfo") or {}
        check("MCP initialize completes", bool(info))
        print(f"        {info.get('name')} {info.get('version')} "
              f"(protocol {init.get('protocolVersion')})")

        name = f"E2E_{int(time.time())}"
        created = client.call("create_project", {
            "name": name,
            "objective": "A lone figure stands on a dark rainy street at night, "
                         "backlit by a streetlamp, slow push-in camera.",
            "duration_s": args.duration,
        })
        project_id = created.get("project_id", "")
        check("create_project returns a project_id", bool(project_id),
              str(created)[:200])
        if not project_id:
            return 1
        print(f"        project_id: {project_id}")

        started = client.call("start_production", {
            "project_id": project_id,
            "duration_s": args.duration,
            "offline": True,
            "engine": "BLENDER_EEVEE",
            "width": args.width, "height": args.height,
        })
        check("start_production returns immediately", started.get("started") is True,
              str(started)[:200])

        print("\n  producing (this takes a few minutes)…")
        deadline = time.monotonic() + args.timeout
        status: dict = {}
        last_line = ""
        while time.monotonic() < deadline:
            status = client.call("get_production_status")
            line = (f"    {status.get('state','?'):<10} "
                    f"{status.get('percent',0):>5.1f}%  "
                    f"{str(status.get('current_stage',''))[:60]}")
            if line != last_line:
                print(line, flush=True)
                last_line = line
            if status.get("state") in {"COMPLETED", "FAILED", "STOPPED"}:
                break
            time.sleep(3)
        print()

        check("production reached a terminal state",
              status.get("state") in {"COMPLETED", "FAILED", "STOPPED"}, str(status)[:200])
        check("production is COMPLETED", status.get("state") == "COMPLETED",
              json.dumps(status)[:400])

        # -- the invariants -------------------------------------------------
        artifacts = client.call("list_artifacts", {"project_id": project_id})
        # Every *deliverable* must exist on disk. Rendered frames and shot
        # previews are deliberately excluded: once a shot is promoted and
        # verified the storage governor releases them, and that is the correct
        # outcome rather than a missing artifact.
        deliverable_kinds = {"shot_video", "final_movie", "timeline",
                             "qa_report", "report", "audio"}
        missing_deliverables = [
            a for a in artifacts.get("artifacts", [])
            if a.get("kind") in deliverable_kinds and not a.get("exists")
        ]
        check("every deliverable artifact exists on disk",
              not missing_deliverables, json.dumps(missing_deliverables)[:400])

        # Anything reported missing must be a releasable artifact, never a
        # deliverable — this is the invariant the original bug violated.
        unexpected_missing = [
            a for a in artifacts.get("artifacts", [])
            if not a.get("exists") and a.get("kind") not in {"render", "preview"}
        ]
        check("no deliverable artifact is missing",
              not unexpected_missing, json.dumps(unexpected_missing)[:400])

        released = [a for a in artifacts.get("artifacts", [])
                    if not a.get("exists")]
        if released:
            print(f"        released after verification: "
                  f"{len(released)} frame/preview artifact(s)")

        integrity = status.get("integrity") or {}
        check("status carries a passing integrity report",
              integrity.get("ok") is True, json.dumps(integrity)[:300])

        movie = client.call("get_final_movie", {"project_id": project_id})
        check("get_final_movie succeeds", not movie.get("_is_error"),
              str(movie)[:300])
        final_path = Path(movie.get("final_movie", ""))
        check("final movie exists on disk", final_path.is_file(), str(final_path))
        if final_path.is_file():
            size = final_path.stat().st_size
            check("final movie is non-zero", size > 0, f"{size} bytes")
            check("final movie is named after the project",
                  final_path.name == f"{name}_FINAL.mp4", final_path.name)
            check("final movie is in final/", final_path.parent.name == "final",
                  str(final_path.parent))
            print(f"\n  FINAL MOVIE : {final_path}")
            print(f"  SIZE        : {size:,} bytes ({size/1e6:.2f} MB)")

        check("ffprobe reports valid H.264 video",
              movie.get("codec") == "h264", str(movie.get("codec")))
        check("ffprobe reports the expected resolution",
              movie.get("resolution") == f"{args.width}x{args.height}",
              str(movie.get("resolution")))
        check("ffprobe reports a plausible duration",
              abs((movie.get("duration_s") or 0) - args.duration) < 1.0,
              str(movie.get("duration_s")))
        print(f"  FFPROBE     : {movie.get('codec')} "
              f"{movie.get('resolution')} {movie.get('fps')}fps "
              f"{movie.get('duration_s')}s")
        print(f"  VERIFIED    : {movie.get('verified')}")

        reports = [a for a in artifacts.get("artifacts", [])
                   if a.get("kind") == "report"]
        check("production report exists",
              bool(reports) and reports[0].get("exists") is True,
              json.dumps(reports)[:250])

        qa = client.call("get_qa_status", {"project_id": project_id})
        check("QA verdict matches the files on disk",
              qa.get("status") == "PASSED", json.dumps(qa)[:300])

        # -- did the agents actually run? -----------------------------------
        print()
        activity = client.call("get_agent_activity",
                               {"project_id": project_id, "limit": 200})
        kinds = {e["kind"] for e in activity.get("events", [])}
        for event_kind, label in (
            ("vision.started", "Vision Agent executed"),
            ("vision.completed", "Vision Agent produced a verdict"),
            ("render.completed", "Blender rendered frames"),
            ("editing.completed", "FFmpeg encoded the shot"),
            ("qa.passed", "final QA passed"),
            ("production.completed", "production completed"),
        ):
            check(label, event_kind in kinds, f"missing {event_kind}")

        # The correction loop only runs when a shot fails review. Report which
        # happened rather than asserting.
        revisions = [e for e in activity.get("events", [])
                     if e["kind"] == "preview.ready"]
        print(f"        preview passes: {len(revisions)} "
              f"({'correction loop exercised' if len(revisions) > 1 else 'shot passed on first review'})")

    finally:
        client.close()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

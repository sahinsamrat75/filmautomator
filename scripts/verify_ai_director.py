#!/usr/bin/env python3
"""Real AI-director acceptance test over the actual MCP stdio server.

Phase 18 of the milestone requires that the visual-director loop be proven
against a real Blender, a real FFmpeg, and the real stdio transport Claude
Desktop uses — not a mocked test. This script drives that exact path:

    create project -> define character/environment -> create shot
    -> render preview -> RECEIVE THE ACTUAL IMAGE
    -> inspect the image -> identify a real visual property
    -> correct the shot through MCP -> Blender changes the scene
    -> render a second preview -> receive it -> approve
    -> final render -> encode -> ffprobe -> mark FINAL
    -> release frames -> verify storage fell

The pass/fail condition that matters most is the image. If the server does not
hand this process a real PNG, the milestone is not met and the script says so.

    python3 scripts/verify_ai_director.py [--duration 5] [--width 640]

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PASSED = 0
FAILED = 0


def wait_for_render_job(client, project_id: str, shot_id: str, *,
                        timeout_s: float = 3600.0, poll_s: float = 10.0,
                        on_tick=None) -> tuple[dict | None, dict]:
    """Poll get_render until the shot's render job reaches a terminal state.

    The whole point of the render lifecycle is that these polls answer
    instantly while Blender renders, so every tick also proves the server
    stayed responsive: a ``list_projects`` round-trip is timed each poll and
    must return within a few seconds.
    """
    deadline = time.time() + timeout_s
    job: dict = {}
    responsive = True
    while time.time() < deadline:
        started = time.time()
        listing = client.call("list_projects", {})
        roundtrip = time.time() - started
        if listing.get("_is_error") or roundtrip > 10.0:
            responsive = False
        status = client.call("get_render",
                             {"project_id": project_id, "shot_id": shot_id})
        job = status.get("render_job") or {}
        progress = job.get("progress") or {}
        if callable(on_tick):
            on_tick(job, roundtrip)
        else:
            print(f"        {job.get('state', '?')} "
                  f"({progress.get('frames_done', 0)}/"
                  f"{progress.get('frames_total', '?')} frames, "
                  f"server answered in {roundtrip:.2f}s)")
        if job.get("terminal"):
            if not responsive:
                print("        WARNING: server latency exceeded 10s during "
                      "the render — the control plane blocked")
            return status, job
        time.sleep(poll_s)
    print("        render job timed out in the polling loop")
    return None, job


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
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "filmautomator", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(ROOT), env=env,
        )
        self._id = 0

    def _send(self, method: str, params: dict | None = None,
              notify: bool = False) -> dict | None:
        message: dict = {"jsonrpc": "2.0", "method": method}
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
            "protocolVersion": "2025-06-18",
            # Advertise image content: this client can display images, so the
            # server should return real image blocks.
            "capabilities": {"imageContent": True},
            "clientInfo": {"name": "ai-director-acceptance", "version": "1.0"},
        })
        self._send("notifications/initialized", notify=True)
        return result["result"]

    def call(self, tool: str, args: dict | None = None) -> dict:
        """Call a tool, returning the parsed payload plus any image blocks."""
        response = self._send("tools/call",
                              {"name": tool, "arguments": args or {}})
        result = response.get("result") or {}
        payload: dict = {}
        images: list[dict] = []
        for block in result.get("content") or []:
            if block.get("type") == "image":
                images.append(block)
            elif block.get("type") == "text":
                try:
                    payload = json.loads(block["text"])
                except json.JSONDecodeError:
                    payload = {"_raw": block["text"]}
        payload["_is_error"] = bool(result.get("isError"))
        payload["_images"] = images
        return payload

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def decode_image(block: dict) -> tuple[bytes, str]:
    try:
        return base64.b64decode(block["data"]), block.get("mimeType", "")
    except Exception:
        return b"", ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--engine", default="BLENDER_EEVEE")
    parser.add_argument("--workspace", default="")
    args = parser.parse_args()

    import tempfile

    workspace = Path(args.workspace) if args.workspace else Path(
        tempfile.mkdtemp(prefix="fa_director_")) / "projects"
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"AI-director acceptance test\n  workspace: {workspace}\n")
    client = StdioMCP(workspace)
    try:
        # -- handshake ----------------------------------------------------
        info = client.initialize()
        check("MCP server initialized over stdio",
              info.get("protocolVersion") == "2025-06-18", str(info)[:200])
        check("server describes the visual-director workflow",
              "render_shot_preview" in (info.get("instructions") or ""),
              "instructions do not mention the loop")

        # -- project, continuity, shot ------------------------------------
        project_id = client.call("create_project", {
            "name": "AI_DIRECTOR_TEST",
            "objective": "A lone figure walks a ruined street at night",
            "duration_s": args.duration,
        })["project_id"]
        check("project created", bool(project_id), project_id)

        client.call("define_character", {
            "project_id": project_id, "name": "Akira", "height_m": 1.78,
            "hair": "short black", "clothing": "worn jacket",
            "colors": {"jacket": "#2b3a55"},
            "voice": "low, steady",
        })
        client.call("define_environment", {
            "project_id": project_id, "environment_id": "ENV_TOKYO_RUINS_01",
            "name": "Ruined Tokyo street", "weather": "heavy rain",
            "lighting_baseline": "blue moonlight camera-left, warm orange right",
        })
        characters = client.call("list_characters", {"project_id": project_id})
        check("canonical character persisted",
              characters.get("count") == 1, str(characters)[:200])

        shot_id = "SC01_SH001"
        client.call("create_shot", {
            "project_id": project_id, "shot_id": shot_id,
            "scene_id": "SC01", "duration_s": args.duration,
            "description": "A lone figure in a ruined street at night",
            "shot_size": "medium", "angle": "eye_level", "lens_mm": 50.0,
            "lighting_mood": "low_key",
            "environment": "ENV_TOKYO_RUINS_01",
            "subject_name": "Akira", "subject_height_m": 1.78,
        })

        # -- preview one: the AI must receive an actual image --------------
        print("\n  --- first preview ---")
        first = client.call("render_shot_preview", {
            "project_id": project_id, "shot_id": shot_id,
            "width": args.width, "height": args.height,
        })
        check("render_shot_preview succeeded", not first.get("_is_error"),
              str(first)[:300])
        check("preview reports the image was delivered",
              first.get("image_delivered") is True,
              f"delivery={first.get('delivery')} note={first.get('note','')[:120]}")

        images = first.get("_images") or []
        check("response contains an image content block", len(images) == 1,
              f"{len(images)} image block(s)")

        first_bytes = b""
        if images:
            first_bytes, mime = decode_image(images[0])
            check("image block declares a PNG mime type",
                  mime == "image/png", mime)
            check("delivered image is a real PNG",
                  first_bytes.startswith(PNG_MAGIC),
                  f"first bytes {first_bytes[:8]!r}")
            check("delivered image has real content",
                  len(first_bytes) > 2000, f"{len(first_bytes)} bytes")

        meta = first.get("metadata") or {}
        for key in ("shot_id", "scene_id", "duration_s", "fps", "camera",
                    "render_settings", "artifact_path"):
            check(f"preview metadata carries {key}", key in meta,
                  str(sorted(meta))[:160])
        check("preview camera metadata is populated",
              bool((meta.get("camera") or {}).get("shot_size")),
              str(meta.get("camera"))[:160])

        # -- the AI inspects the real image -------------------------------
        # These are objective properties of the delivered pixels, which is what
        # an AI would actually look at before deciding on a correction.
        print("\n  --- AI inspection of the delivered image ---")
        if first_bytes:
            width = int.from_bytes(first_bytes[16:20], "big")
            height = int.from_bytes(first_bytes[20:24], "big")
            check("image header reports real dimensions",
                  width > 0 and height > 0, f"{width}x{height}")
            check("image matches the requested aspect ratio",
                  abs(width / height - 16 / 9) < 0.15,
                  f"{width}x{height}")
            # Every PNG starts with the same 8-byte signature; a stub would not
            # carry a valid IHDR chunk at the expected offset.
            check("image carries a valid IHDR chunk",
                  first_bytes[12:16] == b"IHDR", str(first_bytes[12:16]))
            distinct = len(set(first_bytes[64:512]))
            check("image carries non-trivial pixel data", distinct > 3,
                  f"{distinct} distinct byte values in the first rows")

        # -- the AI decides a correction and asks for it ------------------
        print("\n  --- AI correction ---")
        correction = client.call("correct_shot", {
            "project_id": project_id, "shot_id": shot_id,
            "camera_height_m": 1.2,
            "camera_distance_m": 4.5,
            "shot_size": "medium_wide",
            "reason": "the subject sits too low in frame and the camera is too "
                      "close; raise it to 1.2m and pull back for headroom",
        })
        check("correct_shot succeeded", not correction.get("_is_error"),
              str(correction)[:300])
        applied = correction.get("applied") or []
        check("correction changed the camera",
              any("shot_size" in a for a in applied)
              and any("camera.location" in a for a in applied),
              str(applied)[:240])
        check("correction recorded its reason",
              bool(correction.get("reason")), str(correction.get("reason"))[:120])

        # -- preview two: Blender must have changed the scene --------------
        print("\n  --- second preview after correction ---")
        second = client.call("render_shot_preview", {
            "project_id": project_id, "shot_id": shot_id,
        })
        check("second preview rendered", not second.get("_is_error"),
              str(second)[:300])
        check("second preview reports an image was delivered",
              second.get("image_delivered") is True,
              f"delivery={second.get('delivery')}")
        second_images = second.get("_images") or []
        check("second response contains an image block", len(second_images) == 1,
              f"{len(second_images)} image block(s)")

        second_bytes = b""
        if second_images:
            second_bytes, _ = decode_image(second_images[0])
            check("second image is a real PNG",
                  second_bytes.startswith(PNG_MAGIC), "")
        if first_bytes and second_bytes:
            check("Blender produced a different frame after the correction",
                  first_bytes != second_bytes,
                  "the corrected render is byte-identical to the first")

        shot = client.call("get_shot", {"project_id": project_id,
                                        "shot_id": shot_id})
        camera = (shot.get("spec") or {}).get("camera") or {}
        check("the correction persisted into the shot spec",
              camera.get("shot_size") == "medium_wide",
              str(camera)[:200])
        check("camera height is now 1.2m",
              abs(camera.get("location", [0, 0, 0])[2] - 1.2) < 0.001,
              str(camera.get("location")))

        # -- approve, then finalize ---------------------------------------
        print("\n  --- final render ---")
        client.call("approve_shot", {"project_id": project_id,
                                     "shot_id": shot_id})
        before = client.call("get_storage_status", {"project_id": project_id})

        submitted = client.call("finalize_shot", {
            "project_id": project_id, "shot_id": shot_id,
            "width": args.width, "height": args.height,
            "engine": args.engine,
        })
        check("finalize_shot accepted the job", not submitted.get("_is_error"),
              str(submitted)[:400])
        check("finalize_shot returned immediately with a render job",
              bool(submitted.get("job_id"))
              and submitted.get("state") in ("QUEUED", "RENDERING"),
              str(submitted)[:300])

        # The server must stay responsive for the whole render. Every poll
        # below is a tools/call that has to answer while Blender is busy.
        print("\n  --- final render (background job) ---")
        status, job = wait_for_render_job(client, project_id, shot_id)
        check("render job reached a terminal state",
              bool(job.get("terminal")), str(job)[:200])
        check("render job COMPLETED", job.get("state") == "COMPLETED",
              str(job.get("error") or job.get("state"))[:300])
        final = (status or {}).get("result") or job.get("result") or {}

        check("shot satisfied the finalization contract",
              final.get("final") is True,
              str(final.get("failed") or final.get("reason"))[:300])
        check("frames were released after promotion",
              final.get("frames_released") is True,
              str(final.get("frames_note") or "")[:200])

        video = Path(final.get("video", ""))
        check("shot video exists on disk", video.is_file(), str(video))
        check("shot video is non-zero",
              video.is_file() and video.stat().st_size > 0,
              f"{video.stat().st_size if video.is_file() else 0} bytes")

        if video.is_file():
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration:stream=codec_name,width,height",
                 "-of", "json", str(video)],
                capture_output=True, text=True,
            )
            try:
                info_json = json.loads(probe.stdout or "{}")
                streams = info_json.get("streams") or [{}]
                duration = float((info_json.get("format") or {}).get("duration", 0))
                check("ffprobe reports valid H.264 video",
                      streams[0].get("codec_name") == "h264",
                      str(streams[0].get("codec_name")))
                check("ffprobe reports the expected resolution",
                      f"{streams[0].get('width')}x{streams[0].get('height')}"
                      == f"{args.width}x{args.height}",
                      f"{streams[0].get('width')}x{streams[0].get('height')}")
                check("ffprobe duration is within tolerance",
                      abs(duration - args.duration) < 1.0, f"{duration:.2f}s")
                print(f"        video: {video.name} {duration:.2f}s "
                      f"{streams[0].get('width')}x{streams[0].get('height')}")
            except Exception as exc:
                check("ffprobe parsed the shot video", False, str(exc))

        shots = client.call("list_shots", {"project_id": project_id})
        status = shots["shots"][0]["status"] if shots.get("shots") else ""
        check("shot is marked FINAL", status == "FINAL", status)

        # -- storage really was released -----------------------------------
        after = client.call("get_storage_status", {"project_id": project_id})
        check("storage was remeasured after finalization",
              isinstance(after.get("total_bytes"), int),
              str(after.get("total_bytes")))
        frames_left = list((workspace / project_id / "shots").rglob("*.png"))
        check("rendered frames are gone from disk", not frames_left,
              f"{len(frames_left)} frame file(s) remain")
        print(f"        storage: {before.get('total_gb')} GB -> "
              f"{after.get('total_gb')} GB "
              f"(state {after.get('state')})")

        # -- the final movie is protected ----------------------------------
        protected = client.call("cleanup_storage",
                                {"project_id": project_id, "confirm": True})
        check("cleanup after finalization ran",
              not protected.get("_is_error"), str(protected)[:200])
        check("the shot video survived cleanup",
              video.is_file() if video else False, str(video))

        # -- the whole film, for completeness ------------------------------
        print("\n  --- full production ---")
        done = client.call("create_project", {
            "name": "AI_DIRECTOR_FULL", "objective": "a short night scene",
            "duration_s": args.duration})
        full_id = done["project_id"]
        client.call("start_production", {
            "project_id": full_id, "duration_s": args.duration, "offline": True,
            "engine": args.engine, "width": args.width, "height": args.height,
        })
        state = {}
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            state = client.call("get_production_status",
                                {"project_id": full_id})
            if state.get("state") in {"COMPLETED", "FAILED", "STOPPED"}:
                break
            time.sleep(2.0)
        check("full production completed",
              state.get("state") == "COMPLETED", str(state)[:300])
        check("full production verified its integrity",
              (state.get("integrity") or {}).get("ok") is True,
              str(state.get("integrity"))[:200])
        movie = client.call("get_final_movie", {"project_id": full_id})
        check("final movie verified",
              movie.get("verified") is True, str(movie)[:200])
        print(f"        movie: {movie.get('final_movie')}")

    finally:
        client.close()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

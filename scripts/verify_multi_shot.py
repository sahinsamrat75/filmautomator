#!/usr/bin/env python3
"""FILMAUTOMATOR_REAL_MULTI_SHOT_TEST — the real five-shot production.

This is the acceptance test the builders milestone exists for. It drives the
actual MCP stdio server (the same transport Claude Desktop uses) through five
real shots, each one going:

    BUILD -> REAL BLENDER SCENE -> PREVIEW -> ACTUAL IMAGE DELIVERY
        -> AI VISUAL REVIEW -> CORRECT IF NECESSARY
        -> FINAL RENDER -> FFMPEG ENCODE -> VERIFY -> FINALIZE
        -> RELEASE TEMPORARY FRAMES

SH001 establishing courtyard
SH002 Traveler enters
SH003 Traveler performs an action
SH004 close-up
SH005 reaction/ending

Then it assembles the five shot videos into one final movie, ffprobes it, and
checks the filesystem so the finished film is physically real — never a claim
with no file behind it.

The most important assertion is repeated for every shot: the AI Director must
receive the *actual* rendered image, not a path to one. If any shot fails to
deliver a real PNG, the test is a failure.

    python3 scripts/verify_multi_shot.py [--duration 3] [--width 480]

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
                        timeout_s: float = 3600.0,
                        poll_s: float = 10.0) -> tuple[dict | None, dict]:
    """Poll get_render until the shot's job is terminal.

    Each poll also times a ``list_projects`` round-trip: the render lifecycle
    exists precisely so the control plane stays responsive, and this makes
    that a measured assertion rather than an assumption.
    """
    deadline = time.time() + timeout_s
    job: dict = {}
    slowest = 0.0
    while time.time() < deadline:
        started = time.time()
        client.call("list_projects", {})
        roundtrip = time.time() - started
        slowest = max(slowest, roundtrip)
        status = client.call("get_render",
                             {"project_id": project_id, "shot_id": shot_id})
        job = status.get("render_job") or {}
        progress = job.get("progress") or {}
        print(f"        {job.get('state', '?')} "
              f"({progress.get('frames_done', 0)}/"
              f"{progress.get('frames_total', '?')} frames, "
              f"poll round-trip {roundtrip:.2f}s)")
        if job.get("terminal"):
            job["_slowest_poll_s"] = slowest
            return status, job
        time.sleep(poll_s)
    print("        render job did not finish within the polling window")
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
            "capabilities": {"imageContent": True},
            "clientInfo": {"name": "multi-shot-acceptance", "version": "1.0"},
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
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--engine", default="BLENDER_EEVEE")
    parser.add_argument("--workspace", default="")
    args = parser.parse_args()

    import tempfile

    workspace = Path(args.workspace) if args.workspace else Path(
        tempfile.mkdtemp(prefix="fa_multishot_")) / "projects"
    workspace.mkdir(parents=True, exist_ok=True)

    print("MULTI-SHOT ACCEPTANCE TEST (five real shots)")
    print(f"  workspace: {workspace}")

    client = StdioMCP(workspace)
    storage_before = None
    shots_out: dict[str, dict] = {}
    try:
        # -- handshake ----------------------------------------------------
        info = client.initialize()
        check("MCP server initialized over stdio",
              info.get("protocolVersion") == "2025-06-18", str(info)[:200])

        # -- project + canonical continuity -------------------------------
        project = client.call("create_project", {
            "name": "MULTI_SHOT_TEST",
            "objective": "The Traveler crosses the courtyard at dusk, reaches "
                         "the fountain, and reacts",
            "duration_s": args.duration * 5,
        })
        project_id = project["project_id"]
        check("project created", bool(project_id), project_id)

        character = client.call("define_character", {
            "project_id": project_id, "name": "Traveler", "height_m": 1.78,
            "clothing": "coat, trousers, boots",
            "colors": {"coat": "#2b6f6f", "trousers": "#23232a",
                       "boots": "#1d1815", "skin": "#c68a5b", "hair": "#2a241e"},
            "voice": "low, steady", "behavior": "deliberate, wary",
        })
        check("canonical Traveler defined", not character.get("_is_error"),
              str(character)[:200])

        environment = client.call("define_environment", {
            "project_id": project_id, "environment_id": "ENV_COURTYARD_DUSK_01",
            "name": "The courtyard at dusk", "weather": "clear dusk",
            "lighting_baseline": "warm lantern light, cool sky",
            "layout": json.dumps({
                "fountain": [2.6, 2.2, 0.0], "lantern": [-2.4, 1.8, 0.0],
                "archway": [0.0, -3.4, 0.0],
            }),
            "props": ["fountain", "lantern", "archway"],
        })
        check("canonical environment defined", not environment.get("_is_error"),
              str(environment)[:200])

        # The five-shot plan.
        plan = [
            ("SC01_SH001", "Wide establishing shot of the empty courtyard at "
                           "dusk; lanterns lit, fountain in the middle ground.",
             "extreme_wide", "dolly_in", 35.0, "idle", "natural"),
            ("SC01_SH002", "The Traveler steps through the archway and walks "
                           "in, crossing toward the fountain.",
             "wide", "dolly_out", 35.0, "walk", "blue_hour"),
            ("SC01_SH003", "Medium shot as the Traveler reaches out and touches "
                           "the fountain's basin.",
             "medium", "static", 50.0, "reach", "blue_hour"),
            ("SC01_SH004", "Close-up on the Traveler's face; the lantern glow "
                           "catches the profile.",
             "close_up", "tilt_up", 85.0, "lean", "candlelit"),
            ("SC01_SH005", "The Traveler turns and looks back over the "
                           "shoulder, then holds still.",
             "medium_close", "static", 50.0, "react", "blue_hour"),
        ]
        for index, (shot_id, description, shot_size, movement, lens,
                    action, mood) in enumerate(plan, start=1):
            print(f"\n=== SHOT {shot_id} ({index}/5) ===")
            created = client.call("create_shot", {
                "project_id": project_id, "shot_id": shot_id,
                "scene_id": "SC01", "duration_s": args.duration,
                "description": description, "shot_size": shot_size,
                "angle": "eye_level", "lens_mm": lens, "movement": movement,
                "lighting_mood": mood, "environment": "ENV_COURTYARD_DUSK_01",
                "subject_name": "Traveler", "subject_height_m": 1.78,
                "subject_action": action,
            })
            check(f"{shot_id} shot defined", not created.get("_is_error"),
                  str(created)[:200])

            # -- preview one: the AI must receive the actual image ---------
            preview = client.call("render_shot_preview", {
                "project_id": project_id, "shot_id": shot_id,
            })
            check(f"{shot_id} preview one rendered",
                  not preview.get("_is_error"), str(preview)[:200])
            check(f"{shot_id} preview delivered an image",
                  preview.get("image_delivered") is True,
                  f"delivery={preview.get('delivery')} "
                  f"note={str(preview.get('note'))[:120]}")

            images = preview.get("_images") or []
            check(f"{shot_id} response contains one image block",
                  len(images) == 1, f"{len(images)} block(s)")
            png = b""
            if images:
                png, mime = decode_image(images[0])
                check(f"{shot_id} image is a real PNG",
                      png.startswith(PNG_MAGIC), f"{png[:8]!r}")
                w = int.from_bytes(png[16:20], "big")
                h = int.from_bytes(png[20:24], "big")
                check(f"{shot_id} PNG has real dimensions",
                      w > 0 and h > 0, f"{w}x{h}")
                check(f"{shot_id} metadata records the preview version",
                      (preview.get("metadata") or {}).get("version") == 1,
                      str((preview.get("metadata") or {}))[:160])

            # -- AI visual review + correction loop ------------------------
            correction = None
            if shot_id == "SC01_SH003":
                correction = client.call("correct_shot", {
                    "project_id": project_id, "shot_id": shot_id,
                    "shot_size": "medium_close",
                    "reason": ("review: the reach action reads small in frame; "
                               "tighten to a medium-close so the fountain and "
                               "hand both fill the shot"),
                })
            elif shot_id == "SC01_SH004":
                correction = client.call("correct_shot", {
                    "project_id": project_id, "shot_id": shot_id,
                    "camera_height_m": 1.2, "lens_mm": 100.0,
                    "shot_size": "close_up",
                    "reason": ("review: the close-up is too far away; move "
                               "closer with a longer lens and lower camera"),
                })

            corrections_for_shot = 0
            if correction is not None:
                check(f"{shot_id} correction applied",
                      not correction.get("_is_error")
                      and bool(correction.get("applied")),
                      str(correction)[:200])
                corrections_for_shot = len(correction.get("applied") or [])

                # -- preview two: Blender must have changed the scene -----
                second = client.call("render_shot_preview", {
                    "project_id": project_id, "shot_id": shot_id,
                })
                check(f"{shot_id} preview two rendered",
                      not second.get("_is_error"), str(second)[:200])
                check(f"{shot_id} preview two delivered an image",
                      second.get("image_delivered") is True,
                      f"delivery={second.get('delivery')}")
                second_images = second.get("_images") or []
                second_png = b""
                if second_images:
                    second_png, _ = decode_image(second_images[0])
                    check(f"{shot_id} second image is a real PNG",
                          second_png.startswith(PNG_MAGIC), "")
                if png and second_png:
                    check(f"{shot_id} Blender changed the scene after the "
                          "correction", png != second_png,
                          "corrected render is byte-identical to the first")
            # -- finalize: real final render, encode, ffprobe, cleanup -----
            submitted = client.call("finalize_shot", {
                "project_id": project_id, "shot_id": shot_id,
                "width": args.width, "height": args.height, "engine": args.engine,
            })
            check(f"{shot_id} finalize accepted the job",
                  not submitted.get("_is_error"), str(submitted)[:300])
            check(f"{shot_id} finalize returned a background render job",
                  bool(submitted.get("job_id"))
                  and submitted.get("state") in ("QUEUED", "RENDERING"),
                  str(submitted)[:200])

            status, job = wait_for_render_job(client, project_id, shot_id)
            check(f"{shot_id} render job finished", bool(job.get("terminal")),
                  str(job)[:200])
            check(f"{shot_id} render job COMPLETED",
                  job.get("state") == "COMPLETED",
                  str(job.get("error") or job.get("state"))[:200])
            check(f"{shot_id} MCP stayed responsive while Blender rendered",
                  float(job.get("_slowest_poll_s") or 0.0) < 10.0,
                  f"slowest poll {job.get('_slowest_poll_s')}s")
            finalized = (status or {}).get("result") or job.get("result") or {}

            check(f"{shot_id} satisfied the finalization contract",
                  finalized.get("final") is True,
                  str(finalized.get("failed") or finalized.get("reason"))[:200])
            check(f"{shot_id} frames released after promotion",
                  finalized.get("frames_released") is True,
                  str(finalized.get("frames_note") or "")[:160])

            video = Path(finalized.get("video", ""))
            check(f"{shot_id} video exists on disk",
                  video.is_file(), str(video))
            shots_out[shot_id] = {"video": video, "corrections": corrections_for_shot}

            shots = client.call("list_shots", {"project_id": project_id})
            shot_row = next((s for s in shots["shots"] if s["shot_id"] == shot_id), {})
            check(f"{shot_id} is marked FINAL",
                  shot_row.get("status") == "FINAL", str(shot_row)[:120])

        # -- all five shots are done ---------------------------------------
        storage_before = client.call("get_storage_status",
                                     {"project_id": project_id})
        frames_left = list((workspace / project_id / "shots").rglob("*.png"))
        check("no temporary frames remain after all five shots",
              not frames_left, f"{len(frames_left)} frame file(s) remain")

        # -- assemble + verify the final movie -----------------------------
        print("\n=== ASSEMBLY ===")
        assembled = client.call("assemble_movie", {"project_id": project_id})
        check("assemble_movie succeeded", not assembled.get("_is_error"),
              str(assembled)[:300])
        final_path = Path(assembled.get("final_movie", ""))
        check("final movie exists on disk", final_path.is_file(), str(final_path))
        check("final movie is non-zero",
              final_path.is_file() and final_path.stat().st_size > 0,
              f"{final_path.stat().st_size if final_path.is_file() else 0} bytes")
        check("final movie contains all five shots",
              (assembled.get("shots") or 0) == 5,
              str(assembled.get("shot_ids")))

        if final_path.is_file():
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration:stream=codec_name,width,height,r_frame_rate",
                 "-of", "json", str(final_path)],
                capture_output=True, text=True,
            )
            try:
                info_json = json.loads(probe.stdout or "{}")
                streams = info_json.get("streams") or [{}]
                duration = float((info_json.get("format") or {}).get("duration", 0))
                check("final movie ffprobes as H.264",
                      streams[0].get("codec_name") == "h264",
                      str(streams[0].get("codec_name")))
                check("final movie has the expected resolution",
                      f"{streams[0].get('width')}x{streams[0].get('height')}"
                      == f"{args.width}x{args.height}",
                      f"{streams[0].get('width')}x{streams[0].get('height')}")
                check("final movie duration is plausible",
                      abs(duration - args.duration * 5) < 1.5, f"{duration:.2f}s")
                fps = streams[0].get("r_frame_rate", "")
                print(f"        final: codec={streams[0].get('codec_name')} "
                      f"fps={fps} {duration:.2f}s "
                      f"{streams[0].get('width')}x{streams[0].get('height')}")
            except Exception as exc:
                check("final movie ffprobe parsed", False, str(exc))

        # -- filesystem integrity ------------------------------------------
        integrity = client.call("get_final_movie", {"project_id": project_id})
        check("final movie verified by the server",
              integrity.get("verified") is True, str(integrity)[:200])
        storage_after = client.call("get_storage_status",
                                    {"project_id": project_id})
        check("storage remeasured after assembly",
              isinstance(storage_after.get("total_bytes"), int),
              str(storage_after))
        print(f"        storage: {storage_before.get('total_gb')} GB -> "
              f"{storage_after.get('total_gb')} GB (state {storage_after.get('state')})")

    finally:
        client.close()

    total_corrections = sum(shot["corrections"] for shot in shots_out.values())
    print(f"\n  shots finalized: {len(shots_out)}/5   corrections: {total_corrections}")
    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

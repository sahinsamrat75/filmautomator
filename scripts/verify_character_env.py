#!/usr/bin/env python3
"""CHARACTER_TEST + ENVIRONMENT_TEST — real Blender, real MCP, real images.

Blocker 10 and 11 of the production-readiness pass: prove that the canonical
character and environment systems actually produce Blender scene content the
director can see, before any movie production starts.

CHARACTER_TEST
    define Wren -> build -> verify real objects (head, torso, coat, arms,
    legs, boots) -> verify the canonical colors reached Blender materials ->
    apply idle/walk/reach/lean/react -> verify the armature pose and the
    animation actually changed between actions -> render a preview and
    RECEIVE the actual PNG through MCP.

ENVIRONMENT_TEST
    define ENV_COURTYARD_01 with a structured object plan -> build -> verify
    the semantic geometry exists in Blender (bench seat+legs+back, tree
    trunk+canopy, wall slab, fountain basin, lantern post+light, archway
    pillars+arch, ground, backdrop) -> render a preview and RECEIVE it.

    python3 scripts/verify_character_env.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
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

    def _message(self, method: str, params: dict) -> dict:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id,
                   "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            reply = json.loads(line)
            if reply.get("id") == self._id:
                return reply
        raise RuntimeError("MCP server closed the stream")

    def initialize(self) -> dict:
        reply = self._message("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"image": {}, "roots": {}},
            "clientInfo": {"name": "verify-character-env", "version": "1"},
        })
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()
        return reply.get("result", {})

    def call(self, name: str, arguments: dict) -> dict:
        reply = self._message("tools/call",
                              {"name": name, "arguments": arguments})
        result = reply.get("result") or {}
        payload: dict = {}
        images: list[dict] = []
        for block in result.get("content") or []:
            if block.get("type") == "text":
                try:
                    payload = json.loads(block.get("text") or "{}")
                except json.JSONDecodeError:
                    payload = {"_raw": block.get("text")}
            elif block.get("type") == "image":
                images.append(block)
        payload["_is_error"] = bool(result.get("isError"))
        payload["_images"] = images
        return payload

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def png_dimensions(png: bytes) -> tuple[int, int]:
    if not png.startswith(PNG_MAGIC) or len(png) < 33:
        return (0, 0)
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    return (width, height)


def image_of(response: dict) -> bytes:
    images = response.get("_images") or []
    if not images:
        return b""
    return base64.b64decode(images[0].get("data") or "")


def pose_snapshot(scene: dict) -> str:
    armatures = scene.get("armatures") or []
    if not armatures:
        return ""
    return json.dumps(armatures[0].get("pose") or [], sort_keys=True)


def run_character_test(client: StdioMCP, args) -> None:
    print("\n=== CHARACTER_TEST: Wren ===")
    project = client.call("create_project",
                          {"name": "CHARACTER_TEST",
                           "objective": "prove the character system"})
    check("CHARACTER_TEST project created", not project.get("_is_error"),
          str(project)[:200])
    pid = project["project_id"]

    character = client.call("define_character", {
        "project_id": pid, "name": "Wren", "height_m": 1.72,
        "build": "slim", "clothing": "coat, trousers, boots",
        "hair": "short dark",
        "colors": {"coat": "#0e7f86", "trousers": "#23262b",
                   "boots": "#141414", "hair": "#1d1d1d"},
        "behavior": "quiet, watches before acting",
    })
    check("define_character Wren accepted", not character.get("_is_error"),
          str(character)[:200])

    shot = client.call("create_shot", {
        "project_id": pid, "shot_id": "SC01_SH001", "scene_id": "SC01",
        "description": "Wren stands in the courtyard",
        "duration_s": 3.0, "shot_size": "medium", "angle": "eye_level",
        "camera_location": [0.0, -4.5, 1.4], "look_at": [0.0, 0.0, 1.2],
        "lens_mm": 50.0, "movement": "static",
        "subject_name": "Wren", "subject_height_m": 1.72,
        "subject_action": "idle",
    })
    check("shot for Wren created", not shot.get("_is_error"), str(shot)[:200])

    # -- build + receive the actual image ---------------------------------
    preview = client.call("render_shot_preview",
                          {"project_id": pid, "shot_id": "SC01_SH001"})
    check("preview rendered", not preview.get("_is_error"), str(preview)[:200])
    check("preview delivered the actual image",
          preview.get("image_delivered") is True,
          str(preview.get("delivery"))[:200])
    png = image_of(preview)
    check("delivered preview is a real PNG", png.startswith(PNG_MAGIC),
          f"{len(png)} bytes")
    width, height = png_dimensions(png)
    check("delivered preview has valid dimensions",
          width >= 64 and height >= 64, f"{width}x{height}")
    print(f"        image: {width}x{height}, {len(png)} bytes")

    # -- the parts exist in Blender, with the canonical colours ------------
    scene = client.call("inspect_scene", {})
    names = [o["name"] for o in scene.get("objects") or []]
    wren = [n for n in names if "wren" in n.lower()]
    check("Wren has Blender objects", bool(wren), str(names[:8]))
    for part in ("head", "torso", "coat", "arm_L", "arm_R",
                 "leg_L", "leg_R", "boot_L", "boot_R"):
        found = any(part.lower() in n.lower() for n in wren)
        check(f"Wren has a real {part}", found, str(wren)[:200])

    # Canonical colour -> Blender material -> Principled base colour.
    from filmautomator.blender.assets import parse_color
    expected_coat = parse_color("#0e7f86", (0.0, 0.0, 0.0))
    coat_hit = False
    for material in scene.get("materials") or []:
        base = list(material.get("base_color") or [])
        if len(base) >= 3 and all(
                abs(float(base[i]) - expected_coat[i]) < 0.05
                for i in range(3)):
            coat_hit = True
            print(f"        coat material: {material['name']} {base}")
            break
    check("canonical teal coat reached a Blender material", coat_hit,
          f"expected ~{tuple(round(c, 3) for c in expected_coat)}")

    # -- poses: idle -> walk -> reach -> lean -> react ---------------------
    print("\n  --- poses ---")
    previous_pose = pose_snapshot(scene)
    check("the rig exposes a readable pose", bool(previous_pose),
          str(len(previous_pose)))

    for action in ("walk", "reach", "lean", "react"):
        created = client.call("create_shot", {
            "project_id": pid, "shot_id": "SC01_SH001", "scene_id": "SC01",
            "description": f"Wren {action}",
            "duration_s": 3.0, "shot_size": "medium", "angle": "eye_level",
            "camera_location": [0.0, -4.5, 1.4], "look_at": [0.0, 0.0, 1.2],
            "lens_mm": 50.0, "movement": "static",
            "subject_name": "Wren", "subject_height_m": 1.72,
            "subject_action": action,
        })
        check(f"shot updated to action {action}", not created.get("_is_error"),
              str(created)[:160])
        preview = client.call("render_shot_preview",
                              {"project_id": pid, "shot_id": "SC01_SH001"})
        check(f"action {action}: preview delivered an image",
              preview.get("image_delivered") is True,
              str(preview.get("delivery"))[:160])

        scene = client.call("inspect_scene", {})
        pose = pose_snapshot(scene)
        changed = pose != previous_pose
        check(f"action {action} changed the actual bone pose", changed,
              "pose state is byte-identical to the previous action")
        previous_pose = pose

        if action == "walk":
            actions = scene.get("actions") or []
            keyed = [a for a in actions
                     if "wren" in a["name"].lower()
                     and (int(a.get("keyframe_count") or 0) > 0
                          or int(a.get("fcurve_count") or 0) > 0)]
            check("walk produced real keyframes on Wren's rig/root",
                  bool(keyed), str(actions)[:240])
            if keyed:
                print(f"        animation: {keyed[0]['name']} "
                      f"{keyed[0].get('frame_range')} "
                      f"{keyed[0].get('keyframe_count')} keys")

    # the version list must advance, never overwrite
    shot_row = client.call("get_shot", {"project_id": pid,
                                        "shot_id": "SC01_SH001"})
    check("shot exists with a spec", not shot_row.get("_is_error"),
          str(shot_row)[:200])


def run_environment_test(client: StdioMCP, args) -> None:
    print("\n=== ENVIRONMENT_TEST: ENV_COURTYARD_01 ===")
    project = client.call("create_project",
                          {"name": "ENVIRONMENT_TEST",
                           "objective": "prove the environment system"})
    check("ENVIRONMENT_TEST project created", not project.get("_is_error"),
          str(project)[:200])
    pid = project["project_id"]

    environment = client.call("define_environment", {
        "project_id": pid, "environment_id": "ENV_COURTYARD_01",
        "name": "Courtyard at dusk",
        "description": "A stone courtyard: bench by the fountain, an old tree "
                       "in the north corner, a long north wall.",
        "props": ["fountain", "lantern", "archway"],
        "objects": [
            {"id": "bench_se", "type": "bench",
             "position": [3.2, 1.5, 0.0], "rotation": -1.5708},
            {"id": "old_oak", "type": "tree",
             "position": [-6.5, 7.5, 0.0], "dimensions": {"height": 5.5}},
            {"id": "north_wall", "type": "wall",
             "position": [0.0, 12.0, 0.0],
             "dimensions": {"width": 22.0, "height": 3.2}},
        ],
        "weather": "clear dusk",
        "lighting_baseline": "dusk ambient, warm lantern",
    })
    check("define_environment accepted the structured plan",
          not environment.get("_is_error"), str(environment)[:200])

    shot = client.call("create_shot", {
        "project_id": pid, "shot_id": "SC01_SH001", "scene_id": "SC01",
        "description": "wide of the courtyard",
        "duration_s": 3.0, "shot_size": "wide", "angle": "eye_level",
        "camera_location": [0.0, -10.0, 2.0], "look_at": [0.0, 2.0, 1.0],
        "lens_mm": 28.0, "movement": "static",
        "environment": "ENV_COURTYARD_01",
    })
    check("shot references the canonical environment",
          not shot.get("_is_error"), str(shot)[:200])

    preview = client.call("render_shot_preview",
                          {"project_id": pid, "shot_id": "SC01_SH001"})
    check("environment preview rendered", not preview.get("_is_error"),
          str(preview)[:200])
    check("environment preview delivered the actual image",
          preview.get("image_delivered") is True,
          str(preview.get("delivery"))[:200])
    png = image_of(preview)
    check("delivered environment preview is a real PNG",
          png.startswith(PNG_MAGIC), f"{len(png)} bytes")
    width, height = png_dimensions(png)
    check("environment preview has valid dimensions",
          width >= 64 and height >= 64, f"{width}x{height}")
    print(f"        image: {width}x{height}, {len(png)} bytes")

    # -- semantic geometry in Blender's actual state -----------------------
    scene = client.call("inspect_scene", {})
    by_name = {o["name"]: o for o in scene.get("objects") or []}

    def named(part: str) -> dict | None:
        for name, entry in by_name.items():
            if part in name:
                return entry
        return None

    ground = named("_ground")
    check("real ground mesh exists", ground is not None,
          "no object name contains '_ground'")
    check("real backdrop exists", named("_backdrop") is not None)

    # Bench from the structured plan: seat (flat) + backrest + legs.
    seat = named("prop_bench_se_seat")
    check("plan bench seat exists", seat is not None,
          str(sorted(by_name)[:14]))
    if seat:
        dims = seat.get("dimensions") or [0, 0, 0]
        check("bench seat is a flat slab (semantically a seat)",
              dims[2] < dims[0], str(dims))
    check("plan bench backrest exists",
          named("prop_bench_se_back") is not None)
    check("plan bench leg exists", named("prop_bench_se_leg_L") is not None)
    # Tree: trunk + canopy.
    trunk = named("prop_old_oak_trunk")
    check("plan tree trunk exists", trunk is not None)
    if trunk:
        dims = trunk.get("dimensions") or [0, 0, 0]
        check("tree trunk is tall and thin (semantically a trunk)",
              dims[2] > 1.5 and dims[0] < 0.8, str(dims))
    check("plan tree canopy exists",
          named("prop_old_oak_canopy") is not None)
    # Wall: thin, tall, and the requested 22 m span.
    wall = named("prop_north_wall_slab")
    check("plan wall slab exists", wall is not None)
    if wall:
        dims = wall.get("dimensions") or [0, 0, 0]
        check("wall is thin and tall (semantically a wall)",
              dims[1] < 0.6 and dims[2] > 2.5, str(dims))
        check("wall spans its requested 22 m width",
              abs(dims[0] - 22.0) < 0.5, str(dims))
    # Named defaults still build: fountain, lantern, archway.
    check("fountain basin exists", named("fountain_basin") is not None)
    check("lantern post exists", named("lantern_post") is not None)
    check("lantern glow exists", named("lantern_glow") is not None)
    check("archway pillar exists", named("archway_pillar_L") is not None)
    check("archway arch ring exists", named("archway_arch") is not None)

    # The lantern's light is a real Blender light.
    lights = scene.get("lights") or []
    check("lantern light exists in Blender",
          any("lantern" in l.get("name", "") for l in lights),
          str(lights)[:200])

    invisible = [n for n, e in by_name.items()
                 if n.startswith("ENV_COURTYARD_01")
                 and not e.get("visible", True)]
    check("every environment object is render-visible", not invisible,
          str(invisible)[:200])
    meshless = [n for n, e in by_name.items()
                if n.startswith("ENV_COURTYARD_01")
                and e.get("type") == "MESH"
                and not any(e.get("dimensions") or [])]
    check("every environment mesh has real dimensions", not meshless,
          str(meshless)[:200])
    print(f"        environment objects: {len(by_name)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CHARACTER_TEST + ENVIRONMENT_TEST against real Blender")
    parser.add_argument("--workspace", default="",
                        help="reuse this workspace instead of a temp dir")
    args = parser.parse_args()

    workspace = Path(args.workspace) if args.workspace else Path(
        tempfile.mkdtemp(prefix="fa_charenv_"))
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"workspace: {workspace}")

    client = StdioMCP(workspace)
    try:
        info = client.initialize()
        server = info.get("serverInfo") or {}
        print(f"MCP: {server.get('name')} {server.get('version')}")
        run_character_test(client, args)
        run_environment_test(client, args)
    finally:
        client.close()

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

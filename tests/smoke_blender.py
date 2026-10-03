"""End-to-end smoke test for the Blender control layer.

Unlike the rest of the suite this one needs Blender installed, so it is a
script rather than a pytest case. Run it directly:

    /opt/homebrew/bin/python3 tests/smoke_blender.py

It exercises the whole path an agent uses: launch, inspect, build, render,
measure the render, save, restart.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filmautomator.blender.session import BlenderSession  # noqa: E402
from filmautomator.config import load_config  # noqa: E402

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


def main() -> int:
    config = load_config()
    print(f"Blender: {config.blender.executable}")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="fa_smoke_"))
    session = BlenderSession(config.blender)

    try:
        print("startup")
        session.start()
        info = session.ping()
        check("ping returns a version", bool(info.get("blender_version")),
              str(info))
        print(f"        Blender {info.get('blender_version')}")

        print("scene state (observation channel A)")
        session.reset_scene(empty=True)
        state = session.scene_state()
        check("reset gives an empty scene", state["object_count"] == 0,
              f"{state['object_count']} objects")
        print(f"        engine options: {state['engine']}")

        print("construction")
        session.create_primitive("plane", name="Ground", size=50.0)
        session.create_primitive("cube", name="Box", location=(0, 0, 1))
        session.create_primitive("sphere", name="Ball", location=(2, 0, 1),
                                 radius=0.6)

        state = session.scene_state()
        names = {o["name"] for o in state["objects"]}
        check("three objects exist", {"Ground", "Box", "Ball"} <= names,
              str(sorted(names)))

        # The plane size bug: a 50m plane must be 50m, not the operator default.
        ground = next(o for o in state["objects"] if o["name"] == "Ground")
        check("plane honours its size argument",
              abs(ground["dimensions"][0] - 50.0) < 0.5,
              f"dimensions={ground['dimensions']}")

        # The sphere bug: radius=0.6 must give a 1.2m sphere, not the
        # operator's 2.0m default. A silently dropped radius is invisible in
        # the scene graph but wrecks every framing calculation downstream.
        ball = next(o for o in state["objects"] if o["name"] == "Ball")
        check("sphere honours its radius argument",
              abs(ball["dimensions"][0] - 1.2) < 0.05,
              f"dimensions={ball['dimensions']}")

        # A cube given size=3 must be 3m across.
        session.create_primitive("cube", name="SizedCube", size=3.0,
                                 location=(5, 0, 1))
        sized = next(o for o in session.scene_state()["objects"]
                     if o["name"] == "SizedCube")
        check("cube honours its size argument",
              abs(sized["dimensions"][0] - 3.0) < 0.05,
              f"dimensions={sized['dimensions']}")

        print("materials")
        session.create_material("Red", base_color=(0.8, 0.1, 0.1), roughness=0.4)
        session.assign_material("Red", ["Box", "Ball"])
        state = session.scene_state()
        box = next(o for o in state["objects"] if o["name"] == "Box")
        check("material assigned", box["materials"] == ["Red"], str(box["materials"]))

        print("lights and camera")
        session.create_light(name="Key", light_type="AREA", energy=2000.0,
                             location=(4, -4, 6), look_at=(0, 0, 1), size=3.0)
        session.create_camera(name="Cam", lens_mm=50.0, location=(0, -6, 2),
                              look_at=(0, 0, 1))
        state = session.scene_state()
        check("active camera set", state["active_camera"] == "Cam",
              str(state["active_camera"]))
        check("one light present", len(state["lights"]) == 1)

        print("render settings")
        session.set_scene_timing(frame_start=1, frame_end=24, fps=24)
        # EEVEE, because that is what the pipeline previews with. Workbench is
        # faster but renders monochrome headless (asserted below), which would
        # make the colour-based coverage measurement meaningless.
        result = session.set_render_settings(
            engine="BLENDER_EEVEE", resolution_x=320, resolution_y=180,
            samples=8, view_transform="AgX", world_color=(0.05, 0.06, 0.09),
        )
        check("render settings applied", result["resolution"] == [320, 180],
              str(result["resolution"]))
        print(f"        engine={result['engine']} view={result.get('view_transform')}"
              f" note={result.get('engine_note') or 'none'}")

        print("render (observation channel C)")
        preview = tmp / "preview.png"
        rendered = session.render_still(preview)
        check("preview written", preview.is_file() and rendered["bytes"] > 0,
              str(rendered))

        print("image analysis (the Vision Agent's measurement input)")
        metrics = session.call("analyze_image", filepath=str(preview))
        check("metrics have the right dimensions",
              metrics["width"] == 320 and metrics["height"] == 180,
              f"{metrics['width']}x{metrics['height']}")
        check("frame is not black", metrics["luma_mean"] > 0.02,
              f"luma_mean={metrics['luma_mean']}")
        check("frame is not blown out", metrics["luma_mean"] < 0.98,
              f"luma_mean={metrics['luma_mean']}")
        # The subject carries a red material; a colour-aware render must show it.
        check("subject is visible against the background",
              metrics["foreground_coverage"] > 0.01,
              f"coverage={metrics['foreground_coverage']}")
        print(f"        luma={metrics['luma_mean']:.3f} "
              f"std={metrics['luma_std']:.3f} "
              f"coverage={metrics['foreground_coverage']:.1%} "
              f"edges={metrics['edge_density']:.4f}")
        check("histogram has 16 bins", len(metrics["histogram"]) == 16)

        print("Workbench limitation is reported, not silent")
        wb = session.set_render_settings(engine="BLENDER_WORKBENCH")
        note = (wb.get("engine_note") or "").lower()
        check("Workbench report warns it renders monochrome",
              "workbench" in note and "monochrome" in note, repr(note))
        wb_preview = tmp / "wb.png"
        session.render_still(wb_preview)
        wb_metrics = session.call("analyze_image", filepath=str(wb_preview))
        rgb = wb_metrics["mean_rgb"]
        check("Workbench output really is neutral grey",
              max(rgb) - min(rgb) < 0.02, f"mean_rgb={rgb}")
        session.set_render_settings(engine="BLENDER_EEVEE")

        print("animation render (observation channel D)")
        frames = session.render_animation(tmp / "seq", frame_start=1, frame_end=4)
        check("all frames rendered", frames["frame_count"] == 4,
              str(frames))

        print("integrity check (observation channel E)")
        problems = session.scene_errors()
        check("no integrity problems on a healthy scene",
              problems["count"] == 0, str(problems["problems"]))

        print("save / load")
        blend = tmp / "scene.blend"
        saved = session.save_blend(blend)
        check("blend saved", blend.is_file() and saved["bytes"] > 0, str(saved))
        session.reset_scene(empty=True)
        check("scene cleared", session.scene_state()["object_count"] == 0)
        session.load_blend(blend)
        reloaded = session.scene_state()
        check("blend reloaded with its objects", reloaded["object_count"] >= 4,
              f"{reloaded['object_count']} objects")

        print("error reporting for a bad operation")
        try:
            session.set_transform("NoSuchObject", location=(1, 1, 1))
            check("bad op raises BlenderError", False, "no exception raised")
        except Exception as exc:  # noqa: BLE001
            check("bad op raises a typed error",
                  "NoSuchObject" in str(exc), str(exc))

        print("crash recovery (spec section 19)")
        session._proc.kill()  # noqa: SLF001 - deliberately simulating a crash
        import time
        time.sleep(0.5)
        recovered = session.ping()
        check("session recovered after a kill", bool(recovered.get("blender_version")),
              str(recovered))
        check("restart was counted", session.restart_count >= 1,
              str(session.restart_count))

    finally:
        session.stop()

    print()
    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

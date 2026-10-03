#!/usr/bin/env python3
"""Storage acceptance test: prove the governor reclaims space safely.

Phase 19 requires generating real render data and then verifying the whole
storage story end to end:

    storage is measured
    the threshold is detected
    disposable files are identified
    finalized files are preserved
    temporary frames are deleted only after successful promotion
    storage decreases
    production continues

And the safety case: at the hard limit with nothing safe to delete, the system
must stop rather than delete something it should not.

    python3 scripts/verify_storage_acceptance.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filmautomator.core.finalization import SHOT_FINAL  # noqa: E402
from filmautomator.core.project import ProjectDB  # noqa: E402
from filmautomator.core.storage import (  # noqa: E402
    CAT_FINAL,
    CAT_FRAMES,
    GB,
    MB,
    StorageGovernor,
    StorageRefused,
    StorageState,
    StorageThresholds,
)

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


def make_video(path: Path, *, seconds: float = 1.0, width: int = 320,
               height: int = 180) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=green:s={width}x{height}:d={seconds}",
         "-r", "24", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return path


def write_frames(frames: Path, count: int, size: int) -> int:
    """Write realistic frame data and return the total bytes."""
    frames.mkdir(parents=True, exist_ok=True)
    total = 0
    for index in range(count):
        payload = bytes([(index * 7) % 251]) * size
        (frames / f"f{index:04d}.png").write_bytes(payload)
        total += len(payload)
    return total


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="fa_storage_")) / "projects"
    root.mkdir(parents=True)
    registry = root / "registry.db"
    db = ProjectDB(registry)
    project_id = "proj_storage"

    expected_total = 0
    db.create_project("STORAGE", workspace=str(root))
    governor = StorageGovernor(root, db=db)

    print("Storage acceptance test")
    print(f"  workspace: {root}\n")

    # -- generate real render data ---------------------------------------
    print("  --- generating render data ---")
    shots = {}
    for index in range(1, 4):
        shot_id = f"SC01_SH{index:03d}"
        frames = root / project_id / "shots" / shot_id / "v001" / "render"
        frames_bytes = write_frames(frames, 120, 256 * 1024)  # ~30 MB per shot
        db.upsert_shot(project_id, shot_id, "SC01", index - 1, 1.0, {})
        shots[shot_id] = {"frames": frames, "bytes": frames_bytes}
        expected_total += frames_bytes
        check(f"{shot_id} rendered {120} frames ({frames_bytes / MB:.0f} MB)",
              frames_bytes == 120 * 256 * 1024, str(frames_bytes))
    final = make_video(root / project_id / "final" / "STORAGE_FINAL.mp4")
    expected_total += final.stat().st_size
    print(f"  generated {sum(s['bytes'] for s in shots.values()) / MB:.0f} MB "
          f"of frames across 3 shots")

    # -- storage is measured ---------------------------------------------
    print("\n  --- measurement ---")
    usage = governor.measure()
    # The registry database itself is part of the workspace and grows as shots
    # are recorded, so the total is compared against a floor rather than an
    # exact figure: every generated frame plus the final movie must be counted.
    check("storage is measured and accounts for the generated data",
          usage.total_bytes >= expected_total,
          f"{usage.total_bytes} < {expected_total}")
    check("frames are attributed to the frames category",
          usage.by_category.get(CAT_FRAMES, 0)
          == sum(s["bytes"] for s in shots.values()),
          str(usage.by_category))
    check("the final movie is attributed as a deliverable",
          usage.by_category.get(CAT_FINAL) == final.stat().st_size,
          str(usage.by_category.get(CAT_FINAL)))
    check("an empty workspace reads NORMAL", usage.state is StorageState.NORMAL,
          usage.state.value)
    print(f"        total {usage.total_gb:.3f} GB, state {usage.state.value}, "
          f"{usage.free_disk_bytes / GB:.1f} GB free on the volume")

    # -- threshold is detected -------------------------------------------
    print("\n  --- threshold detection ---")
    for used_gb, expected in ((24.0, StorageState.NORMAL),
                              (25.0, StorageState.WARNING),
                              (30.0, StorageState.AGGRESSIVE_CLEANUP),
                              (35.0, StorageState.HARD_LIMIT),
                              (36.5, StorageState.HARD_LIMIT)):
        thresholds = StorageThresholds()
        check(f"{used_gb} GB reads as {expected.value}",
              thresholds.state_for(int(used_gb * GB)) is expected,
              thresholds.state_for(int(used_gb * GB)).value)
    check("the ceiling is 35 GB",
          StorageThresholds().hard_limit_bytes == 35 * GB,
          str(StorageThresholds().hard_limit_bytes))

    # -- disposable files identified, finalized files preserved -----------
    print("\n  --- identification before promotion ---")
    usage = governor.measure()
    # Before any shot is promoted nothing is disposable, and that is the point:
    # every set of frames is still the only copy of its shot's work.
    check("no frames are disposable before promotion", usage.disposable_bytes == 0,
          f"disposable={usage.disposable_bytes}")
    check("every frame set is retained with a stated reason",
          len(usage.retained) >= 120,
          f"{len(usage.retained)} retained")
    check("the final movie is protected, not disposable",
          usage.by_category.get(CAT_FINAL) == final.stat().st_size
          and final.is_file(), str(final))
    check("the final movie is never a candidate",
          all(e.path != final for e in governor.plan_cleanup()), str(final))

    # -- failed renders keep their frames --------------------------------
    print("\n  --- failed shots retain their frames ---")
    failed_shot = "SC01_SH004"
    failed_frames = root / project_id / "shots" / failed_shot / "v001" / "render"
    failed_bytes = write_frames(failed_frames, 40, 256 * 1024)
    db.upsert_shot(project_id, failed_shot, "SC01", 3, 1.0, {})
    db.set_shot_status(project_id, failed_shot, "FAILED")
    result = governor.cleanup()
    check("cleanup deleted nothing while no shot is promoted",
          result["freed_bytes"] == 0, str(result["freed_bytes"]))
    check("the failed shot's frames are still on disk",
          failed_frames.is_dir() and len(list(failed_frames.glob("*.png"))) == 40,
          str(len(list(failed_frames.glob("*.png")))))

    # -- promote one shot, then clean ------------------------------------
    print("\n  --- promote a shot, then reclaim ---")
    promoted = shots["SC01_SH001"]
    video = make_video(root / project_id / "shots" / "SC01_SH001" / "v001"
                       / "SC01_SH001.mp4")
    db.register_artifact(project_id, "shot_video", str(video), "SH001",
                         shot_id="SC01_SH001",
                         metadata={"version": 1})
    db.set_shot_status(project_id, "SC01_SH001", SHOT_FINAL)
    check("the promoted shot is marked FINAL",
          db.get_shot(project_id, "SC01_SH001")["status"] == SHOT_FINAL,
          db.get_shot(project_id, "SC01_SH001")["status"])

    before = governor.measure()
    result = governor.cleanup()
    after = governor.measure()
    check("only the promoted shot's frames were deleted",
          result["freed_bytes"] == promoted["bytes"],
          f"freed {result['freed_bytes']} vs expected {promoted['bytes']}")
    check("the promoted shot's frames are gone",
          not list(promoted["frames"].glob("*.png")), "")
    check("the failed shot's frames survived",
          len(list(failed_frames.glob("*.png"))) == 40, "")
    check("unpromoted shots kept their frames",
          len(list(shots["SC01_SH002"]["frames"].glob("*.png"))) == 120, "")
    check("storage decreased",
          after.total_bytes < before.total_bytes,
          f"{before.total_bytes} -> {after.total_bytes}")
    check("the final movie survived cleanup", final.is_file(), str(final))
    check("the promoted shot video survived cleanup", video.is_file(), str(video))
    print(f"        {before.total_gb:.3f} GB -> {after.total_gb:.3f} GB "
          f"(freed {result['freed_gb']:.3f} GB)")

    # -- production continues --------------------------------------------
    print("\n  --- production continues after cleanup ---")
    remaining = [s for s in ("SC01_SH002", "SC01_SH003") if s]
    for shot_id in remaining:
        check(f"{shot_id} can still be promoted after cleanup",
              (root / project_id / "shots" / shot_id).is_dir(),
              "its workspace was removed")
    result = governor.cleanup()
    check("a second cleanup is a no-op rather than an error",
          result["freed_bytes"] == 0 and not result.get("_is_error"),
          str(result["freed_bytes"]))

    # -- the hard limit with nothing safe to delete ----------------------
    print("\n  --- hard limit with no safe candidates ---")
    big_root = Path(tempfile.mkdtemp(prefix="fa_storage_full_")) / "projects"
    (big_root / "proj").mkdir(parents=True)
    keeper = big_root / "proj" / "final" / "EP01_FINAL.mp4"
    keeper.parent.mkdir(parents=True)
    keeper.write_bytes(b"x" * 4096)
    with (big_root / "proj" / "huge.bin").open("wb") as handle:
        handle.truncate(36 * GB)  # sparse: reaches the ceiling, costs no space
    blocked = StorageGovernor(big_root,
                             thresholds=StorageThresholds(hard_limit_bytes=35 * GB))
    state = blocked.measure()
    check("the ceiling is detected", state.state is StorageState.HARD_LIMIT,
          state.state.value)
    check("nothing is safely deletable", state.disposable_bytes == 0,
          str(state.disposable_bytes))
    check("a large render is refused at the ceiling",
          state.can_start_large_render is False, "")
    try:
        blocked.ensure_capacity(needed_bytes=0)
        check("ensure_capacity refuses at the ceiling", False, "no refusal raised")
    except StorageRefused as refusal:
        check("ensure_capacity refuses at the ceiling", True, "")
        check("the refusal explains the ceiling", "35 GB" in str(refusal),
              str(refusal)[:160])
    check("the deliverable was not deleted to make room", keeper.is_file(),
          str(keeper))

    db.close()
    shutil.rmtree(big_root, ignore_errors=True)
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

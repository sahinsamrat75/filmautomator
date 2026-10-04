"""Builders must produce real, verifiable scene state — not metadata.

Phase 11 of the builders milestone. Covers the parts of the builders that can
be tested without rendering: canonical records compile into deterministic
plans, preview versions advance v001->v002->v003 instead of overwriting, and
the object-manifest resolver maps a canonical scope to real Blender object
names. The rendering-side integration lives in tests/smoke_blender.py and
scripts/verify_ai_director.py, which drive a real Blender.
"""

from __future__ import annotations

from pathlib import Path

from filmautomator.blender.assets import (
    POSES,
    build_environment,
    build_character,
    parse_color,
)
from filmautomator.agents.blender_agent import subject_object_names
from filmautomator.core.spec import CameraSpec, ShotSpec, SubjectSpec


class _FakeSession:
    """A scripted BlenderSession double that records the operations a build
    issues, so the builder logic can be asserted without a live Blender."""

    def __init__(self) -> None:
        self.records: list[str] = []
        self.materials: dict[str, dict] = {}
        self.objects: list[dict] = []
        self._next_name = 0

    def create_primitive(self, kind: str, **kwargs) -> dict:
        self.records.append(f"primitive:{kind}:{kwargs.get('name')}")
        name = kwargs.get("name") or f"obj_{self._next_name}"
        self._next_name += 1
        entry = {"name": name, "type": "MESH", "kind": kind, **kwargs}
        self.objects.append(entry)
        return {"name": name}

    def create_empty(self, name: str, **kwargs) -> dict:
        self.records.append(f"empty:{name}")
        self.objects.append({"name": name, "type": "EMPTY", **kwargs})
        return {"name": name}

    def create_material(self, name: str, **kwargs) -> dict:
        self.records.append(f"material:{name}")
        self.materials[name] = kwargs
        return {"name": name}

    def assign_material(self, material: str, names: list[str]) -> dict:
        self.records.append(f"assign:{material}:{sorted(names)}")
        return {"assigned": names}

    def create_light(self, name: str, **kwargs) -> dict:
        self.records.append(f"light:{name}")
        self.objects.append({"name": name, "type": "LIGHT", **kwargs})
        return {"name": name}

    def create_armature(self, name: str, bones: list[dict], **kwargs) -> dict:
        self.records.append(f"armature:{name}:{len(bones)}bones")
        self.objects.append({"name": name, "type": "ARMATURE", **kwargs})
        return {"name": name, "bones": [b["name"] for b in bones]}

    def set_transform(self, name: str, **kwargs) -> dict:
        self.records.append(f"transform:{name}:{sorted(kwargs)}")
        return {"name": name}

    def pose_bone(self, armature: str, bone: str, **kwargs) -> dict:
        self.records.append(f"pose:{armature}:{bone}:{sorted(kwargs)}")
        return {"armature": armature, "bone": bone}


# ---------------------------------------------------------------------------
# Colour parsing (canonical colours reach builder inputs)
# ---------------------------------------------------------------------------


def test_parse_color_accepts_hex_named_and_rgb():
    assert parse_color("#ff0000", (0, 0, 0)) == (1.0, 0.0, 0.0)
    assert parse_color("teal", (0, 0, 0)) == (0.1, 0.45, 0.45)
    assert parse_color([0.5, 0.25, 0.1], (0, 0, 0)) == (0.5, 0.25, 0.1)
    # Unknown strings fall back rather than failing the build.
    assert parse_color("not-a-real-color", (0.2, 0.3, 0.4)) == (0.2, 0.3, 0.4)


# ---------------------------------------------------------------------------
# A. Environment builder creates actual objects
# ---------------------------------------------------------------------------


def test_environment_builder_creates_ground_backdrop_and_fountain():
    session = _FakeSession()
    built = build_environment(
        session, "ENV_COURTYARD_DUSK_01",
        {"props": ["fountain", "lantern", "archway"], "weather": "clear dusk"},
    )
    names = set(built.objects)
    assert any("_ground" in n for n in names), names
    assert any("_backdrop" in n for n in names), names
    assert any("fountain_basin" in n for n in names), names
    assert any("lantern_housing" in n for n in names), names
    assert any("archway_lintel" in n for n in names), names
    # The fountain material exists and the parts reference real geometry.
    assert any("MAT_stone" in m for m in built.materials), built.materials
    assert any("lantern_light" in n for n in built.lights), built.lights


# ---------------------------------------------------------------------------
# C. Semantic prop geometry: the type decides the shape
# ---------------------------------------------------------------------------


def _prop_names(built) -> dict[str, list[str]]:
    """Map each prop id in the manifest to the objects it built."""
    return {pid: entry["objects"] for pid, entry in built.props.items()}


def test_free_text_props_get_semantic_geometry_not_identical_cubes():
    """The reported defect: bench/tree/wall all became the same box."""
    session = _FakeSession()
    built = build_environment(
        session, "ENV_TEST_01",
        {"props": ["bench", "tree", "wall"],
         "layout": "", "description": "courtyard with bench, tree and wall"},
    )
    manifest = _prop_names(built)
    bench = manifest.get("bench", [])
    tree = manifest.get("tree", [])
    wall = manifest.get("wall", [])

    assert bench, f"bench not built: {manifest.keys()}"
    assert tree, f"tree not built: {manifest.keys()}"
    assert wall, f"wall not built: {manifest.keys()}"

    # Bench: seat + backrest + legs, no seat-without-backrest box.
    assert any("_seat" in n for n in bench), bench
    assert any("_back" in n for n in bench), bench
    assert any("_leg" in n for n in bench), bench
    # Tree: trunk + canopy, built from different primitives.
    assert any("_trunk" in n for n in tree), tree
    assert any("_canopy" in n for n in tree), tree
    kinds = {o["kind"] for o in session.objects
             if o["name"] in set(tree)}
    assert "cylinder" in kinds and "sphere" in kinds, kinds
    # Wall: a thin, tall slab with a cap.
    assert any("_slab" in n for n in wall), wall
    assert any("_cap" in n for n in wall), wall
    slab = next(o for o in session.objects if o["name"] == wall[0])
    sx, sy, sz = slab["scale"]
    assert sy < sx, f"wall slab is not thin: {slab['scale']}"
    assert sz > sy, f"wall slab is not tall: {slab['scale']}"

    # Every prop group differs from every other: no shared shape.
    assert set(bench) != set(tree) != set(wall)


def test_the_structured_object_plan_drives_placement():
    """The plan is authoritative; free text is only context."""
    session = _FakeSession()
    built = build_environment(
        session, "ENV_PLAN_01",
        {
            "objects": [
                {"id": "seat_by_fountain", "type": "bench",
                 "position": [2.5, 3.0, 0.0], "rotation": 1.5708},
                {"id": "old_oak", "type": "tree",
                 "position": [-5.0, 6.0, 0.0],
                 "dimensions": {"height": 6.0}},
                {"id": "north_wall", "type": "wall",
                 "position": [0.0, 10.0, 0.0],
                 "dimensions": {"width": 14.0, "height": 3.5}},
            ],
        },
    )
    manifest = _prop_names(built)
    assert set(manifest) == {"seat_by_fountain", "old_oak", "north_wall"}

    seat = next(o for o in session.objects
                if o["name"] == f"ENV_PLAN_01_prop_seat_by_fountain_seat")
    assert tuple(seat["location"]) == (2.5, 3.0, 0.46), seat["location"]
    assert abs(seat["rotation"][2] - 1.5708) < 1e-6

    trunk = next(o for o in session.objects
                 if o["name"] == "ENV_PLAN_01_prop_old_oak_trunk")
    assert tuple(trunk["location"][0:2]) == (-5.0, 6.0)

    slab = next(o for o in session.objects
                if o["name"] == "ENV_PLAN_01_prop_north_wall_slab")
    # cube primitive scales are half-extents: 14m wide, 3.5m tall.
    assert abs(slab["scale"][0] - 7.0) < 1e-6, slab["scale"]
    assert abs(slab["scale"][2] - 1.75) < 1e-6, slab["scale"]

    # The plan is recorded in the build result so the manifest is inspectable.
    assert built.to_dict()["props"]["north_wall"]["type"] == "wall"
    assert built.to_dict()["props"]["north_wall"]["position"] == [0.0, 10.0, 0.0]


def test_a_plan_entry_supersedes_the_default_named_prop():
    """If the plan already carries a fountain, do not build a second one."""
    session = _FakeSession()
    built = build_environment(
        session, "ENV_PLAN_02",
        {"objects": [{"id": "main_fountain", "type": "fountain",
                      "position": [1.0, 2.0, 0.0]}],
         "props": ["fountain"]},
    )
    basins = [o["name"] for o in session.objects if "fountain_basin" in o["name"]]
    assert len(basins) == 1, f"expected exactly one fountain basin: {basins}"


def test_a_free_text_prop_name_lands_in_the_manifest():
    session = _FakeSession()
    built = build_environment(
        session, "ENV_MANIFEST_01",
        {"props": ["barrel"], "description": "a barrel by the door"},
    )
    manifest = built.to_dict()["props"]
    assert "barrel" in manifest
    entry = manifest["barrel"]
    assert entry["type"] == "barrel"
    assert entry["source"] == "free_text_props"
    assert any("_body" in n for n in entry["objects"]), entry


# ---------------------------------------------------------------------------
# B. Character builder creates real geometry
# ---------------------------------------------------------------------------


def test_character_builder_creates_segmented_geometry_with_materials():
    session = _FakeSession()
    built = build_character(
        session, "CHR_TRAVELER", "Traveler",
        {"height_m": 1.8, "clothing": "coat, trousers, boots",
         "colors": {"coat": "teal", "trousers": "#2b2b2b", "boots": "dark"}},
        location=(0.0, 0.0, 0.0),
    )
    parts = {str(p) for p in built.parts}
    assert any("head" in p and "hair" not in p for p in parts), parts
    assert any("torso" in p for p in parts), parts
    assert any("coat" in p for p in parts), parts
    assert any("arm" in p for p in parts), parts
    assert any("leg" in p for p in parts), parts
    assert any("boot" in p for p in parts), parts
    # Canonical colours reached materials.
    material_names = {m.split("_MAT_")[-1] for m in built.materials}
    assert "coat" in material_names, material_names
    assert "boots" in material_names, material_names
    # An armature drives the pose system.
    assert built.armature
    assert len(built.bones) >= 8, built.bones


def test_pose_vocabulary_covers_the_shot_actions():
    assert set(POSES) >= {"idle", "walk", "enter", "reach", "lean", "react"}


# ---------------------------------------------------------------------------
# C. Canonical colours reach Blender materials
# ---------------------------------------------------------------------------


def test_character_material_colors_come_from_the_canonical_record():
    session = _FakeSession()
    build_character(
        session, "C1", "Akira",
        {"height_m": 1.7, "colors": {"coat": "#2b3a55", "skin": "#c68a5b"}},
    )
    # Verify the material was created with the parsed base colour.
    coats = [m for m in session.materials if m.endswith("_MAT_coat")]
    assert coats, session.materials
    coat = session.materials[coats[0]]
    assert coat["base_color"] == (43 / 255, 58 / 255, 85 / 255), coat


# ---------------------------------------------------------------------------
# D/E. Camera and animation are spec-driven (unit side)
# ---------------------------------------------------------------------------


def test_subject_object_names_resolves_canonical_parts():
    from filmautomator.core.spec import CameraSpec, SubjectSpec

    spec = ShotSpec(
        shot_id="S1", scene_id="SC01", duration_s=1.0,
        camera=CameraSpec(shot_size="medium"),
        subjects=[SubjectSpec(name="Figure", height_m=1.8, action="walk")],
    )
    names = subject_object_names(spec)
    # No record -> legacy names, still renderable.
    assert "Figure_body" in names and "Figure_head" in names

    characters = {"Figure": {
        "parts": ["CHR_Figure_head", "CHR_Figure_torso", "CHR_Figure_arm_L"],
        "root": "CHR_Figure_root", "armature": "CHR_Figure_rig",
    }}
    resolved = subject_object_names(spec, characters)
    assert set(resolved) == {"CHR_Figure_head", "CHR_Figure_torso",
                             "CHR_Figure_arm_L", "CHR_Figure_root",
                             "CHR_Figure_rig"}


# ---------------------------------------------------------------------------
# F. Preview versioning advances instead of overwriting
# ---------------------------------------------------------------------------


def test_next_preview_version_advances_with_registered_previews(tmp_path):
    from filmautomator.core.project import ProjectDB

    db = ProjectDB(tmp_path / "p.db")
    project_id = db.create_project("V", workspace=str(tmp_path))
    db.upsert_shot(project_id, "SC01_SH001", "SC01", 0, 1.0, {})
    assert db.next_preview_version(project_id, "SC01_SH001") == 1
    db.register_artifact(project_id, "preview", "/tmp/v001.png", "p1",
                         shot_id="SC01_SH001", metadata={"version": 1})
    assert db.next_preview_version(project_id, "SC01_SH001") == 2
    db.register_artifact(project_id, "preview", "/tmp/v002.png", "p2",
                         shot_id="SC01_SH001", metadata={"version": 2})
    assert db.next_preview_version(project_id, "SC01_SH001") == 3


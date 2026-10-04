"""Canonical asset builders -- real Blender geometry from AI descriptions.

This module is the answer to the gap where ``define_environment()`` and
``define_character()`` recorded metadata but built nothing. A canonical
description now compiles into actual Blender objects:

* the environment builder turns ground/backdrop/fountain/lantern/archway/layout
  records into meshes, lights and materials;
* the character builder turns proportions/clothing/colors records into a
  segmented humanoid with a real armature, named parts and canonical materials;
* both are *canonical*: built once into a library .blend, then instanced by
  every shot, so the Traveler is the same Traveler in SH001 through SH005.

Procedural primitives, not external assets -- but real geometry that is
unmistakably present in the render, never metadata pretending to be a scene.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------


def parse_color(value: Any,
                default: tuple[float, float, float]
                ) -> tuple[float, float, float]:
    """Best-effort RGB from a canonical colour string.

    Accepts #rrggbb, #rgb, named basics, or [r, g, b] lists. Anything else falls
    back to the default rather than failing the whole build -- a wrong colour is
    a correctable preview, a failed build is nothing.
    """
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        named = {
            "black": (0.02, 0.02, 0.02), "white": (0.9, 0.9, 0.88),
            "grey": (0.4, 0.4, 0.4), "gray": (0.4, 0.4, 0.4),
            "red": (0.7, 0.1, 0.1), "green": (0.15, 0.45, 0.2),
            "blue": (0.15, 0.3, 0.65), "teal": (0.1, 0.45, 0.45),
            "brown": (0.32, 0.2, 0.12), "tan": (0.55, 0.42, 0.3),
            "orange": (0.75, 0.4, 0.12), "yellow": (0.7, 0.6, 0.2),
            "stone": (0.45, 0.44, 0.42), "dark": (0.08, 0.08, 0.09),
            "skin": (0.72, 0.52, 0.4),
        }
        lowered = value.strip().lower()
        if lowered in named:
            return named[lowered]
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) == 6:
            try:
                return (int(text[0:2], 16) / 255.0,
                        int(text[2:4], 16) / 255.0,
                        int(text[4:6], 16) / 255.0)
            except ValueError:
                return default
    return default


def _wants(props: list, spec: dict, name: str, default: bool) -> bool:
    """Does the record ask for this part? Defaults to building it."""
    lowered = [str(p).lower() for p in props]
    text = " ".join([str(spec.get("description") or ""),
                     str(spec.get("geometry") or ""),
                     str(spec.get("name") or "")]).lower()
    if name in lowered or name in text:
        return True
    if any(("no " + name) in entry or ("without " + name) in entry
           for entry in lowered):
        return False
    return default


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentBuild:
    """What was built into Blender for one canonical environment."""

@dataclass
class EnvironmentBuild:
    """What was built into Blender for one canonical environment."""

    environment_id: str
    scope: str = ""
    objects: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    lights: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Per-prop manifest from the structured object plan:
    #: ``{prop_id: {"type": ..., "position": [...], "objects": [...]}}``.
    props: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "scope": self.scope,
            "objects": self.objects,
            "materials": self.materials,
            "lights": self.lights,
            "notes": self.notes,
            "props": self.props,
        }


def _env_material(session: Any, name: str,
                  color: tuple[float, float, float],
                  roughness: float = 0.9,
                  emission: tuple[float, float, float] | None = None,
                  emission_strength: float = 0.0) -> str:
    kwargs: dict[str, Any] = {"base_color": color, "roughness": roughness}
    if emission is not None:
        kwargs["emission_color"] = emission
        kwargs["emission_strength"] = emission_strength
    session.create_material(name, **kwargs)
    return name


# ---------------------------------------------------------------------------
# Semantic prop geometry
#
# A prop's type decides its shape: a bench is a seat on legs with a backrest,
# a tree is a trunk under a canopy, a wall is a thin tall slab. The old
# builder emitted an identical cube for every unknown prop, which made every
# courtyard read as a row of boxes; these builders exist so the recorded
# environment actually influences the rendered image.
# ---------------------------------------------------------------------------

_PROP_PALETTE: dict[str, tuple[float, float, float]] = {
    "wood": (0.32, 0.20, 0.11),
    "trunk": (0.23, 0.15, 0.09),
    "foliage": (0.10, 0.28, 0.08),
    "metal": (0.42, 0.41, 0.40),
    "plaster": (0.62, 0.58, 0.52),
    "rock": (0.36, 0.35, 0.33),
}


def _rot_xy(x: float, y: float, dx: float, dy: float,
            rz: float) -> tuple[float, float]:
    """Offset (dx, dy) rotated by rz around (x, y)."""
    if not rz:
        return (x + dx, y + dy)
    c, s = math.cos(rz), math.sin(rz)
    return (x + c * dx - s * dy, y + s * dx + c * dy)


def _part_rotation(rz: float) -> tuple[float, float, float]:
    return (0.0, 0.0, rz) if rz else (0.0, 0.0, 0.0)


def _build_bench(session: Any, prefix: str, x: float, y: float, z: float,
                 rz: float, width: float) -> list[str]:
    """Seat on legs with a backrest — never a bare box."""
    half = width / 2.0
    parts: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
        # seat slab, backrest on end posts, two legs — (dx, dy, dz), scale
        ("seat", (0.0, 0.0, 0.46), (half, 0.30, 0.03)),
        ("back", (0.0, -0.26, 0.72), (half, 0.035, 0.26)),
        ("back_post_L", (-half * 0.8, -0.26, 0.58), (0.035, 0.035, 0.14)),
        ("back_post_R", (half * 0.8, -0.26, 0.58), (0.035, 0.035, 0.14)),
        ("leg_L", (-half * 0.8, 0.0, 0.22), (0.05, 0.26, 0.22)),
        ("leg_R", (half * 0.8, 0.0, 0.22), (0.05, 0.26, 0.22)),
    ]

    names = []
    for suffix, (dx, dy, dz), scale in parts:
        px, py = _rot_xy(x, y, dx, dy, rz)
        name = f"{prefix}_{suffix}"
        session.create_primitive("cube", name=name,
                                 location=(px, py, z + dz),
                                 rotation=_part_rotation(rz), scale=scale)
        names.append(name)
    return names


def _build_tree(session: Any, prefix: str, x: float, y: float, z: float,
                height: float) -> list[str]:
    """Trunk under an irregular canopy."""
    trunk_h = max(1.2, height * 0.55)
    trunk = f"{prefix}_trunk"
    session.create_primitive("cylinder", name=trunk,
                             location=(x, y, z + trunk_h / 2.0),
                             radius=max(0.08, height * 0.055))
    canopy_r = max(0.5, height * 0.35)
    top = z + trunk_h + canopy_r * 0.55
    canopy = f"{prefix}_canopy"
    session.create_primitive("sphere", name=canopy,
                             location=(x, y, top), radius=canopy_r)
    side_a = f"{prefix}_canopy_a"
    session.create_primitive("sphere", name=side_a,
                             location=(x - canopy_r * 0.55, y + canopy_r * 0.2,
                                       top - canopy_r * 0.25),
                             radius=canopy_r * 0.62)
    side_b = f"{prefix}_canopy_b"
    session.create_primitive("sphere", name=side_b,
                             location=(x + canopy_r * 0.5, y - canopy_r * 0.25,
                                       top - canopy_r * 0.35),
                             radius=canopy_r * 0.55)
    return [trunk, canopy, side_a, side_b]


def _build_wall(session: Any, prefix: str, x: float, y: float, z: float,
                rz: float, width: float, height: float) -> list[str]:
    """A real wall: thin, long, tall — oriented by rotation."""
    name = f"{prefix}_slab"
    session.create_primitive(
        "cube", name=name,
        location=(x, y, z + height / 2.0), rotation=_part_rotation(rz),
        scale=(width / 2.0, 0.18, height / 2.0),
    )
    cap = f"{prefix}_cap"
    session.create_primitive(
        "cube", name=cap,
        location=(x, y, z + height + 0.06), rotation=_part_rotation(rz),
        scale=(width / 2.0 + 0.06, 0.24, 0.06),
    )
    return [name, cap]


def _build_column(session: Any, prefix: str, x: float, y: float, z: float,
                  height: float) -> list[str]:
    name = f"{prefix}_shaft"
    session.create_primitive("cylinder", name=name,
                             location=(x, y, z + height / 2.0),
                             radius=0.22)
    base = f"{prefix}_base"
    session.create_primitive("cube", name=base,
                             location=(x, y, z + 0.12), scale=(0.32, 0.32, 0.12))
    cap = f"{prefix}_cap"
    session.create_primitive("cube", name=cap,
                             location=(x, y, z + height + 0.1),
                             scale=(0.36, 0.36, 0.1))
    return [name, base, cap]


def _build_crate(session: Any, prefix: str, x: float, y: float, z: float,
                 rz: float, size: float) -> list[str]:
    name = f"{prefix}_box"
    session.create_primitive("cube", name=name,
                             location=(x, y, z + size / 2.0),
                             rotation=_part_rotation(rz),
                             scale=(size / 2.0, size / 2.0, size / 2.0))
    lid = f"{prefix}_lid"
    session.create_primitive("cube", name=lid,
                             location=(x, y, z + size + 0.02),
                             rotation=_part_rotation(rz),
                             scale=(size / 2.0 + 0.03, size / 2.0 + 0.03, 0.02))
    return [name, lid]


def _build_barrel(session: Any, prefix: str, x: float, y: float, z: float,
                  height: float) -> list[str]:
    name = f"{prefix}_body"
    session.create_primitive("cylinder", name=name,
                             location=(x, y, z + height / 2.0), radius=0.32)
    hoop_a = f"{prefix}_hoop_a"
    session.create_primitive("torus", name=hoop_a,
                             location=(x, y, z + height * 0.25),
                             major_radius=0.33, minor_radius=0.02)
    hoop_b = f"{prefix}_hoop_b"
    session.create_primitive("torus", name=hoop_b,
                             location=(x, y, z + height * 0.75),
                             major_radius=0.33, minor_radius=0.02)
    return [name, hoop_a, hoop_b]


def _build_rock(session: Any, prefix: str, x: float, y: float, z: float,
                size: float) -> list[str]:
    name = f"{prefix}_stone"
    session.create_primitive("sphere", name=name,
                             location=(x, y, z + size * 0.35), radius=size)
    return [name]


def _build_bush(session: Any, prefix: str, x: float, y: float, z: float,
                size: float) -> list[str]:
    name = f"{prefix}_leaves"
    session.create_primitive("sphere", name=name,
                             location=(x, y, z + size * 0.7), radius=size)
    return [name]


def _build_fence(session: Any, prefix: str, x: float, y: float, z: float,
                 rz: float, width: float) -> list[str]:
    """Rails plus pickets — reads as a fence from any angle."""
    names: list[str] = []
    half = width / 2.0
    for suffix, dz, scale in (
        ("rail_low", 0.42, (half, 0.03, 0.035)),
        ("rail_high", 0.82, (half, 0.03, 0.035)),
    ):
        name = f"{prefix}_{suffix}"
        session.create_primitive("cube", name=name,
                                 location=(x, y, z + dz),
                                 rotation=_part_rotation(rz), scale=scale)
        names.append(name)
    pickets = max(3, int(width / 0.45))
    for index in range(pickets):
        dx = -half + (width / (pickets - 1)) * index
        px, py = _rot_xy(x, y, dx, 0.0, rz)
        name = f"{prefix}_picket_{index:02d}"
        session.create_primitive("cube", name=name,
                                 location=(px, py, z + 0.62),
                                 rotation=_part_rotation(rz),
                                 scale=(0.035, 0.045, 0.31))
        names.append(name)
    return names


def _build_sign(session: Any, prefix: str, x: float, y: float, z: float,
                rz: float) -> list[str]:
    post = f"{prefix}_post"
    session.create_primitive("cylinder", name=post,
                             location=(x, y, z + 1.0), radius=0.05)
    bx, by = _rot_xy(x, y, 0.0, 0.1, rz)
    board = f"{prefix}_board"
    session.create_primitive("cube", name=board,
                             location=(bx, by, z + 1.9),
                             rotation=_part_rotation(rz),
                             scale=(0.45, 0.03, 0.3))
    return [post, board]


def _build_planter(session: Any, prefix: str, x: float, y: float, z: float,
                   size: float) -> list[str]:
    pot = f"{prefix}_pot"
    session.create_primitive("cylinder", name=pot,
                             location=(x, y, z + size * 0.5),
                             radius=size * 0.6)
    leaves = f"{prefix}_leaves"
    session.create_primitive("sphere", name=leaves,
                             location=(x, y, z + size + size * 0.4),
                             radius=size * 0.65)
    return [pot, leaves]


def _build_fountain(session: Any, prefix: str, x: float, y: float,
                    z: float) -> list[str]:
    """Basin, water disc, pedestal and bowl — same silhouette as the default."""
    stone = f"{prefix}_MAT_stone"
    _env_material(session, stone, (0.48, 0.47, 0.44), roughness=0.8)
    basin = f"{prefix}_basin"
    session.create_primitive("cylinder", name=basin,
                             location=(x, y, z + 0.45),
                             radius=1.5, depth=0.9)
    session.assign_material(stone, [basin])
    water = f"{prefix}_water"
    session.create_primitive("cylinder", name=water,
                             location=(x, y, z + 0.82),
                             radius=1.3, depth=0.1)
    water_mat = f"{prefix}_MAT_water"
    _env_material(session, water_mat, (0.12, 0.3, 0.42), roughness=0.15)
    session.assign_material(water_mat, [water])
    pedestal = f"{prefix}_pedestal"
    session.create_primitive("cylinder", name=pedestal,
                             location=(x, y, z + 1.3),
                             radius=0.28, depth=1.6)
    session.assign_material(stone, [pedestal])
    bowl = f"{prefix}_bowl"
    session.create_primitive("cylinder", name=bowl,
                             location=(x, y, z + 2.2),
                             radius=0.65, depth=0.35)
    session.assign_material(stone, [bowl])
    return [basin, water, pedestal, bowl]


def _build_lantern(session: Any, prefix: str, x: float, y: float,
                   z: float) -> list[str]:
    """Post, housing, emissive glow and a live point light."""
    stone = f"{prefix}_MAT_stone"
    _env_material(session, stone, (0.4, 0.39, 0.37), roughness=0.8)
    post = f"{prefix}_post"
    session.create_primitive("cylinder", name=post,
                             location=(x, y, z + 1.3),
                             radius=0.09, depth=2.6)
    session.assign_material(stone, [post])
    housing = f"{prefix}_housing"
    session.create_primitive("cube", name=housing,
                             location=(x, y, z + 2.9),
                             scale=(0.22, 0.22, 0.3))
    session.assign_material(stone, [housing])
    glow = f"{prefix}_glow"
    session.create_primitive("sphere", name=glow,
                             location=(x, y, z + 2.85), radius=0.16)
    glow_mat = f"{prefix}_MAT_glow"
    _env_material(session, glow_mat, (0.4, 0.22, 0.08),
                  emission=(1.0, 0.55, 0.18), emission_strength=6.0)
    session.assign_material(glow_mat, [glow])
    lamp = session.create_light(
        name=f"{prefix}_light", light_type="POINT",
        energy=60.0, color=(1.0, 0.62, 0.3),
        location=(x, y, z + 2.85),
    )
    return [post, housing, glow, lamp["name"]]


def _build_archway(session: Any, prefix: str, x: float, y: float,
                   z: float) -> list[str]:
    """Two pillars, a lintel and a true arch ring."""
    stone = f"{prefix}_MAT_stone"
    _env_material(session, stone, (0.48, 0.47, 0.44), roughness=0.8)
    names: list[str] = []
    for side, suffix in ((-1.6, "pillar_L"), (1.6, "pillar_R")):
        pillar = f"{prefix}_{suffix}"
        session.create_primitive(
            "cube", name=pillar, location=(x + side, y, z + 1.6),
            scale=(0.35, 0.35, 1.6),
        )
        session.assign_material(stone, [pillar])
        names.append(pillar)
    lintel = f"{prefix}_lintel"
    session.create_primitive(
        "cube", name=lintel, location=(x, y, z + 3.35),
        scale=(1.95, 0.4, 0.3),
    )
    session.assign_material(stone, [lintel])
    arch = f"{prefix}_arch"
    session.create_primitive(
        "torus", name=arch, location=(x, y, z + 3.3),
        major_radius=1.6, minor_radius=0.22,
    )
    session.assign_material(stone, [arch])
    names.extend([lintel, arch])
    return names


def build_semantic_prop(session: Any, prefix: str, prop_type: str,
                        location: tuple[float, float, float],
                        rz: float = 0.0,
                        dimensions: dict[str, float] | None = None
                        ) -> list[str]:
    """Build one prop of a semantic type; the type decides the geometry.

    ``dimensions`` carries width/depth/height in metres when the plan pins
    them down; every default below is a deliberate, sensible size.
    """
    dims = {k: float(v) for k, v in (dimensions or {}).items() if v}
    x, y, z = (float(location[0]), float(location[1]), float(location[2]))
    width = dims.get("width", 0.0)
    height = dims.get("height", 0.0)
    size = dims.get("size", dims.get("radius", 0.0))

    kind = str(prop_type or "prop").lower().strip()
    if kind in ("bench", "seat"):
        return _build_bench(session, prefix, x, y, z, rz, width or 1.8)
    if kind == "tree":
        return _build_tree(session, prefix, x, y, z, height or 4.5)
    if kind == "wall":
        return _build_wall(session, prefix, x, y, z, rz, width or 6.0,
                           height or 2.8)
    if kind in ("column", "pillar", "post"):
        return _build_column(session, prefix, x, y, z, height or 3.2)
    if kind in ("crate", "box", "chest"):
        return _build_crate(session, prefix, x, y, z, rz, size or 0.7)
    if kind in ("barrel", "cask"):
        return _build_barrel(session, prefix, x, y, z, height or 0.95)
    if kind in ("rock", "boulder", "stone"):
        return _build_rock(session, prefix, x, y, z, size or 0.45)
    if kind in ("bush", "shrub", "hedge"):
        return _build_bush(session, prefix, x, y, z, size or 0.5)
    if kind in ("fence", "railing"):
        return _build_fence(session, prefix, x, y, z, rz, width or 3.5)
    if kind in ("sign", "placard", "board"):
        return _build_sign(session, prefix, x, y, z, rz)
    if kind in ("planter", "pot", "flowerbed"):
        return _build_planter(session, prefix, x, y, z, size or 0.45)
    if kind == "fountain":
        return _build_fountain(session, prefix, x, y, z)
    if kind == "lantern":
        return _build_lantern(session, prefix, x, y, z)
    if kind == "archway":
        return _build_archway(session, prefix, x, y, z)
    # Unknown type: a pedestal with a small load on top — visibly an object,
    # never the bare uniform cube the old fallback produced.
    pedestal = f"{prefix}_pedestal"
    session.create_primitive("cylinder", name=pedestal,
                             location=(x, y, z + 0.45), radius=0.3)
    load = f"{prefix}_piece"
    session.create_primitive("sphere", name=load,
                             location=(x, y, z + 1.05), radius=0.24)
    return [pedestal, load]


def build_environment(session: Any, environment_id: str,
                      spec: dict[str, Any]) -> EnvironmentBuild:
    """Create the environment's geometry in the current Blender scene.

    Every part gets an ``ENV_<id>_<part>`` name (sanitised) so the object
    manifest can prove each piece exists. Parts default to sensible courtyard
    staging when the record does not pin them down, because the requirement is
    a scene that reads as the described place, not an error about a missing
    layout key.
    """
    result = EnvironmentBuild(environment_id=environment_id)
    base = environment_id[4:] if environment_id.startswith("ENV_") else environment_id
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)

    def env(name: str) -> str:
        return f"ENV_{safe}_{name}"

    result.scope = f"ENV_{safe}"

    spec = dict(spec or {})
    layout = spec.get("layout") or {}
    if isinstance(layout, str):
        # The record stores layout as free text; accept JSON when it is.
        text = layout.strip()
        if text.startswith("{"):
            try:
                import json as _json
                parsed = _json.loads(text)
                if isinstance(parsed, dict):
                    layout = parsed
            except ValueError:
                layout = {}
        else:
            layout = {}
    props = spec.get("props") or []
    colors = spec.get("colors") or {}
    weather = str(spec.get("weather") or "")

    def at(key: str, default: tuple[float, float, float]
           ) -> tuple[float, float, float]:
        raw = layout.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            return (float(raw[0]), float(raw[1]), float(raw[2]))
        return default

    def prop_material(key: str, color_override: Any = None) -> str:
        """One shared material per semantic surface, created once per build."""
        name = env(f"MAT_{key}")
        if name not in result.materials:
            color = parse_color(color_override or colors.get(key),
                                _PROP_PALETTE.get(key, (0.5, 0.48, 0.45)))
            _env_material(session, name, color, roughness=0.85)
            result.materials.append(name)
        return name

    def dress_prop(kind: str, names: list[str]) -> None:
        """Assign the right material to each part of a semantic prop."""
        if kind in ("fountain", "lantern", "archway"):
            # These builders assign their own stone/water/glass materials.
            return
        for name in names:
            low = name.lower()
            if low.endswith("_light"):
                continue  # lights carry no material
            if "glow" in low:
                session.assign_material(prop_material("lantern_glow"), [name])
                continue
            if "trunk" in low:
                session.assign_material(prop_material("trunk"), [name])
            elif "canopy" in low or "leaves" in low:
                session.assign_material(prop_material("foliage"), [name])
            elif kind in ("rock", "boulder", "stone"):
                session.assign_material(prop_material("rock"), [name])
            elif "leg" in low or "hoop" in low or "post" in low:
                session.assign_material(prop_material("metal"), [name])
            elif kind in ("wall", "column", "pillar", "pedestal"):
                session.assign_material(prop_material("plaster"), [name])
            else:
                session.assign_material(prop_material("wood"), [name])

    def entry_location(entry: dict[str, Any],
                       default: tuple[float, float, float]
                       ) -> tuple[float, float, float]:
        pos = entry.get("position") or entry.get("location")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            try:
                return (float(pos[0]), float(pos[1]), float(pos[2]))
            except (TypeError, ValueError):
                return default
        return default

    # -- structured object plan: authoritative for what it names ------------
    # The AI Director converts creative intent into explicit entries
    # (id, type, position, rotation, material, dimensions). Free text stays
    # available as context but is never the only source of spatial truth.
    covered_types: set[str] = set()
    plan = spec.get("objects") or []
    if isinstance(plan, list):
        for index, entry in enumerate(plan):
            if not isinstance(entry, dict):
                continue
            prop_type = str(entry.get("type") or "prop")
            prop_id = str(entry.get("id") or f"{prop_type}_{index}")
            prefix = env(f"prop_{prop_id}")
            location = entry_location(entry, (-4.0 + index * 1.8, 5.0, 0.0))
            rotation = float(entry.get("rotation") or 0.0)
            dims = entry.get("dimensions")
            dims = dims if isinstance(dims, dict) else {}
            names = build_semantic_prop(session, prefix, prop_type,
                                        location, rz=rotation,
                                        dimensions=dims)
            dress_prop(prop_type, names)
            result.objects.extend(names)
            result.lights.extend(n for n in names if n.endswith("_light"))
            covered_types.add(prop_type.lower())
            result.props[prop_id] = {
                "type": prop_type,
                "position": list(location),
                "rotation_z": rotation,
                "objects": names,
            }
        if plan:
            result.notes.append(
                f"object plan: {len(result.props)} prop(s) built from the "
                "structured placement list"
            )

    # -- ground: a real mesh with a usable material, never a void ------------
    ground_color = parse_color(
        colors.get("ground") or spec.get("ground_material") or "stone",
        (0.42, 0.41, 0.39),
    )
    wet = "rain" in weather.lower() or "wet" in str(spec).lower()
    session.create_primitive("plane", name=env("ground"),
                             location=at("ground", (0.0, 0.0, 0.0)), size=60.0)
    _env_material(session, env("MAT_ground"), ground_color,
                  roughness=0.35 if wet else 0.9)
    session.assign_material(env("MAT_ground"), [env("ground")])
    result.objects.append(env("ground"))
    result.materials.append(env("MAT_ground"))

    # -- backdrop: a far wall so the frame opens onto place, not void --------
    session.create_primitive(
        "plane", name=env("backdrop"),
        location=at("backdrop", (0.0, 16.0, 9.0)),
        rotation=(math.radians(90.0), 0.0, 0.0), size=50.0,
    )
    _env_material(session, env("MAT_backdrop"),
                  parse_color(colors.get("backdrop"), (0.13, 0.14, 0.18)),
                  roughness=1.0)
    session.assign_material(env("MAT_backdrop"), [env("backdrop")])
    result.objects.append(env("backdrop"))
    result.materials.append(env("MAT_backdrop"))

    # -- fountain: basin, water disc, central pedestal, upper bowl -----------
    if _wants(props, spec, "fountain", default=True) and "fountain" not in covered_types:
        fx, fy, fz = at("fountain", (2.6, 2.2, 0.0))
        stone = env("MAT_stone")
        _env_material(session, stone,
                      parse_color(colors.get("stone"), (0.48, 0.47, 0.44)),
                      roughness=0.8)
        basin = env("fountain_basin")
        session.create_primitive("cylinder", name=basin,
                                 location=(fx, fy, fz + 0.45),
                                 radius=1.5, depth=0.9)
        session.assign_material(stone, [basin])
        water = env("fountain_water")
        session.create_primitive("cylinder", name=water,
                                 location=(fx, fy, fz + 0.82),
                                 radius=1.3, depth=0.1)
        _env_material(session, env("MAT_water"), (0.12, 0.3, 0.42),
                      roughness=0.15)
        session.assign_material(env("MAT_water"), [water])
        pedestal = env("fountain_pedestal")
        session.create_primitive("cylinder", name=pedestal,
                                 location=(fx, fy, fz + 1.3),
                                 radius=0.28, depth=1.6)
        session.assign_material(stone, [pedestal])
        bowl = env("fountain_bowl")
        session.create_primitive("cylinder", name=bowl,
                                 location=(fx, fy, fz + 2.2),
                                 radius=0.65, depth=0.35)
        session.assign_material(stone, [bowl])
        result.objects.extend([basin, water, pedestal, bowl])
        result.materials.extend([stone, env("MAT_water")])
        result.notes.append(f"fountain assembled at ({fx}, {fy})")

    # -- lantern: housing geometry plus a real light so it reads at dusk -----
    if _wants(props, spec, "lantern", default=True) and "lantern" not in covered_types:
        lx, ly, lz = at("lantern", (-2.4, 1.8, 0.0))
        post = env("lantern_post")
        session.create_primitive("cylinder", name=post,
                                 location=(lx, ly, lz + 1.3),
                                 radius=0.09, depth=2.6)
        session.assign_material(env("MAT_stone"), [post])
        housing = env("lantern_housing")
        session.create_primitive("cube", name=housing,
                                 location=(lx, ly, lz + 2.9),
                                 scale=(0.22, 0.22, 0.3))
        # The housing is the visible body of the lantern, and the glow sphere
        # sits inside it. Painting the housing with the opaque stone material
        # hid that sphere completely, so the lantern rendered as a plain grey
        # box -- an object the shot calls "glowing" that visibly did not. The
        # body itself is emissive, which is what makes it read as the source.
        lantern_glow = float(spec.get("lantern_glow_strength") or 7.0)
        _env_material(session, env("MAT_lantern_body"),
                      (0.55, 0.3, 0.1),
                      emission=(1.0, 0.55, 0.18),
                      emission_strength=lantern_glow)
        session.assign_material(env("MAT_lantern_body"), [housing])
        glow = env("lantern_glow")
        session.create_primitive("sphere", name=glow,
                                 location=(lx, ly, lz + 2.85), radius=0.16)
        _env_material(session, env("MAT_lantern_glow"),
                      (0.4, 0.22, 0.08),
                      emission=(1.0, 0.55, 0.18), emission_strength=6.0)
        session.assign_material(env("MAT_lantern_glow"), [glow])
        lamp = session.create_light(
            name=env("lantern_light"), light_type="POINT",
            energy=float(spec.get("lantern_energy") or 260.0),
            color=(1.0, 0.62, 0.3),
            location=(lx, ly, lz + 2.85),
        )
        result.objects.extend([post, housing, glow, lamp["name"]])
        result.materials.extend([env("MAT_lantern_body"),
                                 env("MAT_lantern_glow")])
        result.lights.append(lamp["name"])
        result.notes.append(f"lantern with live light at ({lx}, {ly})")

    # -- archway: two pillars, a lintel, and a true arch ring ---------------
    if _wants(props, spec, "archway", default=True) and "archway" not in covered_types:
        ax, ay, az = at("archway", (0.0, -3.4, 0.0))
        for side, suffix in ((-1.6, "pillar_L"), (1.6, "pillar_R")):
            pillar = env(f"archway_{suffix}")
            session.create_primitive(
                "cube", name=pillar, location=(ax + side, ay, az + 1.6),
                scale=(0.35, 0.35, 1.6),
            )
            session.assign_material(env("MAT_stone"), [pillar])
            result.objects.append(pillar)
        lintel = env("archway_lintel")
        session.create_primitive(
            "cube", name=lintel, location=(ax, ay, az + 3.35),
            scale=(1.95, 0.4, 0.3),
        )
        session.assign_material(env("MAT_stone"), [lintel])
        arch = env("archway_arch")
        session.create_primitive(
            "torus", name=arch, location=(ax, ay, az + 3.3),
            major_radius=1.6, minor_radius=0.22,
        )
        session.assign_material(env("MAT_stone"), [arch])
        result.objects.extend([lintel, arch])
        result.notes.append(f"archway spanning the entrance at ({ax}, {ay})")

    # -- extra props named in free text get their semantic geometry ----------
    # `props` is prose ("stone bench", "glowing lantern") while the builders are
    # keyed by type ("bench", "lantern"). Comparing the whole string against the
    # type names therefore never matched, so every prose entry the plan had
    # already placed was built a second time as an unknown-type blob at a
    # hardcoded position -- a row of identical pedestals across the courtyard.
    # Match on the words instead, and treat "no X" as the instruction it is
    # rather than as a request for a prop called "no X".
    handled = covered_types | {"fountain", "lantern", "archway",
                               "ground", "backdrop"}
    extra = [
        p for p in props
        if not str(p).lower().strip().startswith(("no ", "without "))
    ]
    for index, prop in enumerate(extra[:8]):
        prop_type = str(prop).lower().strip()
        words = {w.rstrip("s") for w in re.split(r"[^a-z0-9]+", prop_type) if w}
        if words & handled:
            continue
        prop_id = str(prop).replace(" ", "_")[:24] or f"prop_{index}"
        prefix = env(f"prop_{index}_{prop_id}")
        names = build_semantic_prop(
            session, prefix, prop_type,
            (-4.0 + index * 1.8, 5.0, 0.0),
        )
        dress_prop(prop_type, names)
        result.objects.extend(names)
        result.lights.extend(n for n in names if n.endswith("_light"))
        result.props[prop_id] = {
            "type": prop_type,
            "position": [-4.0 + index * 1.8, 5.0, 0.0],
            "rotation_z": 0.0,
            "objects": names,
            "source": "free_text_props",
        }

    result.notes.append(
        f"environment {environment_id}: {len(result.objects)} objects, "
        f"{len(result.materials)} materials, {len(result.lights)} lights"
    )
    return result


# ---------------------------------------------------------------------------
# Character builder
# ---------------------------------------------------------------------------


@dataclass
class CharacterBuild:
    """What was built into Blender for one canonical character."""

    character_id: str
    name: str
    scope: str = ""
    root: str = ""
    armature: str = ""
    parts: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    bones: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "name": self.name,
            "scope": self.scope,
            "root": self.root,
            "armature": self.armature,
            "parts": self.parts,
            "materials": self.materials,
            "bones": self.bones,
            "notes": self.notes,
        }


#: Poses as bone rotations (radians). Deterministic object/armature transforms,
#: not a motion library -- the minimum reliable system the spec asks for.
POSES: dict[str, dict[str, tuple[float, float, float]]] = {
    "idle": {},
    # Every animatable bone explicitly at zero. `idle` is an empty mapping --
    # it says nothing, so it cannot be keyed as the "before" state of a move.
    # `rest` is the pose an animation starts from.
    "rest": {
        "head": (0.0, 0.0, 0.0),
        "torso": (0.0, 0.0, 0.0),
        "upper_arm_L": (0.0, 0.0, 0.0),
        "upper_arm_R": (0.0, 0.0, 0.0),
        "lower_arm_L": (0.0, 0.0, 0.0),
        "lower_arm_R": (0.0, 0.0, 0.0),
        "upper_leg_L": (0.0, 0.0, 0.0),
        "upper_leg_R": (0.0, 0.0, 0.0),
        "lower_leg_L": (0.0, 0.0, 0.0),
        "lower_leg_R": (0.0, 0.0, 0.0),
    },
    "walk": {
        "upper_leg_L": (0.5, 0.0, 0.0),
        "upper_leg_R": (-0.5, 0.0, 0.0),
        "upper_arm_L": (-0.45, 0.0, 0.0),
        "upper_arm_R": (0.45, 0.0, 0.0),
        "lower_leg_L": (-0.35, 0.0, 0.0),
        "lower_leg_R": (0.15, 0.0, 0.0),
    },
    # The same stride with the legs and arms swapped. Alternating `walk` and
    # `walk_b` across a shot is what turns a held pose into a stepping cycle.
    "walk_b": {
        "upper_leg_L": (-0.5, 0.0, 0.0),
        "upper_leg_R": (0.5, 0.0, 0.0),
        "upper_arm_L": (0.45, 0.0, 0.0),
        "upper_arm_R": (-0.45, 0.0, 0.0),
        "lower_leg_L": (0.15, 0.0, 0.0),
        "lower_leg_R": (-0.35, 0.0, 0.0),
    },
    # Head and torso turn toward the shot's accent. The bones point straight
    # up, so their local Y is the world vertical and a positive Y rotation
    # yaws toward camera-right.
    "turn": {
        "head": (0.0, 0.6, 0.0),
        "torso": (0.0, 0.15, 0.0),
    },
    "enter": {
        "upper_leg_L": (0.35, 0.0, 0.0),
        "upper_leg_R": (-0.25, 0.0, 0.0),
        "upper_arm_L": (-0.3, 0.0, 0.0),
        "upper_arm_R": (-0.3, 0.0, 0.0),
        "torso": (0.06, 0.0, 0.0),
    },
    "reach": {
        "upper_arm_R": (-1.5, 0.0, 0.0),
        "lower_arm_R": (-0.25, 0.0, 0.0),
        "torso": (0.12, 0.0, 0.0),
        "head": (0.1, 0.0, 0.0),
    },
    "lean": {
        "torso": (0.28, 0.0, 0.0),
        "head": (0.18, 0.0, 0.0),
        "upper_arm_L": (-0.2, 0.0, -0.15),
        "upper_arm_R": (-0.2, 0.0, 0.15),
    },
    # A recoil: head and torso pull back, arms lift a little to brace. The
    # arms were previously thrown almost fully up and out, which on a
    # segmented body reads as a scarecrow rather than a flinch -- a reaction
    # has to stay legible as a person reacting.
    "react": {
        "head": (-0.22, 0.0, 0.0),
        "torso": (-0.1, 0.0, 0.0),
        "upper_arm_L": (-0.45, 0.0, 0.0),
        "upper_arm_R": (-0.45, 0.0, 0.0),
    },
}


def build_character(session: Any, character_id: str, name: str,
                    canonical: dict[str, Any],
                    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    prefix: str = "") -> CharacterBuild:
    """Build a segmented procedural humanoid driven by the canonical record.

    Parts: head, torso, coat shell, two arms (upper+lower), two legs
    (upper+lower), boots -- each a real mesh with canonical materials for skin,
    coat, trousers and boots. A simple armature (spine, neck, arms, legs) gives
    the deterministic pose system something to drive.

    ``prefix`` scopes the build (e.g. per-shot) so two instances never collide.
    """
    safe_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    scope = (prefix + "_" if prefix else "") + f"CHR_{safe_name}"
    result = CharacterBuild(character_id=character_id, name=name, scope=scope)
    canonical = dict(canonical or {})
    colors = canonical.get("colors") or {}
    clothing = str(canonical.get("clothing") or "")
    height = max(0.5, float(canonical.get("height_m") or 1.8))
    scale = height / 1.8

    skin = parse_color(colors.get("skin"), (0.72, 0.52, 0.4))
    coat = parse_color(colors.get("coat") or colors.get("jacket"),
                       (0.1, 0.45, 0.45))
    trousers = parse_color(colors.get("trousers") or colors.get("pants"),
                           (0.2, 0.2, 0.24))
    boots = parse_color(colors.get("boots"), (0.12, 0.09, 0.07))
    hair = parse_color(colors.get("hair"), (0.1, 0.08, 0.06))
    x, y, z = location

    def part(suffix: str) -> str:
        return f"{scope}_{suffix}"

    def mat(suffix: str, color: tuple[float, float, float],
            roughness: float = 0.75) -> str:
        name = f"{scope}_MAT_{suffix}"
        session.create_material(name, base_color=color, roughness=roughness)
        result.materials.append(name)
        return name

    mat_skin = mat("skin", skin, 0.65)
    mat_coat = mat("coat", coat, 0.85)
    mat_trousers = mat("trousers", trousers, 0.9)
    mat_boots = mat("boots", boots, 0.55)
    mat_hair = mat("hair", hair, 0.5)

    # -- pelvis root: the handle shots move ----------------------------------
    root = part("root")
    session.create_empty(name=root, location=(x, y, z))
    result.root = root
    result.objects = []  # type: ignore[attr-defined]

    def mesh(kind: str, suffix: str, loc: tuple[float, float, float],
             material: str, **kwargs: Any) -> str:
        name = part(suffix)
        session.create_primitive(kind, name=name, location=loc, **kwargs)
        session.assign_material(material, [name])
        result.parts.append(name)
        return name

    # -- legs ----------------------------------------------------------------
    leg_L = mesh("cylinder", "leg_L", (x - 0.14 * scale, y, z + 0.55 * scale),
                 mat_trousers, radius=0.11 * scale, depth=0.75 * scale)
    leg_R = mesh("cylinder", "leg_R", (x + 0.14 * scale, y, z + 0.55 * scale),
                 mat_trousers, radius=0.11 * scale, depth=0.75 * scale)
    boot_L = mesh("cube", "boot_L", (x - 0.14 * scale, y - 0.02, z + 0.12 * scale),
                  mat_boots, scale=(0.11 * scale, 0.16 * scale, 0.12 * scale))
    boot_R = mesh("cube", "boot_R", (x + 0.14 * scale, y - 0.02, z + 0.12 * scale),
                  mat_boots, scale=(0.11 * scale, 0.16 * scale, 0.12 * scale))
    # -- torso ----------------------------------------------------------------
    torso = mesh("cube", "torso", (x, y, z + 1.15 * scale),
                 mat_coat, scale=(0.24 * scale, 0.15 * scale, 0.32 * scale))
    # -- coat shell: a wider, longer skirt around the torso --------------------
    coat_shell = mesh("cube", "coat", (x, y + 0.01, z + 0.95 * scale),
                      mat_coat, scale=(0.30 * scale, 0.19 * scale, 0.48 * scale))
    # -- arms ------------------------------------------------------------------
    arm_L = mesh("cylinder", "arm_L", (x - 0.36 * scale, y, z + 1.18 * scale),
                 mat_coat, radius=0.075 * scale, depth=0.6 * scale)
    arm_R = mesh("cylinder", "arm_R", (x + 0.36 * scale, y, z + 1.18 * scale),
                 mat_coat, radius=0.075 * scale, depth=0.6 * scale)
    hand_L = mesh("sphere", "hand_L", (x - 0.36 * scale, y, z + 0.82 * scale),
                  mat_skin, radius=0.07 * scale)
    hand_R = mesh("sphere", "hand_R", (x + 0.36 * scale, y, z + 0.82 * scale),
                  mat_skin, radius=0.07 * scale)
    # -- head ------------------------------------------------------------------
    head = mesh("sphere", "head", (x, y, z + 1.66 * scale),
                mat_skin, radius=0.14 * scale)
    haircap = mesh("sphere", "hair", (x, y + 0.015, z + 1.72 * scale),
                   mat_hair, radius=0.145 * scale)

    # -- parent everything to the root so a shot moves one handle --------------
    for obj in result.parts:
        session.set_transform(obj, parent=root)

    # -- armature: spine, neck, arms, legs --------------------------------------
    arm_name = part("rig")
    shoulder_y = 1.42 * scale
    bones = [
        {"name": "pelvis", "head": (x, y, z + 0.9 * scale),
         "tail": (x, y, z + 1.1 * scale), "parent": ""},
        {"name": "torso", "head": (x, y, z + 1.1 * scale),
         "tail": (x, y, z + 1.42 * scale), "parent": "pelvis"},
        {"name": "head", "head": (x, y, z + 1.5 * scale),
         "tail": (x, y, z + 1.8 * scale), "parent": "torso"},
        {"name": "upper_arm_L", "head": (x - 0.28 * scale, y, z + shoulder_y),
         "tail": (x - 0.36 * scale, y, z + 1.05 * scale), "parent": "torso"},
        {"name": "lower_arm_L", "head": (x - 0.36 * scale, y, z + 1.05 * scale),
         "tail": (x - 0.36 * scale, y, z + 0.82 * scale), "parent": "upper_arm_L"},
        {"name": "upper_arm_R", "head": (x + 0.28 * scale, y, z + shoulder_y),
         "tail": (x + 0.36 * scale, y, z + 1.05 * scale), "parent": "torso"},
        {"name": "lower_arm_R", "head": (x + 0.36 * scale, y, z + 1.05 * scale),
         "tail": (x + 0.36 * scale, y, z + 0.82 * scale), "parent": "upper_arm_R"},
        {"name": "upper_leg_L", "head": (x - 0.14 * scale, y, z + 0.9 * scale),
         "tail": (x - 0.14 * scale, y, z + 0.45 * scale), "parent": "pelvis"},
        {"name": "lower_leg_L", "head": (x - 0.14 * scale, y, z + 0.45 * scale),
         "tail": (x - 0.14 * scale, y, z + 0.1 * scale), "parent": "upper_leg_L"},
        {"name": "upper_leg_R", "head": (x + 0.14 * scale, y, z + 0.9 * scale),
         "tail": (x + 0.14 * scale, y, z + 0.45 * scale), "parent": "pelvis"},
        {"name": "lower_leg_R", "head": (x + 0.14 * scale, y, z + 0.45 * scale),
         "tail": (x + 0.14 * scale, y, z + 0.1 * scale), "parent": "upper_leg_R"},
    ]
    rig = session.create_armature(arm_name, bones, location=(0.0, 0.0, 0.0))
    result.armature = rig["name"]
    result.bones = rig["bones"]
    session.set_transform(result.armature, parent=root)

    # Rigidly skin each part to the bone that drives it. Without this the
    # armature is decorative: the bones rotate, the mesh does not follow, and
    # every pose in POSES renders identically to the rest pose.
    bindings = [
        {"object": part(name_), "bone": bone_}
        for name_, bone_ in (
            ("leg_L", "upper_leg_L"), ("leg_R", "upper_leg_R"),
            ("boot_L", "lower_leg_L"), ("boot_R", "lower_leg_R"),
            ("torso", "torso"), ("coat", "pelvis"),
            ("arm_L", "upper_arm_L"), ("arm_R", "upper_arm_R"),
            ("hand_L", "lower_arm_L"), ("hand_R", "lower_arm_R"),
            ("head", "head"), ("hair", "head"),
        )
    ]
    try:
        session.call("bind_armature", armature=result.armature,
                     bindings=bindings)
        result.notes.append(
            f"{len(bindings)} parts skinned to bones (rigid, one bone per part)"
        )
    except Exception as exc:  # noqa: BLE001 - an unbound rig must not kill a build
        result.notes.append(f"armature binding skipped: {exc}")

    result.notes.append(
        f"character {name!r}: {len(result.parts)} parts, "
        f"{len(result.bones)} bones, coat={coat}, clothing={clothing or 'coat'}"
    )
    return result


def pose_character(session: Any, armature: str, pose: str,
                   frame: int | None = None) -> dict[str, Any]:
    """Apply a named pose to an armature, optionally keyframing it."""
    rotations = POSES.get(pose)
    if rotations is None:
        raise ValueError(f"unknown pose {pose!r}; known: {sorted(POSES)}")
    applied = []
    for bone, rotation in rotations.items():
        session.pose_bone(armature, bone, rotation_euler=rotation, frame=frame)
        applied.append(bone)
    return {"armature": armature, "pose": pose, "bones": applied,
            "frame": frame}


__all__ = [
    "CharacterBuild",
    "EnvironmentBuild",
    "POSES",
    "build_character",
    "build_environment",
    "parse_color",
    "pose_character",
]

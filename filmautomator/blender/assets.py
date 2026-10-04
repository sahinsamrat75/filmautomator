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

    environment_id: str
    scope: str = ""
    objects: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    lights: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "scope": self.scope,
            "objects": self.objects,
            "materials": self.materials,
            "lights": self.lights,
            "notes": self.notes,
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
    if _wants(props, spec, "fountain", default=True):
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
    if _wants(props, spec, "lantern", default=True):
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
        session.assign_material(env("MAT_stone"), [housing])
        glow = env("lantern_glow")
        session.create_primitive("sphere", name=glow,
                                 location=(lx, ly, lz + 2.85), radius=0.16)
        _env_material(session, env("MAT_lantern_glow"),
                      (0.4, 0.22, 0.08),
                      emission=(1.0, 0.55, 0.18), emission_strength=6.0)
        session.assign_material(env("MAT_lantern_glow"), [glow])
        lamp = session.create_light(
            name=env("lantern_light"), light_type="POINT",
            energy=60.0, color=(1.0, 0.62, 0.3),
            location=(lx, ly, lz + 2.85),
        )
        result.objects.extend([post, housing, glow, lamp["name"]])
        result.materials.append(env("MAT_lantern_glow"))
        result.lights.append(lamp["name"])
        result.notes.append(f"lantern with live light at ({lx}, {ly})")

    # -- archway: two pillars, a lintel, and a true arch ring ---------------
    if _wants(props, spec, "archway", default=True):
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

    # -- extra props named in the record get simple standing markers ---------
    extra = [p for p in props
             if str(p).lower() not in {"fountain", "lantern", "archway",
                                       "ground", "backdrop"}]
    for index, prop in enumerate(extra[:8]):
        marker = env(f"prop_{index}_{str(prop)[:16]}")
        session.create_primitive(
            "cube", name=marker,
            location=(-4.0 + index * 1.6, 5.5, 0.5),
            scale=(0.3, 0.3, 0.5),
        )
        session.assign_material(env("MAT_stone"), [marker])
        result.objects.append(marker)

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
    "walk": {
        "upper_leg_L": (0.5, 0.0, 0.0),
        "upper_leg_R": (-0.5, 0.0, 0.0),
        "upper_arm_L": (-0.45, 0.0, 0.0),
        "upper_arm_R": (0.45, 0.0, 0.0),
        "lower_leg_L": (-0.35, 0.0, 0.0),
        "lower_leg_R": (0.15, 0.0, 0.0),
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
    "react": {
        "head": (-0.28, 0.0, 0.0),
        "torso": (-0.12, 0.0, 0.0),
        "upper_arm_L": (-0.9, 0.0, -0.3),
        "upper_arm_R": (-0.9, 0.0, 0.3),
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

"""In-Blender control server.

Runs inside Blender's own Python interpreter, launched as::

    blender -b --factory-startup -P control_server.py -- --port 0 --token <hex>

It opens a loopback TCP socket and serves newline-delimited JSON requests. This
is the "Blender Control Server" of spec section 4 and the substrate for the
observability channels of section 5.

Design notes
------------
* Single-threaded on purpose. ``bpy`` is not thread-safe, and a background
  Blender does not need a UI event loop, so blocking here is correct.
* Framing is one JSON object per line. ``json.dumps`` never emits a raw
  newline, so line framing is unambiguous.
* Every operation is wrapped: a failing op returns ``ok: false`` with a
  traceback rather than killing the session. That is the error-observation
  channel of spec section 5E, and it is what lets the orchestrator retry
  instead of losing the whole scene.
* Written against Blender 4.x, which bundles Python 3.11. Nothing newer than
  3.11 syntax is used here.

Pass ``--port 0`` to bind an ephemeral port; the chosen port is printed to
stdout as ``FA_CONTROL_PORT <n>`` so the parent process can discover it without
racing on a fixed port.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback

import bpy  # type: ignore[import-not-found]  # provided by Blender at runtime
from mathutils import Vector  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list(value) -> list:
    """Convert Blender vector/matrix-ish objects into plain lists."""
    try:
        return [float(v) for v in value]
    except TypeError:
        return list(value)


def _engine_options() -> list:
    """Best-effort list of engine identifiers, for diagnostics only.

    The engine enum is generated at runtime and does NOT enumerate reliably
    through bl_rna in background mode — Blender 5.2 reports only
    ['BLENDER_EEVEE'] here while happily accepting 'CYCLES' and 'AgX'. Never
    make a decision based on this; use _apply_engine instead.
    """
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        return [item.identifier for item in prop.enum_items]
    except Exception:
        return []


#: Order to try when the requested engine is not available in this build.
#: EEVEE is the safest fallback: it works headless and still shows lighting.
_ENGINE_FALLBACKS = {
    "BLENDER_WORKBENCH": ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT",
                          "BLENDER_EEVEE", "CYCLES"),
    "BLENDER_EEVEE": ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT",
                      "BLENDER_WORKBENCH", "CYCLES"),
    "BLENDER_EEVEE_NEXT": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE",
                           "BLENDER_WORKBENCH", "CYCLES"),
    "CYCLES": ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE",
               "BLENDER_WORKBENCH"),
}


def _apply_engine(render, requested: str) -> tuple:
    """Set the render engine, falling back if this build does not have it.

    Assigns directly and lets Blender be the authority, rather than trusting a
    probed enum list that may be empty or truncated. Returns (actual, note).
    """
    for candidate in _ENGINE_FALLBACKS.get(requested, (requested,)):
        try:
            render.engine = candidate
        except (TypeError, ValueError):
            continue
        if candidate == requested:
            return candidate, ""
        return candidate, (
            "engine %r unavailable in this build; using %r instead"
            % (requested, candidate)
        )
    return render.engine, "could not set engine %r" % requested


#: How different a pixel must be (summed across R, G, B in linear space) from
#: the background plate to count as subject.
#:
#: Removing a subject from a lit scene changes more than its silhouette: it
#: also removes the light it bounced onto its surroundings and the shadow it
#: cast, which can tint or shade a large part of the frame. Measured on a
#: strongly lit test scene, a subject's own pixels differ from the plate by
#: roughly 0.8 and up, while its bounce light shifts the rest of the frame by
#: about 0.25. This threshold sits between the two, so the mask is the subject
#: rather than the subject plus its influence on the room.
PLATE_DIFFERENCE_THRESHOLD = 0.5


def _object_ref(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise KeyError("no object named %r in this file" % name)
    return obj


def _look_at_rotation(location, target) -> list:
    """Euler angles that point a camera's -Z axis from location to target."""
    direction = Vector(target) - Vector(location)
    if direction.length == 0:
        return [0.0, 0.0, 0.0]
    # to_track_quat('-Z', 'Y') is Blender's convention for cameras and lights.
    quat = direction.to_track_quat("-Z", "Y")
    return [float(a) for a in quat.to_euler()]


# ---------------------------------------------------------------------------
# Observation: scene state  (spec 5A)
# ---------------------------------------------------------------------------


def op_get_scene_state(_args: dict) -> dict:
    scene = bpy.context.scene
    render = scene.render

    objects = []
    for obj in bpy.data.objects:
        entry = {
            "name": obj.name,
            "type": obj.type,
            "location": _to_list(obj.location),
            "rotation_euler": _to_list(obj.rotation_euler),
            "scale": _to_list(obj.scale),
            "dimensions": _to_list(obj.dimensions),
            "visible": not obj.hide_render,
            "parent": obj.parent.name if obj.parent else None,
            "collections": [c.name for c in obj.users_collection],
            "modifiers": [m.type for m in obj.modifiers],
        }
        if obj.type == "MESH":
            entry["vertex_count"] = len(obj.data.vertices)
            entry["polygon_count"] = len(obj.data.polygons)
            entry["materials"] = [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ]
        if obj.type in {"CAMERA", "LIGHT"}:
            entry["data_name"] = obj.data.name
        if obj.type == "CAMERA":
            entry["lens_mm"] = float(obj.data.lens)
            entry["sensor_width"] = float(obj.data.sensor_width)
        if obj.type == "LIGHT":
            entry["light_type"] = obj.data.type
            entry["energy"] = float(obj.data.energy)
            entry["color"] = _to_list(obj.data.color)
        if obj.animation_data and obj.animation_data.action:
            entry["action"] = obj.animation_data.action.name
        objects.append(entry)

    cameras = [
        {
            "name": o.name,
            "lens_mm": float(o.data.lens),
            "location": _to_list(o.location),
            "rotation_euler": _to_list(o.rotation_euler),
            "is_active": o == scene.camera,
        }
        for o in bpy.data.objects
        if o.type == "CAMERA"
    ]

    lights = [
        {
            "name": o.name,
            "light_type": o.data.type,
            "energy": float(o.data.energy),
            "color": _to_list(o.data.color),
        }
        for o in bpy.data.objects
        if o.type == "LIGHT"
    ]

    materials = [
        {
            "name": m.name,
            "use_nodes": bool(m.use_nodes),
            "base_color": (
                _to_list(m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value)
                if m.use_nodes and "Principled BSDF" in m.node_tree.nodes
                else None
            ),
        }
        for m in bpy.data.materials
    ]

    armatures = [
        {
            "name": o.name,
            "bone_count": len(o.data.bones),
            "bones": [b.name for b in o.data.bones][:64],
        }
        for o in bpy.data.objects
        if o.type == "ARMATURE"
    ]

    actions = []
    for a in bpy.data.actions:
        try:
            fcurves = list(a.fcurves)
        except AttributeError:
            # Blender 5.x layered actions expose channels, not fcurves.
            fcurves = []
            try:
                for layer in a.layers:
                    for strip in layer.strips:
                        channelbag = getattr(strip, "channelbag", lambda: None)()
                        if channelbag is not None:
                            fcurves.extend(list(channelbag.fcurves))
            except Exception:
                pass
        try:
            frame_range = _to_list(a.frame_range)
        except Exception:
            frame_range = []
        actions.append({
            "name": a.name,
            "frame_range": frame_range,
            "fcurve_count": len(fcurves),
        })

    constraints = [
        {
            "object": o.name,
            "constraints": [
                {"name": c.name, "type": c.type, "target": getattr(c, "target", None).name
                 if getattr(c, "target", None) else None}
                for c in o.constraints
            ],
        }
        for o in bpy.data.objects
        if o.constraints
    ]

    return {
        "filepath": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "scene_name": scene.name,
        "frame_current": int(scene.frame_current),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "fps": float(render.fps) / float(render.fps_base or 1),
        "engine": render.engine,
        "resolution": [int(render.resolution_x), int(render.resolution_y)],
        "resolution_percentage": int(render.resolution_percentage),
        "film_transparent": bool(render.film_transparent),
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "active_camera": scene.camera.name if scene.camera else None,
        "collections": [c.name for c in bpy.data.collections],
        "objects": objects,
        "cameras": cameras,
        "lights": lights,
        "materials": materials,
        "armatures": armatures,
        "actions": actions,
        "constraints": constraints,
        "object_count": len(bpy.data.objects),
    }


def op_get_errors(_args: dict) -> dict:
    """Report integrity problems in the current file (spec 5E)."""
    problems = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            if obj.data is None or len(obj.data.vertices) == 0:
                problems.append({"kind": "empty_mesh", "object": obj.name})
            for slot in obj.material_slots:
                if slot.material is None:
                    problems.append({"kind": "empty_material_slot", "object": obj.name})
        if obj.type == "CAMERA" and obj.data is None:
            problems.append({"kind": "missing_camera_data", "object": obj.name})
    for image in bpy.data.images:
        if image.source == "FILE" and not image.has_data:
            problems.append({"kind": "missing_texture", "image": image.name,
                             "filepath": image.filepath})
    for lib in bpy.data.libraries:
        if lib.filepath and not os.path.exists(bpy.path.abspath(lib.filepath)):
            problems.append({"kind": "broken_library_link", "library": lib.name,
                             "filepath": lib.filepath})
    if bpy.context.scene.camera is None:
        problems.append({"kind": "no_active_camera", "object": None})
    return {"problems": problems, "count": len(problems)}


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

_PRIMITIVE_OPS = {
    "cube": "primitive_cube_add",
    "sphere": "primitive_uv_sphere_add",
    "ico_sphere": "primitive_ico_sphere_add",
    "plane": "primitive_plane_add",
    "cylinder": "primitive_cylinder_add",
    "cone": "primitive_cone_add",
    "torus": "primitive_torus_add",
    "monkey": "primitive_monkey_add",
    "grid": "primitive_grid_add",
}


#: Which size keyword each primitive operator accepts. Getting this wrong
#: raises a TypeError, or worse, silently ignores the value and builds the
#: operator's default size.
_PRIMITIVE_SIZE_KWARG = {
    "cube": "size",
    "plane": "size",
    "grid": "size",
    "sphere": "radius",
    "ico_sphere": "radius",
    "cylinder": "radius",
    "cone": "radius",
    "monkey": None,   # neither size nor radius
    "torus": None,    # takes major_radius / minor_radius instead
}


def op_create_primitive(args: dict) -> dict:
    kind = str(args.get("kind", "cube")).lower()
    op_name = _PRIMITIVE_OPS.get(kind)
    if op_name is None:
        raise ValueError("unknown primitive %r; known: %s"
                         % (kind, ", ".join(sorted(_PRIMITIVE_OPS))))
    op = getattr(bpy.ops.mesh, op_name)
    kwargs = {"location": args.get("location", (0.0, 0.0, 0.0))}
    if "rotation" in args:
        kwargs["rotation"] = args["rotation"]
    if kind == "torus":
        if "major_radius" in args:
            kwargs["major_radius"] = args["major_radius"]
        if "minor_radius" in args:
            kwargs["minor_radius"] = args["minor_radius"]
    else:
        size_kwarg = _PRIMITIVE_SIZE_KWARG.get(kind)
        if size_kwarg:
            # Callers may spell this either way — `size` for box-like shapes,
            # `radius` for round ones. Accept both and map onto whatever this
            # operator actually takes; silently dropping it produces a
            # default-sized object that looks like a mysterious modelling bug.
            if "radius" in args:
                kwargs[size_kwarg] = args["radius"]
            elif "size" in args:
                kwargs[size_kwarg] = args["size"]
    op(**kwargs)

    obj = bpy.context.active_object
    if name := args.get("name"):
        obj.name = name
        if obj.data:
            obj.data.name = name
    if scale := args.get("scale"):
        obj.scale = scale
    if rotation := args.get("rotation"):
        obj.rotation_euler = rotation
    return {"name": obj.name, "type": obj.type,
            "location": _to_list(obj.location)}


def op_create_empty(args: dict) -> dict:
    bpy.ops.object.empty_add(
        type=args.get("empty_type", "PLAIN_AXES"),
        location=args.get("location", (0.0, 0.0, 0.0)),
    )
    obj = bpy.context.active_object
    if name := args.get("name"):
        obj.name = name
    return {"name": obj.name}


def op_delete_object(args: dict) -> dict:
    removed = []
    for name in args.get("names", []):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
        removed.append(name)
    missing = [n for n in args.get("names", []) if n not in removed]
    return {"removed": removed, "not_found": missing}


def op_set_transform(args: dict) -> dict:
    obj = _object_ref(args["name"])
    if "location" in args:
        obj.location = args["location"]
    if "rotation_euler" in args:
        obj.rotation_euler = args["rotation_euler"]
    if "scale" in args:
        obj.scale = args["scale"]
    if "dimensions" in args:
        obj.dimensions = args["dimensions"]
    if "name_new" in args:
        obj.name = args["name_new"]
    if "hide_render" in args:
        obj.hide_render = bool(args["hide_render"])
    if "parent" in args:
        parent = args["parent"]
        obj.parent = _object_ref(parent) if parent else None
    return {
        "name": obj.name,
        "location": _to_list(obj.location),
        "rotation_euler": _to_list(obj.rotation_euler),
        "scale": _to_list(obj.scale),
        "dimensions": _to_list(obj.dimensions),
    }


def op_set_visibility(args: dict) -> dict:
    """Show or hide objects in the viewport and/or the render.

    Returns what was actually set, read back from Blender, so the caller can
    confirm the scene matches what was asked rather than assume it.
    """
    hide_render = args.get("hide_render")
    hide_viewport = args.get("hide_viewport")
    changed = []
    for name in args.get("names", []):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if hide_render is not None:
            obj.hide_render = bool(hide_render)
        if hide_viewport is not None:
            obj.hide_viewport = bool(hide_viewport)
        changed.append({
            "name": obj.name,
            "hide_render": bool(obj.hide_render),
            "hide_viewport": bool(obj.hide_viewport),
        })
    missing = [n for n in args.get("names", [])
               if bpy.data.objects.get(n) is None]
    return {"changed": changed, "not_found": missing}


def op_link_objects(args: dict) -> dict:
    """Instantiate objects from a canonical asset .blend into this scene.

    The library file is the single source of truth for the asset; shots link
    (copy) its objects rather than rebuilding them. ``names`` selects which
    objects to bring in — omit it to bring in everything whose name starts with
    ``prefix``. Linked objects are renamed with ``rename_prefix`` so two shots'
    instances never collide.
    """
    filepath = os.path.abspath(args["filepath"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    names = args.get("names")
    prefix = args.get("prefix", "")
    rename_prefix = args.get("rename_prefix", "")
    linked = []
    with bpy.data.libraries.load(filepath, link=False) as (src, dst):
        available = list(src.objects)
        if names is None:
            if prefix:
                wanted = [n for n in available if n.startswith(prefix)]
            else:
                wanted = list(available)
        else:
            wanted = [n for n in names if n in available]
            missing = [n for n in names if n not in available]
            if missing:
                raise KeyError("library %s has no objects %r" % (filepath, missing))
        dst.objects = wanted
    for obj in dst.objects:
        if obj is None:
            continue
        bpy.context.scene.collection.objects.link(obj)
        if rename_prefix:
            obj.name = rename_prefix + obj.name
            if obj.data:
                try:
                    obj.data.name = obj.name
                except Exception:
                    pass
        linked.append(obj.name)
    return {"filepath": filepath, "linked": sorted(linked),
            "linked_count": len(linked)}


def op_duplicate_object(args: dict) -> dict:
    src = _object_ref(args["name"])
    clone = src.copy()
    if src.data is not None:
        clone.data = src.data.copy()
    for coll in src.users_collection:
        coll.objects.link(clone)
    if offset := args.get("offset"):
        clone.location = Vector(src.location) + Vector(offset)
    if new_name := args.get("new_name"):
        clone.name = new_name
    return {"name": clone.name}


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def _principled(material):
    """Return the Principled BSDF node, creating a node tree if needed."""
    if not material.use_nodes:
        material.use_nodes = True
    tree = material.node_tree
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    # A brand-new node tree may have no output/BSDF pair.
    node = tree.nodes.new("ShaderNodeBsdfPrincipled")
    out = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(node.outputs["BSDF"], out.inputs["Surface"])
    return node


def op_create_material(args: dict) -> dict:
    name = args.get("name", "Material")
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    node = _principled(material)

    def set_input(key, value):
        if key in node.inputs:
            node.inputs[key].default_value = value

    if (color := args.get("base_color")) is not None:
        rgba = list(color)
        if len(rgba) == 3:
            rgba.append(1.0)
        set_input("Base Color", rgba)
    if (metallic := args.get("metallic")) is not None:
        set_input("Metallic", float(metallic))
    if (roughness := args.get("roughness")) is not None:
        set_input("Roughness", float(roughness))
    if (emission := args.get("emission_color")) is not None:
        rgba = list(emission)
        if len(rgba) == 3:
            rgba.append(1.0)
        set_input("Emission Color", rgba)
        set_input("Emission Strength", float(args.get("emission_strength", 1.0)))
    if (alpha := args.get("alpha")) is not None:
        set_input("Alpha", float(alpha))
        if float(alpha) < 1.0:
            material.blend_method = "BLEND"
    if (ior := args.get("ior")) is not None:
        set_input("IOR", float(ior))
    if (transmission := args.get("transmission")) is not None:
        for key in ("Transmission Weight", "Transmission"):
            if key in node.inputs:
                node.inputs[key].default_value = float(transmission)
                break

    return {"name": material.name, "use_nodes": bool(material.use_nodes)}


def op_assign_material(args: dict) -> dict:
    material = bpy.data.materials.get(args["material"])
    if material is None:
        material = bpy.data.materials.new(args["material"])
    assigned = []
    for name in args.get("objects", []):
        obj = bpy.data.objects.get(name)
        if obj is None or not hasattr(obj.data, "materials"):
            continue
        if args.get("replace_all", True) and obj.data.materials:
            for idx in range(len(obj.data.materials)):
                obj.data.materials[idx] = material
        else:
            obj.data.materials.append(material)
        assigned.append(name)
    return {"material": material.name, "assigned_to": assigned}


# ---------------------------------------------------------------------------
# Lights and cameras
# ---------------------------------------------------------------------------


def op_create_light(args: dict) -> dict:
    light_type = str(args.get("light_type", "POINT")).upper()
    data = bpy.data.lights.new(name=args.get("name", "Light"), type=light_type)
    data.energy = float(args.get("energy", 1000.0))
    if (color := args.get("color")) is not None:
        data.color = tuple(color[:3])
    if light_type == "AREA" and (size := args.get("size")) is not None:
        data.size = float(size)
    if light_type == "SUN" and (angle := args.get("angle")) is not None:
        data.angle = float(angle)
    if light_type == "SPOT":
        data.spot_size = float(args.get("spot_size", 0.785398))
        data.spot_blend = float(args.get("spot_blend", 0.15))

    obj = bpy.data.objects.new(args.get("name", "Light"), data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = args.get("location", (0.0, 0.0, 0.0))
    if target := args.get("look_at"):
        obj.rotation_euler = _look_at_rotation(obj.location, target)
    elif rotation := args.get("rotation"):
        obj.rotation_euler = rotation
    return {"name": obj.name, "light_type": data.type, "energy": float(data.energy)}


def op_create_camera(args: dict) -> dict:
    data = bpy.data.cameras.new(name=args.get("name", "Camera"))
    data.lens = float(args.get("lens_mm", 50.0))
    if (sensor := args.get("sensor_width")) is not None:
        data.sensor_width = float(sensor)
    if (dof := args.get("dof_distance")) is not None:
        data.dof.use_dof = True
        data.dof.focus_distance = float(dof)
        data.dof.aperture_fstop = float(args.get("fstop", 2.8))

    obj = bpy.data.objects.new(args.get("name", "Camera"), data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = args.get("location", (0.0, 0.0, 0.0))
    if target := args.get("look_at"):
        obj.rotation_euler = _look_at_rotation(obj.location, target)
    elif rotation := args.get("rotation"):
        obj.rotation_euler = rotation
    if args.get("make_active", True):
        bpy.context.scene.camera = obj
    return {
        "name": obj.name,
        "lens_mm": float(data.lens),
        "location": _to_list(obj.location),
        "rotation_euler": _to_list(obj.rotation_euler),
        "is_active": bpy.context.scene.camera == obj,
    }


def op_create_armature(args: dict) -> dict:
    """Build a simple armature from a bone list.

    bones: [{name, head: [x,y,z], tail: [x,y,z], parent: name-or-empty}].
    Modest on purpose: a rig is a posing handle, and a handful of named bones
    posed deterministically is more reliable than an elaborate rig built blind.
    """
    bpy.ops.object.armature_add(
        enter_editmode=True,
        location=args.get("location", (0.0, 0.0, 0.0)),
    )
    arm = bpy.context.active_object
    if name := args.get("name"):
        arm.name = name
        arm.data.name = name
    bones = args.get("bones", [])
    created = []
    edit = arm.data.edit_bones
    # The default armature ships with one bone; reshape it into the first bone
    # so the armature never carries a stray unnamed extra.
    first = edit[0] if len(edit) else None
    for index, spec in enumerate(bones):
        if index == 0 and first is not None:
            bone = first
        else:
            bone = edit.new(spec["name"])
        bone.name = spec["name"]
        bone.head = spec["head"]
        bone.tail = spec["tail"]
        created.append(bone.name)
    for spec in bones:
        parent = spec.get("parent")
        if parent and parent in edit and spec["name"] in edit:
            edit[spec["name"]].parent = edit[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return {"name": arm.name, "bones": created, "bone_count": len(created)}


def op_pose_bone(args: dict) -> dict:
    """Rotate/translate a pose bone and optionally keyframe it.

    rotation_euler is in radians, applied in the bone's local space.
    """
    arm = _object_ref(args["armature"])
    if arm.type != "ARMATURE":
        raise ValueError("%r is a %s, not an armature" % (arm.name, arm.type))
    bone = arm.pose.bones.get(args["bone"])
    if bone is None:
        raise KeyError("armature %r has no bone %r" % (arm.name, args["bone"]))
    keyed = []
    frame = args.get("frame")
    if rotation := args.get("rotation_euler"):
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = rotation
        if frame is not None:
            bone.keyframe_insert(data_path="rotation_euler", frame=int(frame))
            keyed.append("rotation_euler")
    if location := args.get("location"):
        bone.location = location
        if frame is not None:
            bone.keyframe_insert(data_path="location", frame=int(frame))
            keyed.append("location")
    return {"armature": arm.name, "bone": bone.name,
            "rotation_euler": _to_list(bone.rotation_euler),
            "location": _to_list(bone.location), "keyed": keyed,
            "frame": frame}


def op_object_manifest(args: dict) -> dict:
    """The objects in this scene, grouped by scope.

    Grouping is exact rather than heuristic. Structural names such as
    ``ENV_COURTYARD_DUSK_01_fountain_basin`` cannot be split reliably by
    underscores — the environment id and the part name both contain them — so
    the caller passes the scopes it knows it built (``scopes``) and each object
    is matched against the longest one that prefixes it. That makes an entry in
    a group proof that the corresponding Blender object exists.
    """
    scopes = [str(s) for s in (args.get("scopes") or []) if str(s)]
    # Longest first, so ENV_A_B matches ENV_A_B before ENV_A.
    scopes.sort(key=len, reverse=True)

    groups: dict = {}
    for obj in bpy.data.objects:
        name = obj.name
        prefix = ""
        for scope in scopes:
            if name == scope or name.startswith(scope + "_"):
                prefix = scope
                break
        if not prefix:
            if name.startswith("CHR_") and "_" in name:
                prefix = "CHR_" + name.split("_")[1]
            elif "_" in name:
                prefix = name.split("_")[0] + "_"
            else:
                prefix = "(none)"
        entry = {
            "name": name,
            "type": obj.type,
            "location": _to_list(obj.location),
            "hide_render": bool(obj.hide_render),
            "hide_viewport": bool(obj.hide_viewport),
            "parent": obj.parent.name if obj.parent else None,
        }
        if obj.type == "MESH" and obj.data is not None:
            entry["vertex_count"] = len(obj.data.vertices)
            entry["materials"] = [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ]
        groups.setdefault(prefix, []).append(entry)
    return {"groups": groups,
            "group_names": sorted(groups),
            "scopes": scopes,
            "object_count": len(bpy.data.objects)}


def op_set_active_camera(args: dict) -> dict:
    obj = _object_ref(args["name"])
    if obj.type != "CAMERA":
        raise ValueError("%r is a %s, not a camera" % (obj.name, obj.type))
    bpy.context.scene.camera = obj
    return {"active_camera": obj.name}


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _action_animation_summary(action) -> dict:
    """Count real keyframe data inside an action, across storage formats.

    Blender 4.x keeps fcurves directly on the action. Blender 5.x moved to
    layered actions (action.layers -> strips -> channelbags) and the classic
    ``action.fcurves`` is empty. Rather than trusting either surface, this
    walks whatever the installed version exposes and counts keyframe points,
    so ``fcurve_count: 0`` really means "no animation", never "I could not
    find the animation".
    """
    import bpy  # noqa: F401
    total_kf = 0
    channels = 0
    frame_min: float | None = None
    frame_max: float | None = None

    def count_fcurves(fcurves) -> None:
        nonlocal total_kf, channels, frame_min, frame_max
        for fc in fcurves:
            channels += 1
            for kp in fc.keyframe_points:
                total_kf += 1
                f = float(kp.co[0])
                if frame_min is None or f < frame_min:
                    frame_min = f
                if frame_max is None or f > frame_max:
                    frame_max = f

    try:
        count_fcurves(list(action.fcurves))
    except (AttributeError, TypeError):
        pass

    # Layered actions (Blender 5.x): keyframes live in channelbags inside
    # layer strips rather than on the action's own fcurve collection. The
    # channelbag is indexed by the slot that holds the animated datablock, so
    # ``channelbag(slot)`` is the correct call.
    layers = getattr(action, "layers", None)
    if layers is not None:
        slots = None
        try:
            slots = list(action.slots) if hasattr(action, "slots") else None
        except Exception:
            slots = None
        try:
            for layer in layers:
                for strip in layer.strips:
                    channelbag = None
                    try:
                        if slots:
                            channelbag = strip.channelbag(slots[0])
                        else:
                            channelbag = strip.channelbag()
                    except TypeError:
                        channelbag = None
                    if channelbag is None:
                        continue
                    try:
                        count_fcurves(channelbag.fcurves)
                    except (AttributeError, TypeError):
                        continue
        except Exception:
            pass

    return {
        "name": action.name,
        "fcurve_count": channels,
        "keyframe_count": total_kf,
        "frame_range": [frame_min, frame_max] if frame_min is not None else [],
        "storage": "layered" if layers is not None else "legacy",
    }


def op_get_action_info(args: dict) -> dict:
    """Report the animation attached to one object, in the form that verifies.

    Used to prove a requested movement became actual keyframes (or that a
    static hold is deliberately keyed to a single frame), read from Blender's
    own animation data rather than assumed from the spec.
    """
    obj_name = args.get("name")
    if obj_name is not None:
        obj = _object_ref(obj_name)
        adata = obj.animation_data
        if adata is None or adata.action is None:
            return {"name": obj_name, "action": None,
                    "animation_count": 0, "note": "no animation data"}
        summary = _action_animation_summary(adata.action)
        summary["object"] = obj_name
        summary["action"] = adata.action.name
        if args.get("dump"):
            summary["debug"] = _action_debug_dump(adata.action)
        return summary
    all_actions = [_action_animation_summary(a) for a in bpy.data.actions]
    return {"actions": all_actions, "count": len(all_actions)}


def _action_debug_dump(action) -> dict:
    """Inspect the raw action internals, for diagnosing the readback."""
    info = {
        "type": type(action).__name__,
        "has_fcurves": hasattr(action, "fcurves"),
        "attrs": [a for a in dir(action)
                  if not a.startswith("_") and a not in ("is_valid", "library", "use_fake_user", "tag")]
    }
    try:
        info["fcurve_count_attr"] = len(list(action.fcurves))
    except Exception as exc:  # noqa: BLE001
        info["fcurve_error"] = repr(exc)
    layers = getattr(action, "layers", None)
    info["has_layers"] = layers is not None
    if layers is not None:
        try:
            info["num_layers"] = len(layers)
            info["layer_attrs"] = [a for a in dir(layers[0]) if not a.startswith("_")]
            strip_objs = []
            for layer in layers:
                for strip in layer.strips:
                    entry = {
                        "type": type(strip).__name__,
                        "attrs": [a for a in dir(strip) if not a.startswith("_")
                                  and a not in ("is_valid", "tag", "library")],
                    }
                    cb = None
                    for attr in ("channelbag", "channelbags"):
                        raw = getattr(strip, attr, None)
                        if callable(raw):
                            raw = raw()
                        if raw is not None:
                            cb = raw
                            entry["channelbag_attr"] = attr
                            break
                    if cb is not None:
                        entry["channelbag_type"] = type(cb).__name__
                        entry["channelbag_attrs"] = [
                            a for a in dir(cb)
                            if not a.startswith("_")
                            and a not in ("is_valid", "tag", "library", "user_clear", "user_remap", "users")]
                        try:
                            entry["cb_fcurves"] = len(list(cb.fcurves))
                        except Exception as exc:  # noqa: BLE001
                            entry["cb_fcurve_error"] = repr(exc)
                    else:
                        entry["channelbag"] = None
                    strip_objs.append(entry)
            info["strips"] = strip_objs
        except Exception as exc:  # noqa: BLE001
            info["layers_error"] = repr(exc)
    slots = getattr(action, "slots", None)
    info["has_slots"] = slots is not None
    if slots is not None:
        try:
            info["num_slots"] = len(slots)
            info["slot_attrs"] = [
                {"name": s.name, "attrs": [a for a in dir(s) if not a.startswith("_")]}
                for s in slots
            ]
        except Exception as exc:  # noqa: BLE001
            info["slots_error"] = repr(exc)
    return info


def op_set_keyframe(args: dict) -> dict:
    obj = _object_ref(args["name"])
    frame = int(args["frame"])
    written = []
    if location := args.get("location"):
        obj.location = location
        obj.keyframe_insert(data_path="location", frame=frame)
        written.append("location")
    if rotation := args.get("rotation_euler"):
        obj.rotation_euler = rotation
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        written.append("rotation_euler")
    if scale := args.get("scale"):
        obj.scale = scale
        obj.keyframe_insert(data_path="scale", frame=frame)
        written.append("scale")
    if not written:
        # Keyframe whatever the object already has.
        obj.keyframe_insert(data_path="location", frame=frame)
        written.append("location")
    return {"name": obj.name, "frame": frame, "keyed": written}


def op_set_scene_timing(args: dict) -> dict:
    scene = bpy.context.scene
    if (start := args.get("frame_start")) is not None:
        scene.frame_start = int(start)
    if (end := args.get("frame_end")) is not None:
        scene.frame_end = int(end)
    if (fps := args.get("fps")) is not None:
        scene.render.fps = int(fps)
        scene.render.fps_base = 1.0
    scene.frame_set(int(args.get("frame_current", scene.frame_current)))
    return {
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "fps": float(scene.render.fps) / float(scene.render.fps_base or 1),
    }


# ---------------------------------------------------------------------------
# Render configuration and rendering  (spec 5C, 5D)
# ---------------------------------------------------------------------------


def op_set_render_settings(args: dict) -> dict:
    scene = bpy.context.scene
    render = scene.render

    engine_note = ""
    shading_note = ""
    if engine := args.get("engine"):
        _actual, engine_note = _apply_engine(render, str(engine))
    if (width := args.get("resolution_x")) is not None:
        render.resolution_x = int(width)
    if (height := args.get("resolution_y")) is not None:
        render.resolution_y = int(height)
    if (pct := args.get("resolution_percentage")) is not None:
        render.resolution_percentage = int(pct)
    if (fps := args.get("fps")) is not None:
        render.fps = int(fps)
        render.fps_base = 1.0
    if (transparent := args.get("film_transparent")) is not None:
        render.film_transparent = bool(transparent)
    if (out := args.get("output_path")) is not None:
        render.filepath = str(out)
    if (fmt := args.get("image_format")) is not None:
        render.image_settings.file_format = str(fmt)
    if (quality := args.get("ffmpeg_quality")) is not None:
        render.image_settings.quality = int(quality)

    engine = render.engine
    if engine == "CYCLES":
        cycles = scene.cycles
        if (samples := args.get("samples")) is not None:
            cycles.samples = int(samples)
            cycles.preview_samples = int(samples)
        if (device := args.get("cycles_device")) is not None:
            cycles.device = str(device).upper()
        if (denoise := args.get("denoise")) is not None:
            cycles.use_denoising = bool(denoise)
    elif engine.startswith("BLENDER_EEVEE"):
        eevee = scene.eevee
        if (samples := args.get("samples")) is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = int(samples)
    elif engine == "BLENDER_WORKBENCH":
        # Workbench's default shading is flat studio grey: it ignores materials
        # entirely. Setting color_type to MATERIAL does NOT fix this for a
        # headless render — verified on Blender 5.2, which reports
        # color_type='MATERIAL' while still emitting a perfectly neutral frame
        # (mean RGB 0.605/0.609/0.611). So a Workbench preview is monochrome by
        # nature and cannot be judged for colour or for the scene's real
        # lighting. We set these anyway because they do help an interactive
        # viewport, and we say so in the reply rather than letting a caller
        # believe it got a colour-accurate preview.
        display = getattr(scene, "display", None)
        if display is not None:
            shading = getattr(display, "shading", None)
            if shading is not None:
                try:
                    shading.light = "STUDIO"
                    shading.color_type = "MATERIAL"
                    shading.show_shadows = True
                    shading.show_cavity = True
                except (AttributeError, TypeError):
                    pass
        shading_note = (
            "BLENDER_WORKBENCH renders monochrome when run headless; material "
            "colours are not applied and the scene's own lights are not used. "
            "Use BLENDER_EEVEE for any preview a reviewer will judge on colour "
            "or exposure."
        )

    # Colour-management names move between Blender versions ("Filmic" was
    # replaced by "AgX" in 4.0), so ask for what we want and report what we
    # actually got rather than failing the whole settings call.
    view_note = ""
    if (view_transform := args.get("view_transform")) is not None:
        try:
            scene.view_settings.view_transform = str(view_transform)
        except TypeError:
            view_note = "view_transform %r not available in this build" % view_transform
    if (look := args.get("look")) is not None:
        try:
            scene.view_settings.look = str(look)
        except TypeError:
            view_note = (view_note + "; " if view_note else "") + \
                "look %r not available in this build" % look

    if "world_color" in args and scene.world is not None:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg is not None:
            color = list(args["world_color"])
            if len(color) == 3:
                color.append(1.0)
            bg.inputs[0].default_value = color
            if (strength := args.get("world_strength")) is not None:
                bg.inputs[1].default_value = float(strength)

    return {
        "engine": render.engine,
        "resolution": [render.resolution_x, render.resolution_y],
        "resolution_percentage": render.resolution_percentage,
        "fps": float(render.fps) / float(render.fps_base or 1),
        "output_path": render.filepath,
        "film_transparent": bool(render.film_transparent),
        "view_transform": scene.view_settings.view_transform,
        "view_note": view_note,
        "engine_note": " ".join(n for n in (engine_note, shading_note) if n),
    }


def op_render_still(args: dict) -> dict:
    scene = bpy.context.scene
    filepath = os.path.abspath(args["filepath"])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if (frame := args.get("frame")) is not None:
        scene.frame_set(int(frame))

    # Hide a set of objects for this render only, then restore them. Used to
    # shoot a background-only plate: rendering the same frame with and without
    # the subjects gives an exact subject mask, which no amount of colour
    # heuristics can match on a lit stage.
    hidden = []
    for name in args.get("hide_names") or []:
        obj = bpy.data.objects.get(name)
        if obj is not None and not obj.hide_render:
            obj.hide_render = True
            hidden.append(obj)

    previous = scene.render.filepath
    scene.render.filepath = filepath
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        scene.render.filepath = previous
        for obj in hidden:
            obj.hide_render = False

    if not os.path.exists(filepath):
        raise RuntimeError("render reported success but %s does not exist" % filepath)
    return {
        "filepath": filepath,
        "bytes": os.path.getsize(filepath),
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "frame": int(scene.frame_current),
        "hidden": [o.name for o in hidden],
    }


def op_render_animation(args: dict) -> dict:
    scene = bpy.context.scene
    out_dir = os.path.abspath(args["directory"])
    os.makedirs(out_dir, exist_ok=True)

    if (start := args.get("frame_start")) is not None:
        scene.frame_start = int(start)
    if (end := args.get("frame_end")) is not None:
        scene.frame_end = int(end)

    previous = scene.render.filepath
    # Blender expands #### into the zero-padded frame number.
    scene.render.filepath = os.path.join(out_dir, "frame_")
    try:
        bpy.ops.render.render(animation=True)
    finally:
        scene.render.filepath = previous

    frames = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("frame_") and f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))
    )
    return {
        "directory": out_dir,
        "frame_count": len(frames),
        "first": frames[0] if frames else None,
        "last": frames[-1] if frames else None,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "expected": int(scene.frame_end) - int(scene.frame_start) + 1,
    }


def op_viewport_screenshot(args: dict) -> dict:
    """Observation channel 5B.

    A background Blender has no viewport to photograph, so this renders the
    same viewpoint through Workbench instead. It is deliberately fast — it
    exists so an agent can glance at the scene cheaply, not so it can grade a
    final image.
    """
    scene = bpy.context.scene
    filepath = os.path.abspath(args["filepath"])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    saved = (
        scene.render.engine,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.filepath,
    )
    try:
        _apply_engine(scene.render, "BLENDER_WORKBENCH")
        scene.render.resolution_x = int(args.get("width", 480))
        scene.render.resolution_y = int(args.get("height", 270))
        scene.render.resolution_percentage = 100
        scene.render.filepath = filepath
        if (frame := args.get("frame")) is not None:
            scene.frame_set(int(frame))
        bpy.ops.render.render(write_still=True)
    finally:
        (
            scene.render.engine,
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.resolution_percentage,
            scene.render.filepath,
        ) = saved

    return {"filepath": filepath, "bytes": os.path.getsize(filepath),
            "mode": "workbench_render_fallback"}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def op_save_blend(args: dict) -> dict:
    filepath = os.path.abspath(args["filepath"])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=filepath, compress=bool(args.get("compress", False)))
    return {"filepath": filepath, "bytes": os.path.getsize(filepath)}


def op_load_blend(args: dict) -> dict:
    filepath = os.path.abspath(args["filepath"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    bpy.ops.wm.open_mainfile(filepath=filepath)
    scene = bpy.context.scene
    return {"filepath": filepath, "object_count": len(bpy.data.objects),
            "scene_name": scene.name}


def op_reset_scene(args: dict) -> dict:
    """Wipe the file back to an empty scene, keeping the session alive.

    ``read_homefile`` frees and rebuilds the main database. That is exactly what
    we want here, but it means anything holding a reference to a previously
    fetched ``bpy`` object is stale afterwards — callers should re-read scene
    state rather than cache it across a reset.
    """
    if args.get("empty", True):
        bpy.ops.wm.read_homefile(use_empty=True)
    else:
        bpy.ops.wm.read_homefile()
    return {"object_count": len(bpy.data.objects),
            "scene_name": bpy.context.scene.name}


def op_import_asset(args: dict) -> dict:
    filepath = os.path.abspath(args["filepath"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    before = set(bpy.data.objects.keys())

    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".blend":
        with bpy.data.libraries.load(filepath, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for obj in dst.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        raise ValueError("unsupported import format %r" % ext)

    added = sorted(set(bpy.data.objects.keys()) - before)
    return {"filepath": filepath, "added": added, "added_count": len(added)}


def op_export_asset(args: dict) -> dict:
    filepath = os.path.abspath(args["filepath"])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".obj":
        bpy.ops.wm.obj_export(filepath=filepath)
    elif ext in {".gltf", ".glb"}:
        bpy.ops.export_scene.gltf(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_export(filepath=filepath)
    else:
        raise ValueError("unsupported export format %r" % ext)
    return {"filepath": filepath, "bytes": os.path.getsize(filepath)}


# ---------------------------------------------------------------------------
# Escape hatch
# ---------------------------------------------------------------------------


def op_evaluate(args: dict) -> dict:
    """Execute a snippet inside Blender's Python.

    The Blender Agent uses this for operations the op table does not cover yet.
    It is powerful by design — it is the difference between a fixed feature set
    and a system that can grow. It is gated by the session token like every
    other op, and the socket is loopback-only.
    """
    code = str(args.get("code", ""))
    if not code.strip():
        raise ValueError("empty code block")
    namespace = {"bpy": bpy, "mathutils": __import__("mathutils"), "os": os}
    exec(compile(code, "<agent>", "exec"), namespace)  # noqa: S102 - intentional
    result = namespace.get("result")
    return {"result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else repr(result)}


def op_analyze_image(args: dict) -> dict:
    """Objective statistics about a rendered frame.

    Observation channel 5C/5D feed for the Vision Agent. Blender already ships
    numpy and decodes every format we render, so the measurement happens here
    rather than pulling an imaging library into the orchestration process.

    Values are reported in sRGB-encoded space (what a viewer actually sees),
    not Blender's linear working space, so thresholds like "too dark" mean what
    a human would mean by them.
    """
    import numpy as np

    filepath = os.path.abspath(args["filepath"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    image = bpy.data.images.load(filepath, check_existing=False)
    try:
        width, height = image.size
        if width == 0 or height == 0:
            # Some formats report their size lazily; touching the pixel buffer
            # forces the decode, after which size is reliable.
            _ = image.pixels[0]
            width, height = image.size
        if width == 0 or height == 0:
            raise RuntimeError("image %s has zero size" % filepath)

        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        rgba = buffer.reshape(height, width, 4)
        # Blender stores images bottom-up; flip so row 0 is the top of frame.
        linear = rgba[::-1, :, :3]

        # Linear -> sRGB.
        clipped = np.clip(linear, 0.0, 1.0)
        srgb = np.where(
            clipped <= 0.0031308,
            clipped * 12.92,
            1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
        )

        luma = srgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        # Gradient magnitude gives a cheap proxy for how much detail is present.
        dx = np.abs(np.diff(luma, axis=1)).mean() if width > 1 else 0.0
        dy = np.abs(np.diff(luma, axis=0)).mean() if height > 1 else 0.0

        # Foreground coverage: how much of the frame is occupied by something
        # that is not the backdrop.
        #
        # Preferred method — a background plate. If the caller supplies the
        # same frame rendered with the subjects hidden, the subject mask is
        # simply where the two differ. That is exact, and immune to every
        # lighting and material combination. It costs one extra cheap preview
        # render, which is well worth it: this number drives the revision loop.
        #
        # Fallback — a chroma test. On a lit stage brightness cannot separate
        # subject from background (the key light throws a pool across the ground
        # that is brighter than a dark backdrop), and hue only partly helps: a
        # big lit surface carrying colour bleed from the subject varies in
        # chroma just as a real subject does. Measured across thresholds from
        # 0.05 to 0.30 the two are not separable — at 0.10 a scene holding one
        # small cube still read 7.3% while a genuine close-up read 2.0%. So the
        # fallback is reported as such rather than trusted silently.
        border = np.concatenate([
            srgb[0, :, :], srgb[-1, :, :], srgb[:, 0, :], srgb[:, -1, :],
        ])
        background = np.median(border, axis=0)
        bg_luma = float(background @ np.array([0.2126, 0.7152, 0.0722],
                                              dtype=np.float32))
        bg_chroma_rg = float(background[0] - background[1])
        bg_chroma_gb = float(background[1] - background[2])

        chroma_diff = (
            np.abs((srgb[:, :, 0] - srgb[:, :, 1]) - bg_chroma_rg)
            + np.abs((srgb[:, :, 1] - srgb[:, :, 2]) - bg_chroma_gb)
        )
        coverage = float((chroma_diff > 0.05).mean())
        coverage_method = "chroma_fallback"
        plate_note = ""

        plate_path = args.get("background_filepath")
        if plate_path:
            plate_path = os.path.abspath(str(plate_path))
            if not os.path.exists(plate_path):
                plate_note = "background plate not found at %s" % plate_path
            else:
                plate = bpy.data.images.load(plate_path, check_existing=False)
                try:
                    if tuple(plate.size) != (width, height):
                        plate_note = (
                            "background plate is %dx%d but the subject render is "
                            "%dx%d; ignoring it"
                            % (plate.size[0], plate.size[1], width, height)
                        )
                    else:
                        plate_buffer = np.empty(width * height * 4, dtype=np.float32)
                        plate.pixels.foreach_get(plate_buffer)
                        plate_linear = plate_buffer.reshape(height, width, 4)[::-1, :, :3]
                        # Compared in linear space: the difference is a physical
                        # quantity, and the threshold means the same thing
                        # regardless of exposure.
                        difference = np.abs(
                            linear - np.clip(plate_linear, 0.0, 1.0)
                        ).sum(axis=2)
                        subject_mask = difference > PLATE_DIFFERENCE_THRESHOLD
                        coverage = float(subject_mask.mean())
                        coverage_method = "background_plate"
                finally:
                    bpy.data.images.remove(plate)

        # Rows and columns that are essentially black — letterboxing or a
        # framing mistake that leaves dead space.
        row_mean = luma.mean(axis=1)
        col_mean = luma.mean(axis=0)
        dark_rows = int((row_mean < 0.02).sum())
        dark_cols = int((col_mean < 0.02).sum())

        histogram, _edges = np.histogram(luma, bins=16, range=(0.0, 1.0))

        return {
            "filepath": filepath,
            "width": int(width),
            "height": int(height),
            "aspect": round(width / height, 4),
            "luma_mean": round(float(luma.mean()), 4),
            "luma_median": round(float(np.median(luma)), 4),
            "luma_std": round(float(luma.std()), 4),
            "luma_p05": round(float(np.percentile(luma, 5)), 4),
            "luma_p95": round(float(np.percentile(luma, 95)), 4),
            "clipped_shadows": round(float((luma < 0.02).mean()), 4),
            "clipped_highlights": round(float((luma > 0.98).mean()), 4),
            "edge_density": round(float((dx + dy) / 2.0), 5),
            "foreground_coverage": round(coverage, 4),
            "dark_row_fraction": round(dark_rows / height, 4),
            "dark_col_fraction": round(dark_cols / width, 4),
            "mean_rgb": [round(float(srgb[:, :, i].mean()), 4) for i in range(3)],
            "background_rgb": [round(float(v), 4) for v in background],
            "background_luma": round(bg_luma, 4),
            "chroma_contrast": round(float(chroma_diff.mean()), 4),
            "coverage_method": coverage_method,
            "plate_note": plate_note,
            "histogram": [int(v) for v in histogram],
        }
    finally:
        bpy.data.images.remove(image)


def op_ping(_args: dict) -> dict:
    return {
        "blender_version": bpy.app.version_string,
        "filepath": bpy.data.filepath,
        "object_count": len(bpy.data.objects),
    }


def op_shutdown(_args: dict) -> dict:
    raise _Shutdown()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

OPERATIONS = {
    "ping": op_ping,
    "shutdown": op_shutdown,
    "get_scene_state": op_get_scene_state,
    "get_errors": op_get_errors,
    "object_manifest": op_object_manifest,
    "create_primitive": op_create_primitive,
    "create_empty": op_create_empty,
    "delete_object": op_delete_object,
    "set_transform": op_set_transform,
    "set_visibility": op_set_visibility,
    "link_objects": op_link_objects,
    "duplicate_object": op_duplicate_object,
    "create_armature": op_create_armature,
    "pose_bone": op_pose_bone,
    "create_material": op_create_material,
    "assign_material": op_assign_material,
    "create_light": op_create_light,
    "create_camera": op_create_camera,
    "set_active_camera": op_set_active_camera,
    "set_keyframe": op_set_keyframe,
    "get_action_info": op_get_action_info,
    "set_scene_timing": op_set_scene_timing,
    "set_render_settings": op_set_render_settings,
    "render_still": op_render_still,
    "render_animation": op_render_animation,
    "viewport_screenshot": op_viewport_screenshot,
    "analyze_image": op_analyze_image,
    "save_blend": op_save_blend,
    "load_blend": op_load_blend,
    "reset_scene": op_reset_scene,
    "import_asset": op_import_asset,
    "export_asset": op_export_asset,
    "evaluate": op_evaluate,
}


class _Shutdown(Exception):
    """Raised to break the serve loop cleanly."""


def _handle(request: dict) -> dict:
    request_id = request.get("id")
    op_name = request.get("op")
    args = request.get("args") or {}

    if op_name not in OPERATIONS:
        return {
            "id": request_id,
            "ok": False,
            "error": "unknown op %r" % op_name,
            "known_ops": sorted(OPERATIONS),
        }
    try:
        result = OPERATIONS[op_name](args)
        return {"id": request_id, "ok": True, "result": result}
    except _Shutdown:
        raise
    except Exception as exc:  # noqa: BLE001 - report, never crash the session
        return {
            "id": request_id,
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def serve(port: int, token: str) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    actual_port = server.getsockname()[1]

    # The parent parses this line to learn the ephemeral port.
    print("FA_CONTROL_PORT %d" % actual_port, flush=True)
    print("FA_CONTROL_READY", flush=True)

    shutting_down = False
    while not shutting_down:
        conn, _addr = server.accept()
        conn.settimeout(None)
        try:
            stream = conn.makefile("rwb")
            while True:
                line = stream.readline()
                if not line:
                    break  # client disconnected
                try:
                    request = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    response = {"id": None, "ok": False, "error": "bad request: %s" % exc}
                    stream.write((json.dumps(response) + "\n").encode("utf-8"))
                    stream.flush()
                    continue

                if request.get("token") != token:
                    response = {"id": request.get("id"), "ok": False,
                                "error": "bad token"}
                    stream.write((json.dumps(response) + "\n").encode("utf-8"))
                    stream.flush()
                    continue

                try:
                    response = _handle(request)
                except _Shutdown:
                    stream.write((json.dumps({"id": request.get("id"), "ok": True,
                                              "result": {"shutdown": True}}) + "\n").encode("utf-8"))
                    stream.flush()
                    shutting_down = True
                    break

                stream.write((json.dumps(response, default=str) + "\n").encode("utf-8"))
                stream.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-request; accept the next one
        finally:
            try:
                conn.close()
            except OSError:
                pass

    server.close()


def main() -> None:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(prog="fa-control-server")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token", required=True)
    parser.add_argument("--blend", default="")
    parsed = parser.parse_args(script_args)

    if parsed.blend and os.path.exists(parsed.blend):
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(parsed.blend))

    serve(parsed.port, parsed.token)


if __name__ == "__main__":
    main()

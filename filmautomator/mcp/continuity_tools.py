"""Canonical characters and environments, and the audio capabilities.

Continuity is the thing that separates "a series of renders" from "a film". If
Akira is a different height, colour and silhouette in every shot, the shots do
not cut together no matter how good each one is alone.

So characters and environments are first-class, persistent records with stable
IDs, and shots reference them rather than restating them. A shot that says
"use canonical character Akira" inherits the same proportions, palette and
continuity constraints every time.

The records here are declarative: they describe what must stay consistent and
are stored in the project database alongside the shots. What Blender actually
instantiates from them is a separate concern, handled by the Blender Agent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..audio import (
    AUDIO_CAPABILITIES,
    dialogue_script,
    generate_ambience,
    generate_music,
    generate_sfx,
    generate_tts,
)
from ..core.task import QAStatus
from .preview import client_accepts_images
from .protocol import json_text, tool_result
from .tools import (
    ToolContext,
    fail,
    ok,
    optional_float,
    require_str,
    tool,
)

log = logging.getLogger(__name__)


# ===========================================================================
# CANONICAL CHARACTERS
# ===========================================================================


@tool("define_character",
      "Define or update a canonical character. Characters persist for the whole "
      "project so a shot can reference 'Akira' instead of inventing someone new. "
      "Record proportions, appearance, clothing, colours, voice and continuity "
      "constraints; later shots that reference this character inherit them.",
      properties={
          "project_id": {"type": "string"},
          "name": {"type": "string"},
          "height_m": {"type": "number"},
          "build": {"type": "string", "description": "slim, average, heavy..."},
          "appearance": {"type": "string"},
          "face": {"type": "string"},
          "hair": {"type": "string"},
          "clothing": {"type": "string"},
          "accessories": {"type": "string"},
          "colors": {"type": "object",
                     "description": "e.g. {\"hair\": \"#1a1a1a\", "
                                    "\"jacket\": \"#2b3a55\"}"},
          "materials": {"type": "string"},
          "voice": {"type": "string",
                    "description": "Voice identity: pitch, timbre, accent."},
          "behavior": {"type": "string"},
          "continuity": {"type": "object",
                         "description": "Constraints that must hold in every "
                                        "shot, e.g. {\"scar_left_cheek\": true, "
                                        "\"always_wets_hair\": true}"},
      },
      required=["project_id", "name"])
def tool_define_character(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    name = require_str(args, "name").strip()
    ctx.project_and_workspace(project_id)
    if not name:
        return fail("a character needs a name")

    canonical = {
        "height_m": optional_float(args, "height_m", 1.8) or 1.8,
        "build": require_str(args, "build"),
        "appearance": require_str(args, "appearance"),
        "face": require_str(args, "face"),
        "hair": require_str(args, "hair"),
        "clothing": require_str(args, "clothing"),
        "accessories": require_str(args, "accessories"),
        "colors": args.get("colors") or {},
        "materials": require_str(args, "materials"),
    }
    continuity = {
        "voice": require_str(args, "voice"),
        "behavior": require_str(args, "behavior"),
        "constraints": args.get("continuity") or {},
    }
    canonical = {k: v for k, v in canonical.items() if v not in ("", {}, None)}
    continuity = {k: v for k, v in continuity.items() if v not in ("", {}, None)}

    existing = ctx.db.get_character(project_id, name)
    character_id = ctx.db.upsert_character(
        project_id, name, canonical=canonical, continuity=continuity,
    )
    ctx.emit(project_id, "character.defined",
             f"canonical character {name!r} defined"
             + (" (updated)" if existing else ""),
             agent="ai_director", payload={"name": name, "canonical": canonical})
    return ok({
        "character_id": character_id,
        "name": name,
        "updated": bool(existing),
        "canonical": canonical,
        "continuity": continuity,
        "usage": (f"reference this character in a shot by adding a subject named "
                  f"{name!r}, or set subject.asset_id to {character_id!r}"),
    })


@tool("get_character", "One canonical character, with its continuity rules.",
      properties={"project_id": {"type": "string"}, "name": {"type": "string"}},
      required=["project_id", "name"])
def tool_get_character(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    name = require_str(args, "name")
    character = ctx.db.get_character(project_id, name)
    if character is None:
        return fail(f"no canonical character named {name!r} in {project_id}")
    return ok(character)


@tool("list_characters", "Every canonical character defined in a project.",
      properties={"project_id": {"type": "string"}},
      required=["project_id"])
def tool_list_characters(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    ctx.project_and_workspace(project_id)
    characters = ctx.db.list_characters(project_id)
    return ok({"characters": characters, "count": len(characters)})


# ===========================================================================
# CANONICAL ENVIRONMENTS
# ===========================================================================


@tool("define_environment",
      "Define or update a canonical environment (a reusable location). Shots "
      "that reference the same environment share its layout, lighting baseline, "
      "weather and props, which is what keeps them looking like the same place.",
      properties={
          "project_id": {"type": "string"},
          "environment_id": {"type": "string",
                             "description": "e.g. ENV_TOKYO_RUINS_01"},
          "name": {"type": "string"},
          "description": {"type": "string"},
          "geometry": {"type": "string"},
          "materials": {"type": "string"},
          "lighting_baseline": {"type": "string",
                                "description": "e.g. 'blue moonlight from "
                                               "camera-left, warm orange "
                                               "building light camera-right'"},
          "layout": {"type": "string"},
          "weather": {"type": "string", "description": "e.g. 'heavy rain'"},
          "props": {"type": "array", "items": {"type": "string"}},
          "objects": {"type": "array",
                      "description": "Structured placement plan — the "
                                     "authoritative spatial source. Each "
                                     "entry: {id, type, position: [x,y,z], "
                                     "rotation (z radians), material, "
                                     "dimensions: {width, depth, height}}. "
                                     "Types with real geometry: bench, tree, "
                                     "wall, column, crate, barrel, rock, "
                                     "bush, fence, sign, planter, fountain, "
                                     "lantern, archway.",
                      "items": {"type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "position": {"type": "array",
                                                 "items": {"type": "number"}},
                                    "rotation": {"type": "number"},
                                    "material": {"type": "string"},
                                    "dimensions": {"type": "object"},
                                }}},
          "continuity": {"type": "object",
                         "description": "What must stay identical across shots."},
      },
      required=["project_id", "environment_id"])
def tool_define_environment(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    environment_id = require_str(args, "environment_id").strip()
    _project, workspace = ctx.project_and_workspace(project_id)
    if not environment_id:
        return fail("an environment needs an environment_id")

    spec = {
        "name": require_str(args, "name") or environment_id,
        "description": require_str(args, "description"),
        "geometry": require_str(args, "geometry"),
        "materials": require_str(args, "materials"),
        "lighting_baseline": require_str(args, "lighting_baseline"),
        "layout": require_str(args, "layout"),
        "weather": require_str(args, "weather"),
        "props": list(args.get("props") or []),
        "objects": [o for o in (args.get("objects") or [])
                    if isinstance(o, dict)],
        "continuity": args.get("continuity") or {},
    }
    spec = {k: v for k, v in spec.items() if v not in ("", [], {}, None)}

    # The definition is written to the project's environments directory as well
    # as the registry, so it is a real record on disk that survives a lost
    # database and can be read back by the Blender Agent.
    spec_path = workspace.asset_dir("environments") / f"{environment_id}.json"
    workspace.write_json(spec_path.relative_to(workspace.root), spec)

    ctx.db.upsert_asset(project_id, "environment", environment_id,
                        str(spec_path), canonical=True,
                        metadata={"environment_id": environment_id, **spec})
    ctx.emit(project_id, "environment.defined",
             f"canonical environment {environment_id} defined",
             agent="ai_director", payload=spec)
    return ok({
        "environment_id": environment_id,
        "spec": spec,
        "path": str(spec_path),
        "usage": ("reference it from a shot with create_shot(environment="
                  f"{environment_id!r})"),
    })


@tool("list_environments", "Every canonical environment defined in a project.",
      properties={"project_id": {"type": "string"}},
      required=["project_id"])
def tool_list_environments(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    ctx.project_and_workspace(project_id)
    environments = ctx.db.find_assets(project_id, "environment")
    return ok({"environments": environments, "count": len(environments)})


# ===========================================================================
# AUDIO
# ===========================================================================


@tool("generate_audio",
      "Generate audio for a project. Supports dialogue (text -> timed spoken "
      "audio), music (from a mood/BPM/key/intensity brief), SFX and ambience. "
      "Generation is local, deterministic and needs no model server; every "
      "artifact is labelled synthetic so it is never mistaken for a finished "
      "recording. Real models can be plugged into these capabilities later "
      "without changing this interface.",
      properties={
          "project_id": {"type": "string"},
          "kind": {"type": "string",
                   "description": "dialogue, music, sfx or ambience."},
          "name": {"type": "string", "description": "Artifact name."},
          "text": {"type": "string", "description": "Dialogue line."},
          "character": {"type": "string"},
          "voice": {"type": "string",
                    "description": "neutral, low or high."},
          "duration_s": {"type": "number"},
          "bpm": {"type": "number"},
          "key": {"type": "string", "description": "major or minor."},
          "mood": {"type": "string"},
          "instrumentation": {"type": "string"},
          "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
          "environment": {"type": "string",
                          "description": "For ambience: night_city, rain, "
                                         "forest, room, wind."},
      },
      required=["project_id", "kind"])
def tool_generate_audio(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    project, workspace = ctx.project_and_workspace(project_id)
    kind = (require_str(args, "kind") or "").strip().lower()
    name = require_str(args, "name") or kind
    if kind not in {"dialogue", "music", "sfx", "ambience"}:
        return fail(f"unknown audio kind {kind!r}; expected one of "
                    "dialogue, music, sfx, ambience")

    directory = workspace.audio_dir_for(
        {"dialogue": "dialogue", "music": "music",
         "sfx": "sfx", "ambience": "sfx"}[kind]
    )
    target = directory / f"{name}.wav"

    if kind == "dialogue":
        text = require_str(args, "text")
        if not text.strip():
            return fail("dialogue needs a text field with the line to speak")
        artifact = generate_tts(
            text, target, voice=require_str(args, "voice") or "neutral",
            duration_s=optional_float(args, "duration_s", 0.0) or 0.0,
        )
    elif kind == "music":
        artifact = generate_music(
            target,
            duration_s=optional_float(args, "duration_s", 8.0) or 8.0,
            bpm=int(optional_float(args, "bpm", 72) or 72),
            key=require_str(args, "key") or "minor",
            mood=require_str(args, "mood") or "tense",
            instrumentation=require_str(args, "instrumentation") or "synth",
            intensity=optional_float(args, "intensity", 0.5) if
            args.get("intensity") is not None else 0.5,
        )
    elif kind == "sfx":
        artifact = generate_sfx(
            require_str(args, "name") or name, target,
            duration_s=optional_float(args, "duration_s", 0.6) or 0.6,
        )
    else:
        artifact = generate_ambience(
            target,
            duration_s=optional_float(args, "duration_s", 8.0) or 8.0,
            environment=require_str(args, "environment") or "night_city",
        )

    ctx.db.register_artifact(
        project_id, "audio", str(artifact.path),
        f"{kind}: {name}", shot_id=require_str(args, "character"),
        metadata=artifact.to_dict(),
    )
    ctx.emit(project_id, "audio.completed",
             f"{kind} audio generated: {Path(artifact.path).name} "
             f"({artifact.duration_s:.2f}s, synthetic)",
             agent="audio", payload=artifact.to_dict())
    return ok({
        **artifact.to_dict(),
        "capability": {
            "dialogue": "tts", "music": "music_generation",
            "sfx": "sfx_generation", "ambience": "ambience_generation",
        }[kind],
        "available_capabilities": list(AUDIO_CAPABILITIES),
        "note": ("generated locally and deterministically; no AI audio model was "
                 "used or required"),
    })


@tool("plan_dialogue",
      "Turn a list of dialogue beats into timed lines with computed durations, "
      "ready to generate. Accepts {character, line, start_s} in any order.",
      properties={"project_id": {"type": "string"},
                  "beats": {"type": "array",
                            "items": {"type": "object"}}},
      required=["project_id"])
def tool_plan_dialogue(ctx: ToolContext, args: dict) -> dict:
    project_id = require_str(args, "project_id")
    ctx.project_and_workspace(project_id)
    beats = args.get("beats")
    if not isinstance(beats, list):
        return fail("beats must be an array of {character, line, start_s} objects")
    lines = dialogue_script(beats)
    return ok({
        "lines": lines,
        "count": len(lines),
        "total_duration_s": round(sum(line["duration_s"] for line in lines), 3),
    })


__all__ = ["list_tools", "get_tool"]


# ===========================================================================
# COMPATIBILITY
# ===========================================================================


@tool("get_client_compatibility",
      "What this server supports: transports, image delivery, and which AI "
      "clients are known to work. All clients drive the same Producer, project "
      "database, Blender pipeline and artifact registry — there is no "
      "client-specific backend.")
def tool_get_client_compatibility(ctx: ToolContext, args: dict) -> dict:
    from .compat import compatibility_report, resolve_transport

    report = compatibility_report()
    report["active_transport"] = resolve_transport(ctx.config)
    report["this_client_advertises_image_content"] = bool(
        (ctx.client_capabilities or {}).get("imageContent")
    )
    report["previews_will_include_images"] = client_accepts_images(
        ctx.client_capabilities)
    return ok(report)

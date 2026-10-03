"""Canonical characters, environments, and the audio capability foundation.

Continuity is what makes shots cut together: a character referenced by name must
resolve to the same record every time. The audio half checks that the
capabilities exist, produce real files, and stay honestly labelled as procedural.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from filmautomator.audio import (
    AUDIO_CAPABILITIES,
    SYNTHETIC,
    dialogue_script,
    generate_ambience,
    generate_music,
    generate_sfx,
    generate_tts,
)
from filmautomator.config import AppConfig
from filmautomator.mcp.server import MCPServer
from filmautomator.mcp.tools import ToolContext


def _ffmpeg() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg(), reason="needs ffprobe")


@pytest.fixture
def server(tmp_path):
    cfg = AppConfig()
    cfg.workspace = tmp_path / "ws"
    instance = MCPServer(context=ToolContext.create(cfg), config=cfg)
    yield instance
    instance.close()


def call(server, tool: str, arguments: dict | None = None) -> dict:
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments or {}},
    })
    assert "result" in response, response
    return response["result"]


def payload(server, tool: str, arguments: dict | None = None) -> dict:
    result = call(server, tool, arguments)
    if result.get("isError"):
        raise AssertionError(f"{tool} failed: {result['content'][0]['text'][:400]}")
    return json.loads(result["content"][0]["text"])


@pytest.fixture
def project(server):
    return payload(server, "create_project",
                   {"name": "CONT", "objective": "continuity"})["project_id"]


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------


def test_a_canonical_character_persists_and_is_reused(server, project):
    created = payload(server, "define_character", {
        "project_id": project, "name": "Akira", "height_m": 1.78,
        "hair": "short black", "clothing": "worn jacket",
        "colors": {"jacket": "#2b3a55"},
        "voice": "low, steady", "behavior": "watches before acting",
        "continuity": {"scar_left_cheek": True},
    })
    assert created["name"] == "Akira"
    assert created["canonical"]["height_m"] == 1.78
    assert created["continuity"]["constraints"]["scar_left_cheek"] is True

    fetched = payload(server, "get_character",
                      {"project_id": project, "name": "Akira"})
    assert fetched["canonical"]["hair"] == "short black"

    listed = payload(server, "list_characters", {"project_id": project})
    assert listed["count"] == 1
    assert listed["characters"][0]["name"] == "Akira"


def test_redefining_a_character_updates_rather_than_duplicates(server, project):
    payload(server, "define_character",
            {"project_id": project, "name": "Akira", "height_m": 1.78})
    updated = payload(server, "define_character",
                      {"project_id": project, "name": "Akira", "height_m": 1.80})
    assert updated["updated"] is True
    listed = payload(server, "list_characters", {"project_id": project})
    assert listed["count"] == 1
    assert listed["characters"][0]["canonical"]["height_m"] == 1.80


def test_an_unknown_character_is_reported_not_invented(server, project):
    result = call(server, "get_character",
                  {"project_id": project, "name": "Nobody"})
    assert result["isError"] is True
    assert "Nobody" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def test_an_environment_is_defined_once_and_referenced_by_id(server, project):
    payload(server, "define_environment", {
        "project_id": project, "environment_id": "ENV_TOKYO_RUINS_01",
        "name": "Ruined Tokyo street",
        "lighting_baseline": "blue moonlight camera-left, warm orange camera-right",
        "weather": "heavy rain", "props": ["poles", "signage"],
    })
    listed = payload(server, "list_environments", {"project_id": project})
    assert listed["count"] == 1
    env = listed["environments"][0]
    assert env["name"] == "ENV_TOKYO_RUINS_01"
    assert env["metadata"]["weather"] == "heavy rain"


def test_a_shot_can_reference_a_canonical_environment(server, project):
    payload(server, "define_environment", {
        "project_id": project, "environment_id": "ENV_TOKYO_RUINS_01",
        "weather": "heavy rain",
    })
    shot = payload(server, "create_shot", {
        "project_id": project, "shot_id": "SC01_SH001",
        "environment": "ENV_TOKYO_RUINS_01", "subject_name": "Akira",
        "subject_height_m": 1.78,
    })
    assert shot["spec"]["environment"] == "ENV_TOKYO_RUINS_01"
    subjects = shot["spec"]["subjects"]
    assert subjects[0]["height_m"] == 1.78


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def test_every_audio_capability_is_declared():
    for capability in ("dialogue_generation", "tts", "music_generation",
                       "sfx_generation", "ambience_generation"):
        assert capability in AUDIO_CAPABILITIES


def test_dialogue_generation_produces_a_real_wav(tmp_path):
    artifact = generate_tts("You're still here.", tmp_path / "line.wav")
    assert Path(artifact.path).is_file()
    assert artifact.duration_s > 0
    assert artifact.source == SYNTHETIC
    assert artifact.to_dict()["synthetic"] is True


@needs_ffmpeg
def test_generated_audio_probes_with_the_duration_it_claims(tmp_path):
    artifact = generate_music(tmp_path / "score.wav", duration_s=2.0)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", artifact.path],
        capture_output=True, text=True, check=True,
    )
    probed = float(json.loads(out.stdout)["format"]["duration"])
    assert probed == pytest.approx(artifact.duration_s, abs=0.05)


def test_the_musical_brief_actually_changes_the_music(tmp_path):
    """Otherwise the capability would be decorative rather than functional."""
    slow = generate_music(tmp_path / "a.wav", duration_s=1.0, bpm=60,
                          key="minor")
    fast = generate_music(tmp_path / "b.wav", duration_s=1.0, bpm=140,
                          key="minor")
    major = generate_music(tmp_path / "c.wav", duration_s=1.0, bpm=60,
                           key="major")
    assert Path(slow.path).read_bytes() != Path(fast.path).read_bytes()
    assert Path(slow.path).read_bytes() != Path(major.path).read_bytes()


def test_sfx_and_ambience_generate_distinct_environments(tmp_path):
    rain = generate_sfx("rain", tmp_path / "rain.wav", duration_s=0.5)
    forest = generate_ambience(tmp_path / "forest.wav", duration_s=0.5,
                               environment="forest")
    assert Path(rain.path).is_file() and Path(forest.path).is_file()
    assert Path(rain.path).read_bytes() != Path(forest.path).read_bytes()


def test_dialogue_durations_are_derived_from_the_line(tmp_path):
    short = generate_tts("Go.", tmp_path / "a.wav")
    long = generate_tts(
        "You are still here after everything that happened tonight.",
        tmp_path / "b.wav")
    assert long.duration_s > short.duration_s


def test_dialogue_script_fills_in_timings():
    lines = dialogue_script([
        {"character": "Akira", "line": "You're still here."},
        {"character": "Ken", "line": "I never left.", "start_s": 2.0},
    ])
    assert len(lines) == 2
    assert lines[0]["start_s"] == 0.0
    assert lines[1]["start_s"] == 2.0
    assert all(line["duration_s"] > 0 for line in lines)


def test_generate_audio_tool_works_and_labels_its_output(server, project):
    result = payload(server, "generate_audio", {
        "project_id": project, "kind": "dialogue", "name": "line1",
        "text": "You're still here.", "character": "Akira",
    })
    assert Path(result["path"]).is_file()
    assert result["capability"] == "tts"
    assert result["synthetic"] is True
    assert "no AI audio model was used" in result["note"]


def test_an_unknown_audio_kind_is_rejected_clearly(server, project):
    result = call(server, "generate_audio",
                  {"project_id": project, "kind": "podcast"})
    assert result["isError"] is True
    assert "dialogue, music, sfx, ambience" in result["content"][0]["text"]


def test_plan_dialogue_returns_timed_lines(server, project):
    result = payload(server, "plan_dialogue", {
        "project_id": project,
        "beats": [{"character": "Akira", "line": "You're still here."}],
    })
    assert result["count"] == 1
    assert result["total_duration_s"] > 0

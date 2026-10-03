# FilmAutomator

An autonomous, **zero-budget** AI filmmaking system. You give it a creative
objective in plain language; agents plan shots, build them in Blender, render
previews, review them, correct what looks wrong, render at final quality, and
assemble the movie with FFmpeg.

It runs entirely on your own machine. No paid APIs, no subscriptions, no
accounts, no credit card — and no third-party Python packages, so there is
nothing to install, nothing to keep up to date, and nothing that can start
billing you.

**The MCP server is the primary interface.** Claude Code (or any MCP-compatible
client) connects to it and operates the studio directly.

```
                    FILMAUTOMATOR MCP SERVER
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
        Claude Code      other MCP        any MCP
                         clients          client
            └────────────────┼────────────────┘
                             ▼
                  FILMAUTOMATOR PRODUCTION ENGINE
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
         Director         Blender           Vision
            └────────────────┼────────────────┘
                             ▼
                    render → post → QA → final movie
```

---

## Install

```bash
brew install --cask blender ffmpeg
```

Or just run `./bootstrap.sh`, which checks, installs, tests, and produces a
first film.

| Requirement | Why | Cost |
|---|---|---|
| **Blender 4.x / 5.x** | The 3D production engine | Free (GPL) |
| **FFmpeg + FFprobe** | Encoding, timeline, audio, QA probing | Free (LGPL/GPL) |
| **Python 3.11+** | Runs everything | Free |
| **A model server** *(optional)* | Planning and vision judgement | Free — LM Studio, Ollama, llama.cpp |

Nothing needs `pip install`. The system runs on the standard library plus the
numpy Blender already ships.

### The model backend is optional

With no model server the system still runs: it falls back to a deterministic
plan and takes quality decisions from pixel measurement alone. Every synthetic
response is labelled `[SYNTHETIC]` in the logs and in the report, so a
stand-in decision is never passed off as a model's.

For real planning and vision judgement, install
[LM Studio](https://lmstudio.ai), load an instruct model and a vision model, and
start its server (default `http://127.0.0.1:1234/v1`):

```bash
filmautomator doctor
```

---

## Quick start

```bash
./bootstrap.sh                  # check, install, test, make a first film
filmautomator doctor            # what's available
filmautomator dashboard         # live production view at http://127.0.0.1:8765
filmautomator mcp               # the MCP server on stdio
```

### 1. Start the dashboard

```bash
filmautomator dashboard --open
```

Leave it running. It shows real state — the live preview, the current agent,
the event feed — and updates without refreshing.

### 2. Connect the MCP server to Claude Code

```bash
claude mcp add filmautomator -- /opt/homebrew/bin/python3 -m filmautomator mcp
```

Or add it to `.mcp.json` in this directory:

```json
{
  "mcpServers": {
    "filmautomator": {
      "command": "/opt/homebrew/bin/python3",
      "args": ["-m", "filmautomator", "mcp"],
      "cwd": "/Users/sahinsamrat/Downloads/blender"
    }
  }
}
```

Then in Claude Code:

```
/mcp
```

You should see `filmautomator` listed with 50 tools. (Check without a client:
`filmautomator mcp --list-tools`.)

### 2b. Or install it as a Claude Desktop extension

For Claude Desktop, the same server ships as an `.mcpb` bundle:

```bash
sh scripts/build_mcpb.sh
# -> dist/filmautomator.mcpb
```

Then drag `dist/filmautomator.mcpb` onto Claude Desktop, or open
**Settings → Extensions → Advanced → Install Extension…** and pick it.

The bundle is a launcher, not a copy of the server: it runs the checkout you
point it at, so it cannot drift from your code. Two settings are collected at
install time — the **project folder** (default
`/Users/sahinsamrat/Downloads/blender`) and the **Python 3 executable**
(default `/opt/homebrew/bin/python3`).

It launches exactly what the CLI launches. `packaging/mcpb/server/main.py` puts
the project on `sys.path` and calls the same entry point as
`python3 -m filmautomator mcp`, so there is one server, not two.

The bundle is stdio-only and binds nothing to the network. Building it uses the
official MCPB toolchain via `npx` (free, MIT); nothing is installed permanently
and Blender and FFmpeg remain the only system requirements.

### 3. Make a film

Just talk to it:

> Create a 10-second cinematic night street scene.

Claude will call `create_project`, then `start_production`, then poll
`get_production_status` while it works. Production runs in the background, so
`start_production` returns immediately rather than blocking for minutes.

Or drive it yourself:

```
create_project(name="AWAKE_EP01", objective="a lone figure waits in the rain", duration_s=20)
start_production(project_id=..., duration_s=20)
get_production_status()
get_preview(project_id=..., shot_id="SC01_SH01")   # look at what it rendered
get_final_movie(project_id=...)
```

### 4. Watch it

Open <http://127.0.0.1:8765>. You will see the current agent, the current shot,
the progress bar, the agent activity feed, and the latest Blender preview —
appearing the moment it is rendered, with no refresh.

### 5. Collect the movie

```bash
ls projects/<project_id>/final/
# AWAKE_EP01_FINAL.mp4
```

`get_final_movie` returns the same path plus duration, resolution, fps and codec.

---

## CLI

The CLI remains a first-class interface. There is **one** production engine; the
CLI and the MCP server are two doors into it.

```bash
filmautomator doctor                     # dependency + model report
filmautomator plan "a figure in the rain" --duration 10
filmautomator make "..." --duration 10 --offline
filmautomator make "night street" --plan examples/night_street.json
filmautomator status                     # progress, shots, versions
filmautomator events                     # the production event log
filmautomator decisions                  # anything waiting on you
filmautomator answer <decision_id> "yes"
filmautomator mcp                        # MCP server (stdio)
filmautomator mcp --http-port 8766       # MCP over localhost HTTP
filmautomator dashboard --port 8765 --open
```

---

## Storage

Everything a project produces lands in one tree:

```
projects/<project_id>/
    project.db                 project memory: shots, versions, assets, QA, events
    project.json
    story/
    assets/{characters,environments,props,materials,animations}/
    scenes/
    shots/<shot_id>/v001/      SC01_SH01.blend, preview.png, render/, SC01_SH01.mp4
    previews/                  the latest preview of each shot, easy to find
    renders/
    audio/{dialogue,music,sfx}/
    editorial/                 the assembled picture cut
    qa/                        QA report
    final/                     <PROJECT>_FINAL.mp4   ← the deliverable
    logs/
```

Every generated file is registered in the database, so an AI client or the
dashboard can find it without knowing the layout. Nothing is overwritten:
each revision of a shot gets its own `vNNN/` directory.

`projects/registry.db` indexes all projects.

---

## The MCP tools

50 tools, grouped:

| Group | Tools |
|---|---|
| **Projects** | `create_project` `list_projects` `get_project` `delete_project` |
| **Production** | `start_production` `pause_production` `resume_production` `stop_production` `get_production_status` |
| **Director** | `submit_objective` `get_director_status` `get_director_decision` `approve_director_decision` |
| **Tasks** | `list_tasks` `get_task` `retry_task` `cancel_task` `approve_task` `reject_task` |
| **Scenes** | `list_scenes` `get_scene` `create_scene` `render_scene_preview` |
| **Shots** | `list_shots` `get_shot` `create_shot` `render_shot_preview` `render_shot_final` `approve_shot` `reject_shot` |
| **Blender** | `inspect_blender` `inspect_scene` `inspect_objects` `get_viewport_preview` |
| **Vision** | `inspect_preview` `get_visual_evaluation` |
| **Artifacts** | `list_artifacts` `get_artifact` `get_preview` `get_render` `get_final_movie` |
| **Agents** | `list_agents` `get_agent_status` `get_agent_activity` |
| **QA** | `run_qa` `get_qa_status` `get_qa_report` |
| **Control** | `pause` `resume` `stop` |

`get_preview` and `render_shot_preview` return the image itself, not a path — so
the AI looks at the same pixels the Vision Agent judged and you see on the
dashboard.

### Safety

Deliberately **not** exposed: shell execution, arbitrary code execution, and
unrestricted deletion. There is no tool that runs a command and none that takes
a filesystem path.

- `delete_project` requires `confirm=true` and *moves* the project to
  `projects/.trash/` — files are never erased.
- The dashboard serves artifacts **by id**, never by path, so a request cannot
  walk outside the workspace.
- Both servers bind to loopback only.
- The Blender control server does have an `evaluate` operation for the internal
  agent, but it is not reachable from MCP.

---

## Architecture

```
filmautomator/
  mcp/          MCP server: protocol, stdio + HTTP transports, 50 tools
  dashboard.py  live production dashboard (stdlib HTTP + SSE)
  runtime.py    controllable production runs (pause / resume / stop)
  producer.py   drives the pipeline end to end
  agents/       Director, Blender Agent, Vision Agent
  blender/      in-Blender control server (27 ops) + process client
  gateway/      capability-routed Model Gateway + drivers
  core/         tasks, SQLite movie memory, specs, event bus, workspace
  post/         FFmpeg encode / concat / mux / ffprobe
  qa/           objective final QA
```

**One engine, several doors.** The CLI, the MCP tools, and the dashboard all
call the same `Producer`. There is no parallel implementation to drift.

**The event bus is the nervous system.** Every meaningful action is published,
persisted with a monotonic cursor, and fanned out live. The dashboard, the MCP
activity tools and any AI client read the same stream — which is why the
dashboard cannot show anything that did not happen.

**Model-agnostic.** Agents ask the gateway for a *capability* (reasoning,
planning, vision…), never for a model. Swapping backends is a config change.
`video_generation` and `music_generation` currently report as **UNAVAILABLE** —
there is no free local driver, and the system says so rather than failing a
task at runtime.

---

## Testing

```bash
python3 -m pytest tests -q          # 172 unit + integration tests
python3 tests/smoke_blender.py      # 26 checks against real Blender
sh scripts/verify_mcp_sdk.sh        # interop check with the official MCP SDK
sh scripts/build_mcpb.sh            # build the .mcpb and verify it end to end
python3 scripts/check_dashboard_live.py   # dashboard vs a real production
```

The suite needs no Blender, no FFmpeg and no model server — everything external
is either mocked at the boundary or skipped. `smoke_blender.py` is the one that
drives real Blender.

MCP correctness is verified two ways: protocol-conformance tests that drive the
exact JSON-RPC shapes a client sends, and an interoperability check that
connects the **official MCP SDK** to this server as an independent client.

---

## Known limitations

Stated plainly so nothing here is mistaken for finished.

- **Characters are proxy geometry.** A "subject" is a blocked-in figure at the
  right height and position — enough to judge framing, scale, lighting and
  composition, but not a character. The asset directories and the character
  records exist; the pipeline that fills them does not.
- **No animation.** Cameras and lights are placed and the control server can
  keyframe, but the Director does not plan motion, so every shot is locked off.
- **No audio generation.** The audio layer, timeline placement and loudness
  normalisation are built and tested, but nothing generates dialogue, music or
  SFX. The directories are there and empty.
- **Single scene.** The plan produces one scene; multi-scene episode structure
  is not wired up.
- **One production at a time.** A production owns a Blender session on one
  machine. `start_production` refuses rather than interleaving two.
- **`video_generation` and `music_generation` have no driver.**

### Measurement honesty

The Vision Agent judges frames two ways and says which it used. The measurement
pass is objective pixel statistics and always runs; the judgement pass asks a
vision model and only runs if one exists. A report built without a model is
flagged `heuristic_only`.

Subject coverage is measured by rendering the preview **twice** — once normally,
once with the subjects hidden — and differencing the two. Getting there took
three attempts, all recorded in the code because they failed instructively:

1. *Colour distance from the frame border* reported **98% coverage** for a stage
   holding one small box — it was measuring the key light's pool on the ground.
2. *Chroma distance* was better and gave correct readings on real shots, but a
   threshold sweep from 0.05 to 0.30 showed a subject and colour-bleed across a
   lit surface are **not separable**: at 0.10 a near-empty scene read 7.3% while
   a genuine close-up read 2.0%.
3. *Background plate* is exact. Hiding a subject also removes its bounce light
   and shadow, so the threshold sits between the subject's own pixels (~0.8) and
   its influence on the room (~0.25).

Both methods are computed and reported, so disagreement is visible. On the
reference 3-shot run they agree with an independently derived geometric
expectation:

| shot | chroma | plate | geometry expects |
|---|---|---|---|
| wide | 0.4% | 0.3% | 0.4% |
| medium | 9.8% | 8.0% | 9.4% |
| close-up | 22.7% | 16.4% | 27.8% |

---

## Design commitments

**Zero cost is a hard constraint.** Nothing in the default path can bill you.
External providers exist only behind adapters that are off unless you enable
them.

**The system never lies about what happened.** Synthetic model output is
labelled. The dashboard reads real state — there is no simulated activity, no
mock progress, no placeholder. If a check cannot run, it says so instead of
passing.

**Measurement has veto power.** A vision model saying a frame looks great does
not override pixels that are measurably black.

**Corrections are concrete.** Vision findings carry spec paths
(`camera.shot_size`, `lighting.key_energy`) the Blender Agent applies directly,
so revision is deterministic rather than another model call.

**Iteration is bounded.** Every loop has a cap, and hitting it is reported.
A correction that would not change anything stops the loop instead of spinning.

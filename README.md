# FilmAutomator

An autonomous, **zero-budget** AI filmmaking system. The owner supplies creative
direction; agents plan, build, inspect, revise, render and assemble the movie in
Blender, then encode it with FFmpeg.

No paid APIs. No subscriptions. No accounts. No credit card. Everything runs
locally on open-source software.

---

## Status: the first vertical slice runs

This is milestone 1 from the specification — the smallest thing that runs the
whole loop end to end, verified on Blender 5.2.2 LTS and FFmpeg 9.0.2:

```
objective
  -> Director plans shots                    (model, or a supplied plan, or a fallback)
  -> Blender Agent builds the scene          (real Blender, via control server)
  -> preview render                          (EEVEE)
  -> Vision Agent inspects the pixels        (measurement + optional vision model)
  -> Director corrects and rebuilds          (up to N rounds, then it stops and says so)
  -> final render                            (EEVEE or Cycles)
  -> FFmpeg encodes each shot and the timeline
  -> Final QA
  -> production report
```

A real 3-shot run (`examples/night_street.json`, 8.5s, 960x540, 24fps) completes
in about 3 minutes with every shot approved on the first review round and final
QA passing at 1.00.

### What works — and is verified by running it

- Persistent Blender control server over loopback TCP: 27 scene operations, full
  scene-state introspection, pixel-level image analysis, structured error
  reporting, and automatic restart after a crash.
- Model Gateway with capability-based routing, an LM Studio / OpenAI-compatible
  driver, and a deterministic offline stand-in that never pretends to be a model.
- The autonomous observe → evaluate → correct loop, with a hard iteration cap and
  a no-op guard so it cannot spin.
- SQLite project memory scoped per project: scenes, shots, shot versions, assets,
  characters, renders, QA results, owner decisions, event log.
- Versioned shots — a revision never overwrites the previous one.
- FFmpeg encoding, timeline assembly, audio muxing with per-track offsets and
  loudness normalisation, and ffprobe-based technical QA.
- Windows-safe defaults: EEVEE previews, Cycles final, both resolved with a
  fallback chain against whatever the installed Blender actually supports.

### What is deliberately not built yet

Stated plainly so nothing here is mistaken for finished:

- **Characters are proxy geometry.** There is no character asset pipeline yet,
  so a "subject" is a blocked-in figure at the right height and position. That is
  enough to judge framing, scale, lighting and composition — which is what the
  revision loop needs — but it is not a character.
- **No animation.** Cameras and lights are placed and the control server can
  keyframe, but the Director does not yet plan motion, so every shot is locked off.
- **No audio generation.** The audio layer, timeline placement and loudness
  normalisation are built, but nothing generates dialogue, music or SFX. Those
  need a speech/music model you supply.
- **Single scene.** The plan produces one scene; multi-scene episode structure is
  not wired up.
- **The owner interface is a CLI**, not the control panel of spec section 18.
  It covers the same operations (start, stop, retry, approve, reject, answer
  decisions, talk to the Director) but without a GUI.
- **`video_generation` and `music_generation` have no driver.** The gateway
  reports them as unimplemented rather than failing a task at runtime.

### Measurement honesty

The Vision Agent judges frames two ways, and says which one it used. The
measurement pass is objective pixel statistics and always runs. The judgement
pass asks a vision model, and only runs if one is available. A report built
without a model is flagged `heuristic_only`, and the gateway logs
`[SYNTHETIC]` for every stand-in response, so a mechanical decision is never
presented as a model's opinion.

Subject coverage — "how much of the frame is the subject" — is measured by
rendering the preview **twice**: once normally and once with the subjects
hidden, then differencing the two. That mask is exact and immune to lighting.

Getting there took three attempts, all of which failed in instructive ways, so
the reasoning is recorded in the code:

1. *Colour distance from the frame border.* On a lit stage the key light throws
   a pool across the ground that is brighter than a dark backdrop, so this
   reported **98% coverage** for a stage holding one small box.
2. *Chroma distance from the border.* Better, and it gave correct readings on
   the real shots — but a large lit surface carrying colour bleed from the
   subject varies in hue just as a real subject does. Sweeping the threshold
   from 0.05 to 0.30 showed the two are **not separable**: at 0.10 a near-empty
   scene still read 7.3% while a genuine close-up read 2.0%.
3. *Background plate.* Exact. Hiding a subject also removes its bounce light
   and shadow, so the threshold is set between "the subject's own pixels"
   (~0.8 summed linear RGB) and "its influence on the room" (~0.25).

The two methods are both computed and both reported, so disagreement between
them is visible rather than hidden. On the reference 3-shot run they agree with
an expectation derived independently from the framing geometry:

| shot | chroma | plate | geometry expects |
|---|---|---|---|
| wide | 0.4% | 0.3% | 0.4% |
| medium | 9.8% | 8.0% | 9.4% |
| close-up | 22.7% | 16.4% | 27.8% |

---

## Requirements

| Tool | Why | Cost |
|---|---|---|
| **Blender 4.x or 5.x** | The 3D production engine | Free (GPL) |
| **FFmpeg + FFprobe** | Shot encoding, timeline, audio, QA probing | Free (LGPL/GPL) |
| **Python 3.11+** | Runs the orchestrator | Free |
| **A model server** *(optional)* | Planning and vision judgement | Free — LM Studio, Ollama, llama.cpp |

```bash
brew install --cask blender ffmpeg
```

Or just run `./bootstrap.sh`, which checks, installs, tests and produces a first
movie.

Nothing needs `pip install` — the system runs on the Python standard library
plus the numpy that Blender already ships. (`pytest` is only needed for the test
suite.)

### The model backend is optional

Without a model server the system still runs: it falls back to a deterministic
plan and takes quality decisions from pixel measurement alone. Every synthetic
response is labelled `[SYNTHETIC]` in the logs and flagged in the report, so a
stand-in decision is never mistaken for a model's.

To get real planning and vision judgement, install
[LM Studio](https://lmstudio.ai), load an instruct model and a vision model, and
start its server (default `http://127.0.0.1:1234/v1`). Then:

```bash
filmautomator doctor
```

---

## Usage

```bash
# Check everything is wired up
filmautomator doctor

# See the shot plan without rendering anything
filmautomator plan "a lone figure on a rain-soaked street at night" --duration 10

# Produce the movie
filmautomator make "a lone figure on a rain-soaked street at night" --duration 10

# Produce it without touching a model at all
filmautomator make "..." --duration 10 --offline

# Hand in your own shot list
filmautomator make "night street" --plan examples/night_street.json

# Inspect progress
filmautomator status
filmautomator events
filmautomator decisions
```

Working offline needs no model at all: the Director falls back to a
deterministic single-shot plan and the Vision Agent judges from pixel
measurement alone. Everything it produces is labelled as such.

Output lands in `runs/<project_id>/`:

```
project.db                     all project memory
scenes/  shots/  assets/  audio/
shots/SC01_SH01/
    v001/SC01_SH01.blend       the scene, per revision
    v001/preview.png           what the Vision Agent judged
    v001/render/frame_0001.png final frames
    v001/SC01_SH01.mp4         the encoded shot
final/final.mp4                the movie
reports/production_report.txt  what happened and why
```

Each revision gets its own `vNNN/` directory. Nothing is overwritten.

---

## Architecture

```
OWNER
  |
  v
Producer  ──────────────►  Project DB (SQLite)  ◄── movie memory
  |
  v
Director (AI CEO)  ──plans──►  ShotSpec  ──►  Blender Agent
  |                                                |
  |                                          Blender control server
  |                                                |
  |                                             Blender
  |                                                |
  |   ┌─────────── Vision Agent ◄── preview render ┘
  |   |                |
  |   └── findings ────┘
  |        (correct and repeat, up to N rounds)
  v
FFmpeg  ──►  Final QA  ──►  production report
```

Every model call goes through the **Model Gateway**, which routes by
*capability* — reasoning, planning, vision, and so on — never by model name.
Swapping backends is a config change, not a code change.

---

## Configuration

Copy `filmautomator.toml.example` to `filmautomator.toml`. Everything is also
overridable by environment variable; see the comments in that file.

The most useful knobs:

```toml
[gateway]
openai_compat_base_url = "http://127.0.0.1:1234/v1"

[render]
final_engine = "CYCLES"
cycles_device = "METAL"     # big speedup on Apple Silicon
final_samples = 64

[limits]
max_shot_revision_rounds = 4
```

---

## Design commitments

**Zero cost is a hard constraint, not a preference.** Nothing in the default
path can bill you. External providers exist only behind adapters that are
disabled unless you turn them on.

**The system never lies about what happened.** Synthetic model output is
labelled. Unmeasured claims are not made. If a QA check cannot run, it reports
that it could not run rather than passing.

**Measurement has veto power.** A vision model saying a frame looks great does
not override pixels that are measurably black.

**Corrections are concrete.** Vision findings carry spec paths
(`camera.shot_size`, `lighting.key_energy`) that the Blender Agent applies
directly, so revision is deterministic rather than another model call.

**Iteration is bounded.** Every loop has a cap, and hitting it is reported
rather than hidden.

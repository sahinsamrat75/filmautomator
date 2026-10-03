# FilmAutomator — MCP extension

Autonomous AI filmmaking on Blender and FFmpeg. Give it a creative objective in
plain language and it plans shots, builds them in Blender, renders previews,
reviews them, corrects what looks wrong, renders at final quality, and assembles
the movie with FFmpeg.

This extension is a launcher. It does not contain a copy of the server: it runs
the Filmautomator checkout you point it at, so you always get the code you have,
not a snapshot.

## Before you start

FilmAutomator needs two programs that are **not** bundled, because they are
large and you may already have them:

```bash
brew install --cask blender ffmpeg
```

Both are free and open source. Nothing here needs an account, a subscription, or
a credit card, and the server binds nothing to the network — it speaks MCP over
stdio only.

A model server is **optional**. Point it at LM Studio, Ollama or llama.cpp for
real planning and vision judgement; without one it falls back to a deterministic
plan and judges frames from pixel measurement, labelling every such decision as
synthetic.

## Settings

| Setting | Meaning |
|---|---|
| **FilmAutomator project folder** | The checkout containing the `filmautomator` package and the `projects` output directory. Defaults to `/Users/sahinsamrat/Downloads/blender`. |
| **Python 3 executable** | The `python3` that runs the server, 3.11 or newer. Defaults to `/opt/homebrew/bin/python3`. |

## What to ask for

Once connected, just describe the film:

> Create a 10-second cinematic night street scene.

Or drive the tools yourself, in order:

| Step | Tool |
|---|---|
| 1 | `create_project` — name it, describe it |
| 2 | `start_production` — begins in the background and returns at once |
| 3 | `get_production_status` — poll until `COMPLETED` |
| 4 | `get_preview` — see the frame the Vision Agent judged |
| 5 | `get_final_movie` — the deliverable, its path and its technical details |

## Where the film goes

```
<project folder>/projects/<project_id>/final/<PROJECT>_FINAL.mp4
```

Every preview, render, encoded shot and report is registered in the project
database, so `list_artifacts` finds them without you knowing the layout.

## Watching it happen

The extension exposes the tools; it does not add a window. For a live view of the
current agent, the progress bar and the preview updating as it renders, run the
dashboard alongside it:

```bash
/opt/homebrew/bin/python3 -m filmautomator dashboard --open
```

`dashboard` is local to this machine (`http://127.0.0.1:8765`) and is not part of
the MCP surface.

## Things worth knowing

- **Rendering takes real time.** A short scene is minutes, not seconds.
  `start_production` returns immediately for exactly that reason.
- **One production at a time.** A production owns a Blender session on one
  machine; a second `start_production` is refused rather than interleaved.
- **Revision is bounded.** Each shot gets a fixed number of build/preview/review
  rounds. Hitting the limit is reported, not hidden.
- **Nothing is overwritten.** Every revision of a shot gets its own `vNNN/`
  directory, and `approve_shot` marks a version without deleting earlier ones.
- **Deletion is reversible.** `delete_project` requires explicit confirmation and
  moves the project to `.trash/` rather than erasing it.

## If it does not start

Run the same checks by hand:

```bash
/opt/homebrew/bin/python3 -m filmautomator doctor
```

That reports Blender, FFmpeg, the model gateway and every capability, and tells
you exactly what is missing. The extension performs the same checks at startup
and writes any warnings to stderr, where the host's logs will show them.

## Licence

MIT.

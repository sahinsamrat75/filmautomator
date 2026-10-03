#!/usr/bin/env python3
"""Filmautomator launcher for the Claude Desktop bundle.

This starts the existing Filmautomator MCP server. It does not reimplement
anything: it puts the project on ``sys.path`` and calls the very same CLI entry
point that ``python3 -m filmautomator mcp`` calls, so the bundled server and the
command-line server are the same code with no divergence to drift apart.

Contract with the MCP host: **stdout carries protocol messages and nothing
else.** A stray ``print`` here corrupts the JSON-RPC stream and the client
silently drops the connection. Every diagnostic in this file goes to stderr.

Server configuration comes from the environment the host sets from the
manifest:

    FA_PROJECT_DIR   the Filmautomator checkout to run
    PYTHONPATH       same directory, so ``import filmautomator`` resolves
    FA_WORKSPACE     where projects are written

The transport is stdio only. Nothing is bound to the network.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]  # noqa: F821
    """Report a fatal startup problem on stderr and exit non-zero.

    Never writes to stdout: the host is already listening there for JSON-RPC.
    """
    sys.stderr.write(f"filmautomator: {message}\n")
    sys.stderr.flush()
    raise SystemExit(1)


def _resolve_project_dir() -> Path:
    """Find the Filmautomator checkout this bundle should run."""
    for source in ("FA_PROJECT_DIR", "PYTHONPATH"):
        raw = os.environ.get(source, "").strip()
        if not raw:
            continue
        # PYTHONPATH may be a path list; take the first entry that looks right.
        for candidate in raw.split(os.pathsep):
            candidate = candidate.strip()
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if (path / "filmautomator" / "__init__.py").is_file():
                return path

    _fail(
        "could not find the Filmautomator project.\n"
        "  Set the 'FilmAutomator project folder' option to the checkout that\n"
        "  contains the 'filmautomator' package, for example\n"
        "  /Users/sahinsamrat/Downloads/blender"
    )


def _check_environment(project_dir: Path) -> None:
    """Warn about missing optional tools without refusing to start.

    The MCP server is useful even when Blender is absent — project management,
    artifact browsing and status all work. Production tools report the missing
    dependency themselves with instructions, so this only warns.
    """
    notes: list[str] = []
    if not Path("/Applications/Blender.app").exists():
        notes.append("Blender not found at /Applications/Blender.app "
                     "(brew install --cask blender)")
    if not (Path("/opt/homebrew/bin/ffmpeg").exists()
            or any((Path(p) / "ffmpeg").exists()
                   for p in os.environ.get("PATH", "").split(os.pathsep) if p)):
        notes.append("FFmpeg not found on PATH (brew install ffmpeg)")

    if notes:
        sys.stderr.write(
            "filmautomator: starting with reduced capability — "
            + "; ".join(notes)
            + ". The server is up; production tools will explain what is "
              "missing.\n"
        )
        sys.stderr.flush()


def main() -> int:
    project_dir = _resolve_project_dir()

    # Make the project importable regardless of how the host set things up.
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    # Default the workspace alongside the project, as the CLI does.
    os.environ.setdefault("FA_WORKSPACE", str(project_dir / "projects"))

    try:
        from filmautomator.cli import main as filmautomator_main
    except Exception as exc:  # noqa: BLE001 - report, never traceback onto stdout
        _fail(
            f"could not import filmautomator from {project_dir}: "
            f"{type(exc).__name__}: {exc}"
        )

    _check_environment(project_dir)

    # Identical to `python3 -m filmautomator mcp`: same argument parsing, same
    # server construction, same stdio transport.
    return filmautomator_main(["mcp"])


if __name__ == "__main__":
    sys.exit(main())

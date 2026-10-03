#!/usr/bin/env bash
# One-shot setup and smoke test for FilmAutomator.
#
# Safe to re-run: every step checks before it acts.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-/opt/homebrew/bin/python3}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*" >&2; }

step "Checking required tools"
MISSING=()
command -v brew >/dev/null 2>&1 || fail "Homebrew not found — install from https://brew.sh"

if [[ ! -d /Applications/Blender.app ]]; then
  MISSING+=("--cask blender")
else
  echo "  Blender: present"
fi

if ! command -v ffmpeg >/dev/null 2>&1 && [[ ! -x /opt/homebrew/bin/ffmpeg ]]; then
  MISSING+=("ffmpeg")
else
  echo "  FFmpeg:  present"
fi

if (( ${#MISSING[@]} )); then
  step "Installing: ${MISSING[*]}"
  echo "  These are free and open-source; no account or payment is involved."
  HOMEBREW_NO_AUTO_UPDATE=1 NONINTERACTIVE=1 brew install "${MISSING[@]}"
fi

step "Dependency report"
"$PY" -m filmautomator doctor || true

step "Running the offline test suite"
if "$PY" -c "import pytest" 2>/dev/null; then
  "$PY" -m pytest tests -q
else
  echo "  pytest not installed; skipping. Install with: $PY -m pip install pytest"
fi

step "Producing a first movie (offline, no model server needed)"
"$PY" -m filmautomator make \
  "A lone figure stands in the rain on a dark street, lit from behind" \
  --duration 8 --offline

printf '\n\033[32mDone.\033[0m Output is under %s/runs/\n' "$ROOT"

"""Python-side client for the in-Blender control server.

Owns the Blender process lifecycle: launch, port discovery, request/response,
crash detection and restart. Agents talk to this object, never to Blender
directly.

Spec section 19 requires automatic recovery from a Blender crash, so
:meth:`BlenderSession.call` detects a dead process and :meth:`restart` brings a
fresh one up without the orchestrator needing to know anything went wrong.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..config import BlenderConfig, find_blender

log = logging.getLogger(__name__)

CONTROL_SERVER = Path(__file__).parent / "scripts" / "control_server.py"


class BlenderError(RuntimeError):
    """A control operation returned ``ok: false``."""

    def __init__(self, op: str, message: str, traceback_text: str = "") -> None:
        super().__init__(f"{op} failed: {message}")
        self.op = op
        self.message = message
        self.traceback_text = traceback_text


class BlenderUnavailable(RuntimeError):
    """Blender is not installed, or exited before it could serve requests."""


class BlenderSession:
    """A persistent, observable Blender instance under agent control."""

    def __init__(self, config: BlenderConfig | None = None) -> None:
        self.config = config or BlenderConfig()
        self.executable: Path | None = self.config.executable or find_blender()
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._stream = None
        self._token = secrets.token_hex(16)
        self._port: int = 0
        self._log: deque[str] = deque(maxlen=4000)
        #: Output from the session before the current one, kept for post-mortem
        #: after a crash restart clears the live buffer.
        self._previous_session_log: str = ""
        self._log_lock = threading.Lock()
        self._drain_thread: threading.Thread | None = None
        self._request_seq = 0
        self._restarts = 0
        #: Set when the session died unexpectedly, so callers can report why.
        self.last_failure: str = ""

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._sock is not None
        )

    def start(self) -> None:
        """Launch Blender and wait until its control socket answers."""
        if self.running:
            return
        if self.executable is None:
            raise BlenderUnavailable(
                "Blender executable not found. Install it with "
                "`brew install --cask blender`, or set FA_BLENDER_PATH to the "
                "Blender binary."
            )
        if not self.executable.is_file():
            raise BlenderUnavailable(f"Blender path does not exist: {self.executable}")

        cmd = [
            str(self.executable),
            "--background",
            "--factory-startup",
            "--python",
            str(CONTROL_SERVER),
            "--",
            "--port", str(self.config.control_port),
            "--token", self._token,
        ]
        log.info("launching Blender: %s", " ".join(cmd))

        # The port announcement is read out of the log buffer, so the buffer
        # must not still contain the previous session's announcement — that
        # would hand us a dead port and fail the reconnect. Keep the old lines
        # for diagnosis, but start the scan from a clean buffer.
        with self._log_lock:
            self._previous_session_log = "\n".join(self._log)
            self._log.clear()

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._start_log_drain()

        try:
            self._port = self._await_port()
        except Exception:
            self.stop()
            raise

        self._connect()
        log.info("Blender control session ready on port %d", self._port)

    def _start_log_drain(self) -> None:
        """Continuously consume Blender's output so its pipe never fills.

        A full pipe would deadlock Blender mid-render, and the buffered lines
        are the raw material for the error-observation channel.
        """
        proc = self._proc
        assert proc is not None and proc.stdout is not None

        def drain() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                with self._log_lock:
                    self._log.append(line.rstrip("\n"))

        self._drain_thread = threading.Thread(
            target=drain, name="blender-log-drain", daemon=True
        )
        self._drain_thread.start()

    def _await_port(self) -> int:
        """Read Blender's stdout until the control server announces its port."""
        proc = self._proc
        assert proc is not None
        deadline = time.monotonic() + self.config.startup_timeout_s

        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise BlenderUnavailable(
                    "Blender exited during startup (code %s). Output:\n%s"
                    % (proc.returncode, self.tail_log(40))
                )
            with self._log_lock:
                lines = list(self._log)
            for line in lines:
                if line.startswith("FA_CONTROL_PORT"):
                    return int(line.split()[1])
            time.sleep(0.1)

        raise BlenderUnavailable(
            "Blender did not open its control socket within %.0fs. Output:\n%s"
            % (self.config.startup_timeout_s, self.tail_log(40))
        )

    def _connect(self) -> None:
        self._sock = socket.create_connection(("127.0.0.1", self._port), timeout=30.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._stream = self._sock.makefile("rwb")

    def stop(self) -> None:
        """Ask Blender to exit, then make sure it actually did."""
        if self._stream is not None:
            try:
                self._send_raw("shutdown", {})
            except Exception:  # noqa: BLE001 - we are tearing down anyway
                pass
        for closer in (self._stream, self._sock):
            if closer is not None:
                try:
                    closer.close()
                except OSError:
                    pass
        self._stream = None
        self._sock = None

        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("Blender ignored SIGTERM; killing")
                    self._proc.kill()
                    self._proc.wait(timeout=10)
        self._proc = None

    def restart(self) -> None:
        """Bring up a clean Blender after a crash (spec section 19)."""
        self._restarts += 1
        log.warning("restarting Blender session (restart #%d)", self._restarts)
        self.stop()
        self._token = secrets.token_hex(16)
        self.start()

    def __enter__(self) -> "BlenderSession":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- diagnostics -------------------------------------------------------

    def tail_log(self, lines: int = 50) -> str:
        with self._log_lock:
            return "\n".join(list(self._log)[-lines:])

    def blender_log(self) -> str:
        with self._log_lock:
            return "\n".join(self._log)

    @property
    def restart_count(self) -> int:
        return self._restarts

    # -- request plumbing --------------------------------------------------

    def _send_raw(self, op: str, args: dict[str, Any]) -> dict:
        """Write one request and read one response. No retry, no recovery."""
        assert self._stream is not None, "session not started"
        self._request_seq += 1
        request_id = self._request_seq
        payload = {"id": request_id, "token": self._token, "op": op, "args": args}
        self._stream.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._stream.flush()

        line = self._stream.readline()
        if not line:
            raise BlenderUnavailable("Blender closed the control connection")
        response = json.loads(line.decode("utf-8"))
        if response.get("id") != request_id:
            raise BlenderUnavailable(
                f"desynchronised control channel: sent id {request_id}, "
                f"got {response.get('id')}"
            )
        return response

    def call(self, op: str, **args: Any) -> Any:
        """Run one operation, restarting Blender once if it has died.

        Returns the operation's ``result`` payload. Raises :class:`BlenderError`
        when the operation itself failed (which is a scene problem, not a
        process problem) and :class:`BlenderUnavailable` when Blender could not
        be reached even after a restart.
        """
        if not self.running:
            self.last_failure = "session was not running"
            self.restart()

        assert self._sock is not None and self._stream is not None
        self._sock.settimeout(self.config.op_timeout_s)

        try:
            response = self._send_raw(op, args)
        except (socket.timeout, BlenderUnavailable, OSError, json.JSONDecodeError) as exc:
            # One automatic recovery attempt, per spec section 19.
            self.last_failure = f"{type(exc).__name__}: {exc}"
            log.error("control call %r failed (%s); restarting Blender", op, exc)
            try:
                self.restart()
                response = self._send_raw(op, args)
            except Exception as retry_exc:  # noqa: BLE001
                raise BlenderUnavailable(
                    f"Blender unavailable after restart: {retry_exc}. "
                    f"Original failure: {exc}. Blender output:\n{self.tail_log(30)}"
                ) from retry_exc

        if not response.get("ok"):
            raise BlenderError(
                op,
                str(response.get("error", "unknown error")),
                str(response.get("traceback", "")),
            )
        return response.get("result")

    # -- typed convenience wrappers ---------------------------------------

    def ping(self) -> dict:
        return self.call("ping")

    def scene_state(self) -> dict:
        """Observation channel 5A."""
        return self.call("get_scene_state")

    def scene_errors(self) -> dict:
        """Observation channel 5E."""
        return self.call("get_errors")

    def reset_scene(self, empty: bool = True) -> dict:
        return self.call("reset_scene", empty=empty)

    def create_primitive(self, kind: str, **kwargs: Any) -> dict:
        return self.call("create_primitive", kind=kind, **kwargs)

    def set_transform(self, name: str, **kwargs: Any) -> dict:
        return self.call("set_transform", name=name, **kwargs)

    def delete_objects(self, names: list[str]) -> dict:
        return self.call("delete_object", names=names)

    def create_material(self, name: str, **kwargs: Any) -> dict:
        return self.call("create_material", name=name, **kwargs)

    def assign_material(self, material: str, objects: list[str], **kwargs: Any) -> dict:
        return self.call("assign_material", material=material, objects=objects, **kwargs)

    def create_light(self, **kwargs: Any) -> dict:
        return self.call("create_light", **kwargs)

    def create_camera(self, **kwargs: Any) -> dict:
        return self.call("create_camera", **kwargs)

    def set_active_camera(self, name: str) -> dict:
        return self.call("set_active_camera", name=name)

    def set_scene_timing(self, **kwargs: Any) -> dict:
        return self.call("set_scene_timing", **kwargs)

    def set_render_settings(self, **kwargs: Any) -> dict:
        return self.call("set_render_settings", **kwargs)

    def render_still(self, filepath: str | Path, **kwargs: Any) -> dict:
        """Observation channel 5C."""
        return self.call("render_still", filepath=str(filepath), **kwargs)

    def render_animation(self, directory: str | Path, **kwargs: Any) -> dict:
        """Observation channel 5D."""
        return self.call("render_animation", directory=str(directory), **kwargs)

    def viewport_screenshot(self, filepath: str | Path, **kwargs: Any) -> dict:
        """Observation channel 5B."""
        return self.call("viewport_screenshot", filepath=str(filepath), **kwargs)

    def save_blend(self, filepath: str | Path, **kwargs: Any) -> dict:
        return self.call("save_blend", filepath=str(filepath), **kwargs)

    def load_blend(self, filepath: str | Path) -> dict:
        return self.call("load_blend", filepath=str(filepath))

    def evaluate(self, code: str) -> Any:
        """Run a Python snippet inside Blender. Escape hatch for the agent."""
        return self.call("evaluate", code=code)

    # -- environment checks ------------------------------------------------

    @staticmethod
    def diagnose() -> str:
        """Human-readable readiness report, used by the owner CLI."""
        exe = find_blender()
        if exe is None:
            return (
                "Blender: NOT FOUND\n"
                "  Install with:  brew install --cask blender\n"
                "  Or set:        FA_BLENDER_PATH=/path/to/Blender"
            )
        version = "unknown"
        try:
            out = subprocess.run(
                [str(exe), "--version"],
                capture_output=True, text=True, timeout=30,
            )
            first = (out.stdout or "").strip().splitlines()
            if first:
                version = first[0]
        except Exception as exc:  # noqa: BLE001
            version = f"could not query ({exc})"
        return f"Blender: {exe}\n  {version}"


def diagnose_all() -> str:
    """Readiness report covering every external tool the slice needs."""
    from ..config import find_ffmpeg, find_ffprobe

    lines = [BlenderSession.diagnose()]

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        lines.append(
            "FFmpeg: NOT FOUND\n"
            "  Install with:  brew install ffmpeg"
        )
    else:
        version = "unknown"
        try:
            out = subprocess.run(
                [str(ffmpeg), "-version"], capture_output=True, text=True, timeout=30
            )
            version = (out.stdout or "").splitlines()[0] if out.stdout else "unknown"
        except Exception as exc:  # noqa: BLE001
            version = f"could not query ({exc})"
        lines.append(f"FFmpeg: {ffmpeg}\n  {version}")

    ffprobe = find_ffprobe()
    lines.append(
        f"FFprobe: {ffprobe}" if ffprobe else
        "FFprobe: NOT FOUND (ships with ffmpeg)"
    )

    lines.append(f"Python: {sys.version.split()[0]} ({sys.executable})")
    return "\n".join(lines)

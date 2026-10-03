"""Configuration: discovery of local tools, gateway endpoints, and limits.

Everything here is overridable by environment variable so the owner never has
to edit source to point the system at a different Blender or model server.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Tool discovery
# --------------------------------------------------------------------------

#: Places to look for the Blender executable, in priority order. macOS app
#: bundles first (that is what `brew install --cask blender` produces), then
#: anything on PATH.
BLENDER_CANDIDATES: tuple[str, ...] = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "~/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/local/bin/blender",
    "/opt/homebrew/bin/blender",
    "/usr/bin/blender",
)

FFMPEG_CANDIDATES: tuple[str, ...] = (
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)

FFPROBE_CANDIDATES: tuple[str, ...] = (
    "/opt/homebrew/bin/ffprobe",
    "/usr/local/bin/ffprobe",
    "/usr/bin/ffprobe",
)


def _first_existing(candidates: tuple[str, ...], which_name: str) -> Path | None:
    """Return the first candidate that exists, else whatever is on PATH."""
    for raw in candidates:
        p = Path(raw).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
    found = shutil.which(which_name)
    return Path(found) if found else None


def find_blender() -> Path | None:
    """Locate the Blender executable, or None if Blender is not installed."""
    override = os.environ.get("FA_BLENDER_PATH")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    return _first_existing(BLENDER_CANDIDATES, "blender")


def find_ffmpeg() -> Path | None:
    override = os.environ.get("FA_FFMPEG_PATH")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    return _first_existing(FFMPEG_CANDIDATES, "ffmpeg")


def find_ffprobe() -> Path | None:
    override = os.environ.get("FA_FFPROBE_PATH")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    return _first_existing(FFPROBE_CANDIDATES, "ffprobe")


# --------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GatewayConfig:
    """Where the Model Gateway sends work.

    Only free/local endpoints are configured by default. External paid
    providers exist behind adapters that must be explicitly enabled.
    """

    #: OpenAI-compatible base URL. LM Studio serves this by default.
    openai_compat_base_url: str = "http://127.0.0.1:1234/v1"
    openai_compat_api_key: str = "not-needed"  # LM Studio ignores the value
    openai_compat_timeout_s: float = 300.0

    #: Model name per capability. Empty string means "let the server pick"
    #: (LM Studio routes to whatever is loaded when the model field is a
    #: placeholder, and we resolve the real id at runtime).
    models: dict[str, str] = field(default_factory=dict)

    #: When no capable local driver answers, fall back to the deterministic
    #: mock driver rather than failing the task outright.
    allow_mock_fallback: bool = True

    #: Paid/external providers are off unless the owner turns them on.
    enable_external_providers: bool = False


@dataclass(slots=True)
class BlenderConfig:
    executable: Path | None = None
    #: Port for the in-Blender control server. 0 means "pick a free port".
    control_port: int = 0
    #: Seconds to wait for Blender to boot and open its control socket.
    startup_timeout_s: float = 120.0
    #: Seconds to wait for a single operation to return.
    op_timeout_s: float = 900.0


@dataclass(slots=True)
class RenderConfig:
    #: "BLENDER_EEVEE" | "BLENDER_EEVEE_NEXT" | "BLENDER_WORKBENCH" | "CYCLES"
    #:
    #: Previews default to EEVEE rather than the faster Workbench on purpose:
    #: the preview exists so the Vision Agent can judge exposure and lighting,
    #: and Workbench renders with its own flat studio lighting that bears no
    #: relation to the scene's actual lights. A fast preview that cannot be
    #: trusted about lighting makes the correction loop meaningless.
    #: The engine name is resolved at runtime with a fallback chain, so a
    #: build without EEVEE degrades instead of failing.
    preview_engine: str = "BLENDER_EEVEE"
    final_engine: str = "CYCLES"
    #: Cycles device: "CPU" | "METAL" | "CUDA" | "OPTIX"
    cycles_device: str = "CPU"
    preview_samples: int = 16
    final_samples: int = 64
    #: Preview resolution is deliberately small — it only has to be good
    #: enough for the Vision Agent to judge composition and lighting.
    preview_width: int = 480
    preview_height: int = 270
    final_width: int = 1280
    final_height: int = 720
    fps: int = 24


@dataclass(slots=True)
class LimitsConfig:
    """Iteration caps. Spec section 6 requires these to prevent infinite loops."""

    max_shot_revision_rounds: int = 4
    max_task_retries: int = 3
    max_agent_calls_per_task: int = 40


@dataclass(slots=True)
class StorageConfig:
    """The working-storage budget for production.

    This is a ceiling on *working* storage — frames, previews, temporaries and
    other disposable production artifacts. It is not a quota on deliverables:
    a finalized movie stays on disk no matter how large it is, and no cleanup
    path will ever remove one to make room.

    Defaults suit a laptop with roughly 80 GB free. The 35 GB ceiling leaves the
    owner's own files room to exist alongside a production in progress.
    """

    #: Total working storage Filmautomator may occupy before refusing to start
    #: another large render.
    max_working_gb: float = 35.0
    #: Cross this and the owner is told the disk is filling.
    warning_gb: float = 25.0
    #: Cross this and production cleans up disposable artifacts on its own.
    aggressive_cleanup_gb: float = 30.0
    #: Refuse to begin a large render once above the limit unless a safe
    #: cleanup can first free enough room.
    hard_limit_gb: float = 35.0
    #: Clean up automatically once the disk enters AGGRESSIVE_CLEANUP.
    cleanup_enabled: bool = True
    #: How much space a large render is assumed to need when checking headroom.
    #: Deliberately conservative: refusing slightly early is harmless, whereas
    #: starting a render that cannot finish is not.
    estimated_render_gb: float = 2.0

    def thresholds(self) -> "StorageThresholds":
        """Build the governor's threshold object from this configuration."""
        from .core.storage import StorageThresholds

        return StorageThresholds(
            warning_bytes=int(self.warning_gb * 1024 ** 3),
            aggressive_bytes=int(self.aggressive_cleanup_gb * 1024 ** 3),
            hard_limit_bytes=int(self.hard_limit_gb * 1024 ** 3),
        )


@dataclass(slots=True)
class AppConfig:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    blender: BlenderConfig = field(default_factory=BlenderConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    #: Root directory where projects are written. Each project gets its own
    #: directory tree under here (spec section 9).
    workspace: Path = field(default_factory=lambda: Path.cwd() / "projects")
    #: Port for the local production dashboard. 0 picks a free port.
    dashboard_port: int = 8765
    #: Port for the MCP server's optional network transport. 0 disables it.
    mcp_http_port: int = 0


def _resolve_workspace(value: str | Path) -> Path:
    """Make a workspace path absolute and normalised.

    Everything downstream derives from this — the project root, the registry,
    every artifact path. Leaving it relative means the same configuration
    produces different absolute paths depending on each process's working
    directory, so the MCP server (launched by a GUI host with an arbitrary cwd)
    and the CLI can disagree about where a project lives. Resolving once, here,
    is what keeps them identical.
    """
    return Path(value).expanduser().resolve()


def _apply_toml(cfg: AppConfig, data: dict) -> None:
    """Overlay a parsed TOML document onto an AppConfig in place."""
    for section_name in ("gateway", "blender", "render", "limits", "storage"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        target = getattr(cfg, section_name)
        for key, value in section.items():
            if not hasattr(target, key):
                continue
            if section_name == "blender" and key == "executable" and value:
                value = Path(str(value)).expanduser()
            setattr(target, key, value)
    if "workspace" in data:
        cfg.workspace = _resolve_workspace(str(data["workspace"]))


def load_config(path: Path | None = None) -> AppConfig:
    """Build the effective configuration.

    Precedence: environment variables > TOML file > built-in defaults.
    """
    cfg = AppConfig()

    if path is None:
        env_path = os.environ.get("FA_CONFIG")
        candidate = Path(env_path).expanduser() if env_path else Path("filmautomator.toml")
        path = candidate if candidate.is_file() else None

    if path is not None and Path(path).is_file():
        with open(path, "rb") as fh:
            _apply_toml(cfg, tomllib.load(fh))

    # Environment overrides.
    if url := os.environ.get("FA_MODEL_BASE_URL"):
        cfg.gateway.openai_compat_base_url = url
    if key := os.environ.get("FA_MODEL_API_KEY"):
        cfg.gateway.openai_compat_api_key = key
    if ws := os.environ.get("FA_WORKSPACE"):
        cfg.workspace = _resolve_workspace(ws)
    if os.environ.get("FA_ENABLE_EXTERNAL_PROVIDERS") == "1":
        cfg.gateway.enable_external_providers = True
    # Storage overrides, so an owner can raise the ceiling on a machine with
    # more room without editing a file.
    if gb := os.environ.get("FA_STORAGE_MAX_GB"):
        cfg.storage.hard_limit_gb = float(gb)
        cfg.storage.max_working_gb = float(gb)
    if os.environ.get("FA_STORAGE_CLEANUP_DISABLED") == "1":
        cfg.storage.cleanup_enabled = False

    # Always absolute, whichever source supplied it.
    cfg.workspace = _resolve_workspace(cfg.workspace)

    if cfg.blender.executable is None:
        cfg.blender.executable = find_blender()

    return cfg

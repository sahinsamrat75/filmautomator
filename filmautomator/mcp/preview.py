"""Multimodal preview delivery for MCP clients (spec: the visual director loop).

The point of this module is narrow and important: the external AI must receive
the *actual rendered image*, not a filesystem path describing one. Telling a
model "the preview is at /some/path/preview.png" asks it to imagine a picture.
Delivering the bytes asks it to look at one.

MCP models image return values as a content block:

    {"type": "image", "data": "<base64>", "mimeType": "image/png"}

This module builds those blocks and, just as importantly, is honest when it
cannot. A client that cannot accept images gets a clearly labelled fallback --
the path, the metadata, and an explicit statement that no image was delivered --
rather than a silent omission that would let the AI believe it had seen a frame
it never received.

Nothing here interprets the image. The external model is the visual reasoning
system; this code only gets the picture to it intact.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Formats Blender and this pipeline actually emit for previews.
PREVIEW_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

#: Beyond this, an inline base64 image is large enough to be a problem rather
#: than a feature, so previews above it are reported rather than attached.
MAX_INLINE_BYTES = 12 * 1024 * 1024


class ImageDelivery:
    """How a preview was or was not delivered to the client."""

    DELIVERED = "delivered"
    UNSUPPORTED_BY_CLIENT = "unsupported_by_client"
    FILE_MISSING = "file_missing"
    UNREADABLE = "unreadable"
    TOO_LARGE = "too_large"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


@dataclass
class PreviewDelivery:
    """A preview, its image block if one could be built, and why not."""

    shot_id: str
    path: str
    delivery: str = ImageDelivery.FILE_MISSING
    image_block: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    bytes: int = 0
    mime_type: str = ""

    @property
    def delivered(self) -> bool:
        return self.image_block is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "path": self.path,
            "delivery": self.delivery,
            "image_delivered": self.delivered,
            "bytes": self.bytes,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
            "note": self.note,
        }


def mime_for(path: str | Path) -> str:
    """The MIME type to declare for a preview file."""
    suffix = Path(path).suffix.lower()
    if suffix in PREVIEW_MIME_TYPES:
        return PREVIEW_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"


def build_image_block(path: str | Path) -> dict[str, Any] | None:
    """Encode an image file as an MCP image content block.

    Returns None when the file cannot be read or is implausibly large, so the
    caller can report the limitation rather than emitting a broken block.
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        size = candidate.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    if size > MAX_INLINE_BYTES:
        log.warning(
            "preview %s is %.1f MB, above the %.1f MB inline limit",
            candidate, size / 1e6, MAX_INLINE_BYTES / 1e6,
        )
        return None
    try:
        data = candidate.read_bytes()
    except OSError as exc:
        log.warning("could not read preview %s: %s", candidate, exc)
        return None
    return {
        "type": "image",
        "data": base64.b64encode(data).decode("ascii"),
        "mimeType": mime_for(candidate),
    }


def client_accepts_images(client_capabilities: dict[str, Any] | None) -> bool:
    """Should this client be sent image content blocks?

    Defaults to True. That is a deliberate choice, not a loose one: the major
    MCP hosts — Claude Desktop included — do not advertise an ``imageContent``
    capability at all, yet they render image blocks perfectly well. Inferring
    "no capability declared, therefore no images" would silently strip the image
    from every working client, which is the one outcome the multimodal loop
    cannot recover from.

    MCP provides no standard way for a client to say "I cannot display images",
    so a host that genuinely cannot will either ignore the block or error
    visibly. Either is better than a working host being told it cannot see.

    A client may still opt out explicitly with ``imageContent: false``, which
    is honoured here so a constrained surface degrades to a clear text report
    instead of a broken response.
    """
    if not isinstance(client_capabilities, dict):
        return True
    if client_capabilities.get("imageContent") is False:
        return False
    experimental = client_capabilities.get("experimental")
    if isinstance(experimental, dict) and experimental.get("imageContent") is False:
        return False
    return True


def deliver_preview(
    shot_id: str,
    image_path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    client_capabilities: dict[str, Any] | None = None,
    include_image: bool = True,
) -> PreviewDelivery:
    """Package a preview for delivery, with an explicit verdict either way.

    The returned :class:`PreviewDelivery` always states whether an image was
    actually attached. Callers that build an MCP response from it can therefore
    never imply the AI saw something it did not.
    """
    path = Path(image_path) if image_path else None
    delivery = PreviewDelivery(
        shot_id=shot_id,
        path=str(path) if path else "",
        metadata=dict(metadata or {}),
    )

    if path is None or not path.is_file():
        delivery.delivery = ImageDelivery.FILE_MISSING
        delivery.note = (
            f"no preview image exists yet for {shot_id}. Render one with "
            f"render_shot_preview and call this again -- no image was delivered."
        )
        return delivery

    try:
        delivery.bytes = path.stat().st_size
    except OSError:
        delivery.bytes = 0
    delivery.mime_type = mime_for(path)

    if not include_image:
        delivery.delivery = ImageDelivery.UNSUPPORTED_BY_CLIENT
        delivery.note = (
            f"image delivery was not requested for this call; the preview is at "
            f"{path}. No image was delivered."
        )
        return delivery

    if client_capabilities is not None and not client_accepts_images(client_capabilities):
        delivery.delivery = ImageDelivery.UNSUPPORTED_BY_CLIENT
        delivery.note = (
            f"this MCP client did not advertise image content support, so no "
            f"image was delivered. The preview is at {path} "
            f"({delivery.bytes:,} bytes, {delivery.mime_type}). You cannot see "
            f"it from here; connect with a client that supports image content "
            f"to inspect frames visually."
        )
        return delivery

    block = build_image_block(path)
    if block is None:
        size = path.stat().st_size if path.exists() else 0
        delivery.delivery = (
            ImageDelivery.TOO_LARGE if size > MAX_INLINE_BYTES
            else ImageDelivery.UNREADABLE
        )
        delivery.note = (
            f"the preview at {path} could not be attached as an image "
            f"({size:,} bytes). No image was delivered."
        )
        return delivery

    delivery.image_block = block
    delivery.delivery = ImageDelivery.DELIVERED
    delivery.note = (
        f"the image above is the actual {shot_id} preview as rendered by "
        f"Blender ({delivery.bytes:,} bytes, {delivery.mime_type})."
    )
    return delivery


def build_preview_metadata(
    *,
    shot_id: str,
    scene_id: str = "",
    duration_s: float = 0.0,
    fps: int = 0,
    camera: dict[str, Any] | None = None,
    render_settings: dict[str, Any] | None = None,
    artifact_path: str = "",
    version: int = 0,
    qa: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the AI needs to judge a frame beyond the pixels themselves.

    Camera and duration change how an image should be read -- a 35mm wide shot at
    1.2m reads very differently from an 85mm close-up -- so the metadata travels
    with the image rather than being a separate lookup the client may skip.
    """
    metadata: dict[str, Any] = {
        "shot_id": shot_id,
        "scene_id": scene_id,
        "duration_s": round(duration_s, 3) if duration_s else 0.0,
        "fps": fps,
        "version": version,
        "camera": dict(camera or {}),
        "render_settings": dict(render_settings or {}),
        "artifact_path": artifact_path,
    }
    if qa is not None:
        metadata["qa"] = dict(qa)
    for key, value in (extra or {}).items():
        metadata.setdefault(key, value)
    return metadata


__all__ = [
    "ImageDelivery",
    "PreviewDelivery",
    "build_image_block",
    "build_preview_metadata",
    "client_accepts_images",
    "deliver_preview",
    "mime_for",
]

"""Core types for the Model Gateway.

Drivers translate these neutral types into whatever their backend speaks. No
agent ever imports a driver directly — they hand a :class:`ModelRequest` to the
gateway and get a :class:`ModelResponse` back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .capabilities import Capability


@dataclass(slots=True)
class ImageInput:
    """An image handed to a vision-capable model."""

    path: str
    #: Optional caption/context the model should know about this image.
    label: str = ""


@dataclass(slots=True)
class ModelRequest:
    """A single unit of work for a model."""

    capability: Capability
    prompt: str
    #: Optional system instruction.
    system: str = ""
    #: Images for vision requests.
    images: list[ImageInput] = field(default_factory=list)
    #: When set, the driver must return JSON matching this schema. Drivers
    #: that cannot enforce schemas still get the schema in the prompt.
    json_schema: dict[str, Any] | None = None
    #: Soft cap on generated tokens.
    max_tokens: int = 2048
    temperature: float = 0.7
    #: Free-form passthrough for driver-specific knobs.
    extra: dict[str, Any] = field(default_factory=dict)

    def schema_hint(self) -> str:
        """Render the JSON schema as an instruction block for the prompt."""
        if not self.json_schema:
            return ""
        return (
            "\n\nRespond with a single JSON object and nothing else. "
            "It must validate against this JSON Schema:\n"
            f"{json.dumps(self.json_schema, indent=2)}"
        )


@dataclass(slots=True)
class ModelResponse:
    """What came back, plus enough provenance to debug a bad result."""

    text: str
    #: Parsed JSON when a schema was requested and parsing succeeded.
    data: dict[str, Any] | None = None
    driver: str = ""
    model: str = ""
    #: True when a driver could not actually serve the request and a
    #: deterministic stand-in produced this response instead. Agents surface
    #: this so the owner is never misled about what a real model decided.
    synthetic: bool = False

    def require_data(self) -> dict[str, Any]:
        """Return parsed JSON, raising if the model did not produce any."""
        if self.data is None:
            raise ValueError(
                f"driver {self.driver!r} returned no parsable JSON "
                f"(first 200 chars: {self.text[:200]!r})"
            )
        return self.data


class DriverUnavailable(RuntimeError):
    """Raised when a driver cannot serve a request (down, model missing)."""


@runtime_checkable
class ModelDriver(Protocol):
    """What every model backend adapter must implement."""

    #: Stable identifier used in logs and config.
    name: str

    def capabilities(self) -> frozenset[Capability]:
        """Capabilities this driver can serve right now."""
        ...

    def available(self) -> bool:
        """Cheap health check — is the backend reachable?"""
        ...

    def describe(self) -> str:
        """One-line human description for the owner interface."""
        ...

    def invoke(self, request: ModelRequest) -> ModelResponse:
        """Perform the request. Raises DriverUnavailable on failure."""
        ...

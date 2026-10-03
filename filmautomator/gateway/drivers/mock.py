"""Deterministic offline driver.

This exists so the production pipeline can be exercised end-to-end without any
model server running, and so a task degrades instead of dying when the owner's
backend is down.

It is NOT a language model and never pretends to be one: every response it
produces is flagged ``synthetic=True``, and agents propagate that flag so the
owner interface can say plainly that a decision was made by the stand-in rather
than by a model.
"""

from __future__ import annotations

from typing import Any

from ..capabilities import Capability
from ..base import ModelRequest, ModelResponse


def synthesize_from_schema(schema: dict[str, Any], seed_text: str = "") -> Any:
    """Build a schema-valid placeholder value.

    Honours ``enum``, ``type``, ``required`` and ``properties``. Numbers are
    biased into a sane filmmaking range so downstream code that clamps or
    validates them does not immediately reject the stand-in's output.
    """
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]

    schema_type = schema.get("type")

    if schema_type == "object" or "properties" in schema:
        props = schema.get("properties", {})
        required = set(schema.get("required", props.keys()))
        return {
            key: synthesize_from_schema(sub, seed_text)
            for key, sub in props.items()
            if key in required
        }

    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [synthesize_from_schema(item_schema, seed_text)]

    if schema_type == "integer":
        return _bounded_number(schema, default=1, integral=True)
    if schema_type == "number":
        return _bounded_number(schema, default=1.0, integral=False)
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None

    # Strings, and anything unspecified.
    return _placeholder_string(schema, seed_text)


def _bounded_number(schema: dict[str, Any], default: float, integral: bool) -> Any:
    low = schema.get("minimum")
    high = schema.get("maximum")
    value = default
    if isinstance(low, (int, float)) and value < low:
        value = low
    if isinstance(high, (int, float)) and value > high:
        value = high
    return int(value) if integral else float(value)


def _placeholder_string(schema: dict[str, Any], seed_text: str) -> str:
    description = str(schema.get("description", "")).lower()
    # Give downstream numeric parsers something safe to fail on rather than a
    # sentence, when the field clearly wants a machine-readable value.
    if "hex" in description or "color" in description:
        return "#808080"
    if "path" in description:
        return ""
    return seed_text.strip()[:120] or "placeholder"


class MockDriver:
    """Always-available stand-in that satisfies the driver protocol."""

    name = "mock"

    def capabilities(self) -> frozenset[Capability]:
        # Claims everything so the pipeline can always make progress offline.
        return frozenset(Capability)

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "Deterministic offline stand-in — no model involved (synthetic output)"

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if request.json_schema:
            data = synthesize_from_schema(request.json_schema)
            import json

            text = json.dumps(data, indent=2)
        else:
            data = None
            text = (
                "[synthetic] No model backend was reachable, so no generated "
                f"text is available for capability {request.capability}."
            )
        return ModelResponse(
            text=text,
            data=data,
            driver=self.name,
            model="mock",
            synthetic=True,
        )

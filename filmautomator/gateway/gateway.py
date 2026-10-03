"""The Model Gateway.

Spec section 2: every model invocation in the system goes through here. Agents
ask for a *capability*, never for a specific model, so swapping the owner's
backend is a config change rather than a code change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import GatewayConfig
from .base import (
    DriverUnavailable,
    ImageInput,
    ModelDriver,
    ModelRequest,
    ModelResponse,
)
from .capabilities import Capability
from .drivers.mock import MockDriver
from .drivers.openai_compat import OpenAICompatDriver

log = logging.getLogger(__name__)


class NoCapableDriver(RuntimeError):
    """No registered driver can serve the requested capability."""


@dataclass(slots=True)
class GatewayReport:
    """What the owner interface shows about model availability."""

    drivers: list[tuple[str, bool, str]] = field(default_factory=list)
    capability_map: dict[str, list[str]] = field(default_factory=dict)
    unimplemented: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["Model Gateway"]
        for name, ok, description in self.drivers:
            mark = "ok " if ok else "-- "
            lines.append(f"  [{mark}] {name}: {description}")
        for cap in sorted(self.capability_map):
            providers = self.capability_map[cap]
            lines.append(f"  {cap:<20} <- {', '.join(providers) or 'NONE'}")
        if self.unimplemented:
            lines.append(
                "  not implemented (no free local driver yet): "
                + ", ".join(sorted(self.unimplemented))
            )
        return "\n".join(lines)


class ModelGateway:
    """Routes capability requests to the first driver that can serve them."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()
        self._drivers: list[ModelDriver] = []
        self._mock = MockDriver()
        self._install_default_drivers()

    def _install_default_drivers(self) -> None:
        local = OpenAICompatDriver(
            base_url=self.config.openai_compat_base_url,
            api_key=self.config.openai_compat_api_key,
            timeout_s=self.config.openai_compat_timeout_s,
            model_overrides=self.config.models,
        )
        self.register(local)
        # The stand-in goes last so it is only ever reached when nothing real
        # can serve the request.
        self.register(self._mock)

    # -- registration ------------------------------------------------------

    def register(self, driver: ModelDriver) -> None:
        self._drivers.append(driver)

    def drivers(self) -> list[ModelDriver]:
        return list(self._drivers)

    def capability_map(self) -> dict[Capability, list[str]]:
        """Which driver names can serve each capability."""
        mapping: dict[Capability, list[str]] = {}
        for driver in self._drivers:
            if driver is self._mock:
                continue  # never advertise the stand-in as a real provider
            try:
                caps = driver.capabilities()
            except Exception:  # noqa: BLE001 - a broken driver is just unusable
                continue
            for cap in caps:
                mapping.setdefault(cap, []).append(driver.name)
        return mapping

    def report(self) -> GatewayReport:
        drivers: list[tuple[str, bool, str]] = []
        for driver in self._drivers:
            try:
                ok = driver.available()
            except Exception:  # noqa: BLE001
                ok = False
            drivers.append((driver.name, ok, driver.describe()))

        mapping = self.capability_map()
        from .capabilities import UNIMPLEMENTED_CAPABILITIES

        return GatewayReport(
            drivers=drivers,
            capability_map={str(c): names for c, names in mapping.items()},
            unimplemented=[str(c) for c in UNIMPLEMENTED_CAPABILITIES],
        )

    # -- routing -----------------------------------------------------------

    def _candidates(self, capability: Capability) -> list[ModelDriver]:
        """Real drivers that claim the capability, in registration order."""
        out: list[ModelDriver] = []
        for driver in self._drivers:
            if driver is self._mock:
                continue
            try:
                if capability in driver.capabilities() and driver.available():
                    out.append(driver)
            except Exception as exc:  # noqa: BLE001
                log.debug("driver %s failed capability probe: %s", driver.name, exc)
        return out

    def supports(self, capability: Capability) -> bool:
        return bool(self._candidates(capability))

    def invoke(
        self,
        capability: Capability,
        prompt: str,
        *,
        system: str = "",
        images: list[ImageInput] | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        extra: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Run one model call, falling back across drivers.

        Raises :class:`NoCapableDriver` only when nothing at all can serve the
        request and mock fallback is disabled.
        """
        request = ModelRequest(
            capability=capability,
            prompt=prompt,
            system=system,
            images=list(images or []),
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=dict(extra or {}),
        )

        errors: list[str] = []
        for driver in self._candidates(capability):
            try:
                return driver.invoke(request)
            except DriverUnavailable as exc:
                errors.append(f"{driver.name}: {exc}")
                log.warning("driver %s unavailable: %s", driver.name, exc)
            except Exception as exc:  # noqa: BLE001 - try the next driver
                errors.append(f"{driver.name}: {exc!r}")
                log.warning("driver %s raised: %r", driver.name, exc)

        if self.config.allow_mock_fallback:
            reason = "; ".join(errors) if errors else "no driver claims this capability"
            log.warning(
                "falling back to synthetic output for %s (%s)", capability, reason
            )
            return self._mock.invoke(request)

        detail = "; ".join(errors) if errors else "capability not implemented"
        raise NoCapableDriver(f"no driver served {capability}: {detail}")

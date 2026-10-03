"""OpenAI-compatible driver.

Speaks the ``/v1/chat/completions`` and ``/v1/models`` endpoints that LM Studio,
llama.cpp's server, vLLM, text-generation-webui and Ollama's compatibility shim
all expose. Because it is OpenAI-shaped it also covers any local server the
owner already runs, without the system taking a dependency on a vendor.

Uses only the standard library, so the gateway has no third-party HTTP
dependency and cannot break because a package changed underneath it.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from ..capabilities import Capability
from ..base import DriverUnavailable, ImageInput, ModelRequest, ModelResponse

#: Substrings that reliably indicate a vision-capable model. LM Studio does not
#: report modality in /v1/models, so this is a name heuristic; the owner can
#: override it by naming the vision model explicitly in the config.
_VISION_NAME_HINTS = (
    "vl", "vision", "llava", "bakllava", "moondream", "minicpm-v",
    "gemma-3", "gemma3", "pixtral", "internvl", "qwen2-vl", "qwen2.5-vl",
    "phi-3.5-vision", "phi-4-multimodal", "smolvlm", "idefics",
)

#: Text-only reasoning/planning/coding are served by any instruct model, so a
#: driver with at least one loaded model can claim them.
_TEXT_CAPABILITIES = frozenset(
    {
        Capability.REASONING,
        Capability.PLANNING,
        Capability.CODING,
    }
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model reply.

    Local models wrap JSON in prose or code fences far more often than hosted
    ones, so try the fence first, then a brace-balanced scan.
    """
    if not text:
        return None

    for candidate in _FENCE_RE.findall(text):
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Brace-balanced scan, string-aware so braces inside strings don't fool us.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : idx + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    return None


class OpenAICompatDriver:
    """Driver for any OpenAI-compatible local inference server."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        timeout_s: float = 300.0,
        model_overrides: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.model_overrides = dict(model_overrides or {})
        self._models: list[str] | None = None
        self._model_cache_error: str = ""

    # -- discovery ---------------------------------------------------------

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self, refresh: bool = False) -> list[str]:
        """Model ids the server currently offers. Cached after first success."""
        if self._models is not None and not refresh:
            return self._models
        try:
            payload = self._get("/models")
        except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
            self._model_cache_error = str(exc)
            self._models = []
            return self._models
        data = payload.get("data")
        if isinstance(data, list):
            self._models = [
                str(entry.get("id"))
                for entry in data
                if isinstance(entry, dict) and entry.get("id")
            ]
        else:
            self._models = []
        return self._models

    def _vision_models(self) -> list[str]:
        return [m for m in self.list_models() if self._looks_like_vision(m)]

    @staticmethod
    def _looks_like_vision(model_id: str) -> bool:
        low = model_id.lower()
        return any(hint in low for hint in _VISION_NAME_HINTS)

    def _pick_model(self, capability: Capability) -> str | None:
        """Resolve which model to use for a capability."""
        override = self.model_overrides.get(capability.value)
        if override:
            return override
        ids = self.list_models()
        if not ids:
            return None
        if capability is Capability.VISION:
            vision = [m for m in ids if self._looks_like_vision(m)]
            return vision[0] if vision else None
        # Prefer a non-vision instruct model for text work, but any model works.
        text_only = [m for m in ids if not self._looks_like_vision(m)]
        return (text_only or ids)[0]

    # -- capability reporting ---------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        """Capabilities this driver can serve *right now*.

        An unreachable server can serve nothing, so this returns empty rather
        than advertising what it would offer if it were up. The gateway shows
        reachability separately, and the owner-facing report saying NONE is the
        accurate and actionable answer.
        """
        if not self.available():
            return frozenset()
        caps = set(_TEXT_CAPABILITIES)
        if self._vision_models():
            caps.add(Capability.VISION)
        return frozenset(caps)

    def available(self) -> bool:
        return bool(self.list_models())

    def describe(self) -> str:
        models = self.list_models()
        if not models:
            detail = f"unreachable ({self._model_cache_error})" if self._model_cache_error else "no models loaded"
            return f"OpenAI-compatible @ {self.base_url} — {detail}"
        vision = self._vision_models()
        vision_note = f", vision: {vision[0]}" if vision else ", no vision model"
        return f"OpenAI-compatible @ {self.base_url} — {len(models)} model(s){vision_note}"

    # -- invocation --------------------------------------------------------

    def invoke(self, request: ModelRequest) -> ModelResponse:
        model = self._pick_model(request.capability)
        if model is None:
            raise DriverUnavailable(
                f"no model on {self.base_url} satisfies capability "
                f"{request.capability}"
            )

        content: list[dict] = [{"type": "text", "text": request.prompt}]
        for image in request.images:
            content.append(self._image_part(image))

        messages: list[dict] = []
        system = request.system
        if request.json_schema:
            system = (system + request.schema_hint()).strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {"role": "user", "content": content if request.images else request.prompt}
        )

        body = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        body.update(request.extra)

        try:
            payload = self._post("/chat/completions", body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise DriverUnavailable(
                f"{self.base_url} rejected the request (HTTP {exc.code}): {detail}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise DriverUnavailable(f"{self.base_url} unreachable: {exc}") from exc

        text = self._first_message_text(payload)
        data = _extract_json(text) if request.json_schema else None
        return ModelResponse(text=text, data=data, driver=self.name, model=model)

    @staticmethod
    def _image_part(image: ImageInput) -> dict:
        path = Path(image.path)
        if not path.is_file():
            raise DriverUnavailable(f"image not found: {image.path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lstrip(".").lower() or "png"
        mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _first_message_text(payload: dict) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some servers return content as a list of parts.
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        return ""

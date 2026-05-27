from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .base import AIAdapter

T = TypeVar("T", bound=BaseModel)


class MockAIAdapter(AIAdapter):
    """Deterministic adapter for testing pipelines without a real AI provider."""

    def __init__(self, handler: Callable[[str, dict | None], str] | None = None) -> None:
        self._handler = handler or (lambda prompt, ctx: f"mock response to: {prompt[:40]}")

    def run(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        return self._handler(prompt, context)

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        raw = self._handler(prompt, context)
        try:
            return response_model.model_validate_json(raw)
        except Exception:
            schema = response_model.model_json_schema()
            defaults = {
                k: self._type_default(v)
                for k, v in schema.get("properties", {}).items()
            }
            return response_model.model_validate(defaults)

    async def run_async(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        return self.run(prompt, context, system)

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        return self.run_structured(prompt, response_model, context, system)

    async def run_structured_streaming(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
        on_chunk=None,
    ) -> T:
        """Simulate streaming by calling on_chunk once with the complete JSON."""
        result = self.run_structured(prompt, response_model, context, system)
        if on_chunk:
            on_chunk(1, result.model_dump_json())
        return result

    async def run_streaming(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ):
        """Yield text one character at a time so streaming tests are realistic."""
        text = self.run(prompt, context, system)
        for char in text:
            yield char

    @staticmethod
    def _type_default(field_schema: dict) -> Any:
        if "default" in field_schema:
            return field_schema["default"]

        # anyOf / oneOf (e.g. Optional[X] → [X, null])
        for union_key in ("anyOf", "oneOf"):
            if union_key in field_schema:
                for opt in field_schema[union_key]:
                    if opt.get("type") != "null":
                        return MockAIAdapter._type_default(opt)
                return None

        t = field_schema.get("type")
        if t == "string":
            return ""
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return False
        if t == "array":
            return []
        # object, additionalProperties (dict[K, V]), $ref — all map to {}
        if t == "object" or "additionalProperties" in field_schema or "$ref" in field_schema:
            return {}
        return None

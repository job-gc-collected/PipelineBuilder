from __future__ import annotations

from typing import AsyncGenerator, TypeVar

from pydantic import BaseModel

from .base import AIAdapter

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o"


class OpenAIAdapter(AIAdapter):
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("Run: pip install openai")

        self._client = openai.OpenAI(api_key=api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    def run(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        messages = self._build_messages(prompt, context, system=system)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        """Use OpenAI structured outputs (guaranteed JSON matching schema)."""
        messages = self._build_messages(prompt, context, system=system)
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    async def run_async(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> str:
        messages = self._build_messages(prompt, context, system=system)
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    async def run_structured_async(
        self,
        prompt: str,
        response_model: type[T],
        context: dict | None = None,
        system: str | None = None,
    ) -> T:
        messages = self._build_messages(prompt, context, system=system)
        response = await self._async_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    async def run_streaming(
        self,
        prompt: str,
        context: dict | None = None,
        system: str | None = None,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, context, system=system)
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _build_messages(
        self, prompt: str, context: dict | None, system: str | None = None
    ) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(super()._build_messages(prompt, context))
        return msgs

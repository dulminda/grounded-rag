"""LLM client abstraction with a Claude implementation.

Every call returns a validated pydantic object plus token usage and cost, so the pipeline and
the evaluation harness can report $/query alongside quality.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from grounded.generation.schemas import strict_json_schema

T = TypeVar("T", bound=BaseModel)

# USD per million tokens (input, output). Keep in sync with https://www.anthropic.com/pricing
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMRefusal(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    usage: Usage
    model: str
    latency_ms: float


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, prompt: str, schema: type[T]) -> LLMResult[T]: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


class ClaudeClient:
    """Claude via the Anthropic SDK, with JSON-schema structured output.

    Server-side refusal fallback (``fallbacks="default"``) is enabled so a safety-classifier
    decline is retried on Anthropic's recommended fallback model instead of failing the query.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        fallback: bool = True,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)
        self.model, self.effort = model, effort
        self.max_tokens, self.fallback = max_tokens, fallback

    def generate(self, system: str, prompt: str, schema: type[T]) -> LLMResult[T]:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
        )
        start = time.perf_counter()
        try:
            if self.fallback:
                response = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = self._client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the Claude API: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMRefusal(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("response contained no text block")
        try:
            parsed = schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            raise LLMError(f"response did not match {schema.__name__}: {e}") from e

        served_by = response.model
        u = response.usage
        usage = Usage(u.input_tokens, u.output_tokens, cost_usd(served_by, u.input_tokens,
                                                                 u.output_tokens))
        return LLMResult(parsed=parsed, usage=usage, model=served_by, latency_ms=latency_ms)

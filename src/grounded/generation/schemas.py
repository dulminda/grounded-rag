"""Structured outputs the LLM must return. Enforced server-side via JSON-schema output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The answer, with inline [n] markers for each cited source.")
    citations: list[int] = Field(description="Source numbers the answer relies on.")
    insufficient_context: bool = Field(
        description="True if the sources do not contain enough information to answer."
    )


class ClaimVerdict(BaseModel):
    verdict: Literal["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    rationale: str = Field(description="One or two sentences explaining the verdict.")
    citations: list[int] = Field(description="Source numbers that support or contradict the claim.")


class JudgedClaim(BaseModel):
    claim: str
    supported: bool
    source_ids: list[int]


class FaithfulnessJudgement(BaseModel):
    claims: list[JudgedClaim] = Field(description="Every factual claim made in the answer.")
    answers_question: bool = Field(description="Whether the answer addresses the question asked.")


class SyntheticQuestion(BaseModel):
    question: str
    answer: str
    answerable: bool = Field(description="False if the passage cannot support a good question.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema -> strict JSON schema (all fields required, no extra properties)."""
    schema = model.model_json_schema()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            node.pop("title", None)
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(schema)
    return schema

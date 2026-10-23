"""LLM-as-judge for answer faithfulness.

The judge decomposes an answer into atomic claims and checks each one against the sources it
was generated from. Faithfulness = supported claims / total claims. Before trusting the judge
on a new domain, spot-check ~50 judgements by hand and report the agreement rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from grounded.generation.llm import LLMClient
from grounded.generation.prompts import JUDGE_SYSTEM, format_sources, judge_prompt
from grounded.generation.schemas import FaithfulnessJudgement
from grounded.types import ScoredChunk


@dataclass
class Verdict:
    faithfulness: float
    answers_question: bool
    n_claims: int
    unsupported: list[str]
    cost_usd: float


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def judge(self, question: str, sources: list[ScoredChunk], answer: str) -> Verdict:
        prompt = judge_prompt(question, format_sources(sources), answer)
        result = self.llm.generate(JUDGE_SYSTEM, prompt, FaithfulnessJudgement)
        claims = result.parsed.claims
        supported = sum(c.supported for c in claims)
        return Verdict(
            faithfulness=supported / len(claims) if claims else 1.0,
            answers_question=result.parsed.answers_question,
            n_claims=len(claims),
            unsupported=[c.claim for c in claims if not c.supported],
            cost_usd=result.usage.cost_usd,
        )

"""Protocol and shared types for LLM-based extractors.

An AssumptionExtractor takes a DiagramReading and the problem text and returns
an Extraction (assumptions list + goal string). LLM-based implementations
(GeminiExtractor, ClaudeExtractor) follow this Protocol; the deterministic
regex path in assumption_extraction.py does not implement it but is called
directly by the pipeline when consensus mode is disabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from diagram_theorem_assistant.schema import DiagramReading


@dataclass(frozen=True)
class Extraction:
    """Minimal assumption set + goal for a problem."""

    assumptions: list[str]
    goal: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": list(self.assumptions),
            "goal": self.goal,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Extraction":
        return cls(
            assumptions=list(data.get("assumptions", [])),
            goal=str(data.get("goal", "")),
            raw=dict(data.get("raw", {})),
        )


class AssumptionExtractor(Protocol):
    def extract(self, reading: DiagramReading, problem_text: str) -> Extraction: ...


EXTRACTION_PROMPT = """You are extracting the formal structure of a geometry problem.

You are given:
- A structured VLM reading of the diagram (objects, relations, visual marks)
- The problem text (may describe givens and ask to "prove", "show", or "find" something)

Your job is to output the MINIMAL set of ASSUMPTIONS (given facts) and the single GOAL (what to prove).

Rules:
1. Assumptions must come from the reading or from clearly-stated givens in the text. Do NOT invent facts the diagram does not support.
2. The goal must NEVER appear in the assumption list. If the text says "Prove PA = PB", then "PA = PB" is the goal and is NOT an assumption.
3. Prefer canonical forms: "triangle A B C" over "triangle ABC" or "△ABC"; "AB = AC" over "AB is equal to AC".
4. If the problem text gives the goal explicitly after "prove", "show", or "find", use that verbatim (normalized) as the goal. Otherwise, pick the most likely goal based on the diagram.
5. Keep the assumption list short and meaningful. Skip trivial incidence facts unless they're load-bearing.

VLM reading:
{reading}

Problem text:
{problem_text}

Respond with ONLY valid JSON in this exact shape:
{{"assumptions": ["...", "..."], "goal": "..."}}
"""

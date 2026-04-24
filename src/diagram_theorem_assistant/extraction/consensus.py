from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from diagram_theorem_assistant.consensus.judge import JudgeLLM, JudgeVerdict
from diagram_theorem_assistant.extraction.base import AssumptionExtractor, Extraction
from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError


class ConsensusExtractor:
    """Dual-provider extraction with judge-mediated consensus.

    Merge strategy is conservative: when the judge says "merge", we take the
    INTERSECTION of the two assumption lists (rejecting anything only one
    side produced) to avoid inheriting hallucinations.
    """

    def __init__(
        self,
        primary: AssumptionExtractor,
        secondary: AssumptionExtractor,
        judge: JudgeLLM,
        *,
        max_attempts: int = 3,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.judge = judge
        self.max_attempts = max_attempts

    def extract(self, reading: DiagramReading, problem_text: str) -> Extraction:
        last_a: Extraction | None = None
        last_b: Extraction | None = None
        last_reason = ""
        for _ in range(self.max_attempts):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(self.primary.extract, reading, problem_text)
                fut_b = pool.submit(self.secondary.extract, reading, problem_text)
                last_a = fut_a.result()
                last_b = fut_b.result()
            verdict: JudgeVerdict = self.judge.compare_extractions(last_a, last_b)
            if verdict.equivalent:
                if verdict.preferred == "a":
                    return last_a
                if verdict.preferred == "b":
                    return last_b
                return _merge_intersection(last_a, last_b, strategy="judge-preferred-merge")
            last_reason = verdict.reason
        # Attempts exhausted — fall back to intersection merge. Conservative
        # by design: we drop anything only one provider produced, keeping the
        # facts both agreed on.
        assert last_a is not None and last_b is not None
        return _merge_intersection(
            last_a, last_b,
            strategy=f"fallback-after-{self.max_attempts}-disagreements: {last_reason}",
        )


def _merge_intersection(a: Extraction, b: Extraction, *, strategy: str = "merge") -> Extraction:
    b_set = set(b.assumptions)
    intersected = [x for x in a.assumptions if x in b_set]
    if a.goal and b.goal and a.goal == b.goal:
        goal = a.goal
    elif a.goal:
        goal = a.goal
    else:
        goal = b.goal
    return Extraction(
        assumptions=intersected,
        goal=goal,
        raw={"a": a.raw, "b": b.raw, "strategy": strategy},
    )

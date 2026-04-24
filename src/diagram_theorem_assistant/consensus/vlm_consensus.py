from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from diagram_theorem_assistant.consensus.judge import JudgeLLM, JudgeVerdict
from diagram_theorem_assistant.schema import DiagramMark, DiagramReading
from diagram_theorem_assistant.vlm.base import DiagramUnderstander


class ConsensusVLMAdapter:
    """Dual-provider VLM adapter with judge-mediated consensus.

    Calls primary and secondary adapters in parallel. A JudgeLLM decides
    whether their readings agree. On disagreement, retries up to max_attempts.
    When retries are exhausted, falls back to a union-merge of the last two
    readings rather than raising, since partial disagreement on real diagrams
    is common and both readings typically capture valid facts.
    """

    def __init__(
        self,
        primary: DiagramUnderstander,
        secondary: DiagramUnderstander,
        judge: JudgeLLM,
        *,
        max_attempts: int = 3,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.judge = judge
        self.max_attempts = max_attempts

    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading:
        last_a: DiagramReading | None = None
        last_b: DiagramReading | None = None
        last_reason = ""
        for _ in range(self.max_attempts):
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_a = pool.submit(self.primary.read, image_path, fixture_path=fixture_path)
                fut_b = pool.submit(self.secondary.read, image_path, fixture_path=fixture_path)
                last_a = fut_a.result()
                last_b = fut_b.result()
            verdict: JudgeVerdict = self.judge.compare_readings(last_a, last_b)
            if verdict.equivalent:
                if verdict.preferred == "a":
                    return last_a
                if verdict.preferred == "b":
                    return last_b
                return _merge(last_a, last_b, reason="judge-preferred-merge")
            last_reason = verdict.reason
        # Attempts exhausted — fall back to union merge. This is a best-effort
        # recovery: downstream stages (extraction → Lean) can still filter
        # hallucinations, and the judge's blame loop can redo a bad stage.
        assert last_a is not None and last_b is not None
        return _merge(last_a, last_b, reason=f"fallback-after-{self.max_attempts}-disagreements: {last_reason}")


def _merge(a: DiagramReading, b: DiagramReading, *, reason: str = "merge") -> DiagramReading:
    objects = _dedup(a.objects + b.objects)
    relations = _dedup(a.relations + b.relations)
    marks = _dedup_marks(a.marks + b.marks)
    return DiagramReading(
        objects=objects,
        relations=relations,
        marks=marks,
        raw_vlm_output={
            "a": a.raw_vlm_output,
            "b": b.raw_vlm_output,
            "strategy": reason,
        },
    )


def _dedup(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _mark_key(mark: DiagramMark) -> tuple:
    return (
        mark.type,
        tuple(mark.points),
        tuple(mark.segments),
        tuple(mark.lines),
        mark.point,
        mark.segment,
    )


def _dedup_marks(marks: list[DiagramMark]) -> list[DiagramMark]:
    seen = set()
    out: list[DiagramMark] = []
    for mark in marks:
        key = _mark_key(mark)
        if key not in seen:
            seen.add(key)
            out.append(mark)
    return out

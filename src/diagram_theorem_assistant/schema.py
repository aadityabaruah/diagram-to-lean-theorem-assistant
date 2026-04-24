from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagramMark:
    type: str
    points: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    point: str | None = None
    segment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [])}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagramMark:
        return cls(
            type=data["type"],
            points=list(data.get("points", [])),
            segments=list(data.get("segments", [])),
            lines=list(data.get("lines", [])),
            point=data.get("point"),
            segment=data.get("segment"),
        )


@dataclass(frozen=True)
class DiagramReading:
    objects: list[str]
    relations: list[str]
    marks: list[DiagramMark]
    raw_vlm_output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": list(self.objects),
            "relations": list(self.relations),
            "marks": [mark.to_dict() for mark in self.marks],
            "raw_vlm_output": dict(self.raw_vlm_output),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagramReading:
        return cls(
            objects=list(data.get("objects", [])),
            relations=list(data.get("relations", [])),
            marks=[DiagramMark.from_dict(item) for item in data.get("marks", [])],
            raw_vlm_output=dict(data.get("raw_vlm_output", {})),
        )


class LeanStatus(StrEnum):
    OK = "ok"
    TYPE_ERROR = "type_error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PipelineResult:
    reading: DiagramReading
    candidate_assumptions: list[str]
    confirmed_assumptions: list[str]
    goal_candidates: list[str]
    selected_goal: str
    lean_source: str
    lean_status: LeanStatus
    lean_stderr: str | None


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    image: str
    problem_text: str
    objects: list[str]
    diagram_marks: list[dict[str, Any]]
    gold_assumptions: list[str]
    gold_goal: str
    acceptable_goals: list[str]
    lean_theorem_name: str
    vlm_fixture: str = ""
    category: str = "text-guided"
    gold_relations: list[str] = field(default_factory=list)


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    raw_examples = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkExample(**raw_example) for raw_example in raw_examples]

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    raw_examples = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkExample(**raw_example) for raw_example in raw_examples]

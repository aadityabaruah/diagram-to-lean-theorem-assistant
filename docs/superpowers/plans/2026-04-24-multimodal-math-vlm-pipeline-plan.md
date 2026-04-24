# Multimodal Math VLM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing text-only diagram-theorem prototype into a working VLM→Lean pipeline: Gemini-backed diagram understanding with a fixture fallback for tests, real mathlib-backed Lean type-checking via subprocess, a Streamlit UI, and a benchmark that runs end-to-end on 5 synthetic examples.

**Architecture:** Four layers — Streamlit UI, pipeline orchestrator, pluggable VLM adapters (Gemini + Fixture), and a Lake-based Lean project invoked as a subprocess. Each layer talks through typed dataclasses in `schema.py`. All tests default to the fixture adapter; `@pytest.mark.lean` and `@pytest.mark.live` opt into real tooling.

**Tech Stack:** Python 3.11, google-genai, Pillow, matplotlib, Streamlit, python-dotenv, pytest, pytest-mock; Lean 4.30 + Lake 5 + mathlib4.

---

## File Structure

After all tasks complete, the repo looks like:

```
diagram-to-lean-theorem-assistant/
├── app.py                                              # NEW (task 15)
├── pyproject.toml                                      # MOD (task 1)
├── .env                                                # already present
├── examples/
│   ├── benchmark.json                                  # MOD (task 3)
│   ├── images/synthetic/*.png                          # NEW (task 3)
│   ├── images/geometry3k/                              # fetched (task 14), gitignored
│   └── vlm_fixtures/*.json                             # NEW (task 3)
├── lean_project/                                       # NEW (task 6)
│   ├── lakefile.toml
│   ├── lean-toolchain
│   └── DiagramTheorems/
│       ├── Basic.lean
│       └── Generated.lean                              # gitignored
├── scripts/
│   ├── generate_synthetic.py                           # NEW (task 3)
│   ├── record_vlm_fixtures.py                          # NEW (task 14)
│   └── fetch_geometry3k.py                             # NEW (task 14)
├── src/diagram_theorem_assistant/
│   ├── __init__.py
│   ├── schema.py                                       # MOD (task 2)
│   ├── vlm/
│   │   ├── __init__.py                                 # NEW (task 4)
│   │   ├── base.py                                     # NEW (task 4)
│   │   ├── fixture.py                                  # NEW (task 4)
│   │   ├── prompts.py                                  # NEW (task 5)
│   │   └── gemini.py                                   # NEW (task 5)
│   ├── assumption_extraction.py                        # MOD (task 8)
│   ├── goal_generation.py                              # already complete, no change
│   ├── lean_export.py                                  # REWRITE (task 9)
│   ├── lean_runner.py                                  # NEW (task 10)
│   ├── pipeline.py                                     # NEW (task 11)
│   ├── metrics.py                                      # RENAMED from evaluation.py (task 12)
│   └── demo.py                                         # MOD (task 13)
└── tests/
    ├── test_schema.py                                  # MOD (task 2)
    ├── test_assumption_extraction.py                   # MOD (task 8)
    ├── test_goal_generation.py                         # no change
    ├── test_lean_export.py                             # REWRITE (task 9)
    ├── test_metrics.py                                 # RENAMED (task 12)
    ├── test_demo.py                                    # MOD (task 13)
    ├── test_pipeline.py                                # NEW (task 11)
    ├── test_lean_runner.py                             # NEW (task 10)
    ├── test_gemini_live.py                             # NEW (task 5, @pytest.mark.live)
    └── vlm/
        ├── __init__.py                                 # NEW (task 4)
        ├── test_fixture_adapter.py                     # NEW (task 4)
        └── test_gemini_adapter.py                      # NEW (task 5)
```

---

### Task 1: Add dependencies and pytest markers to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `pyproject.toml`**

Replace the entire file content with:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "diagram-theorem-assistant"
version = "0.2.0"
description = "Translate geometry diagrams and problem text into Lean 4 theorem candidates via a vision-language model."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "google-genai>=0.3",
  "pillow>=10",
  "matplotlib>=3.8",
  "streamlit>=1.32",
  "python-dotenv>=1.0",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-mock>=3.12",
  "pytest-cov>=5",
]

[project.scripts]
diagram-theorem-demo = "diagram_theorem_assistant.demo:main"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
  "lean: tests that shell out to a real `lake build` (slow, require Lean + mathlib cached)",
  "live: tests that call the live Gemini API (cost money, require GEMINI_API_KEY)",
]
```

- [ ] **Step 2: Install the deps into the venv**

Run:

```bash
cd "/Users/aadityab/Documents/New project" && source .venv/bin/activate && pip install -e ".[dev]"
```

Expected: pip installs all listed packages. Takes ~2 minutes first time.

- [ ] **Step 3: Verify the existing tests still pass**

Run:

```bash
cd "/Users/aadityab/Documents/New project" && source .venv/bin/activate && pytest -q --strict-markers
```

Expected: all existing tests pass (`17 passed`). No marker warnings.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml && git commit -m "chore: add VLM, UI, and test deps; register lean/live pytest markers"
```

---

### Task 2: Expand `schema.py` with VLM and pipeline types

**Files:**
- Modify: `src/diagram_theorem_assistant/schema.py`
- Modify: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests for the new schema types**

Replace `tests/test_schema.py` with:

```python
import json
from pathlib import Path

import pytest

from diagram_theorem_assistant.schema import (
    BenchmarkExample,
    DiagramMark,
    DiagramReading,
    LeanStatus,
    PipelineResult,
    load_benchmark,
)


def test_benchmark_example_accepts_new_fields():
    example = BenchmarkExample(
        id="isosceles_triangle_base_angles",
        image="examples/images/synthetic/isosceles_triangle.png",
        vlm_fixture="examples/vlm_fixtures/isosceles_triangle_base_angles.json",
        category="text-guided",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C"],
        diagram_marks=[{"type": "equal_length", "segments": ["AB", "AC"]}],
        gold_relations=["triangle A B C"],
        gold_assumptions=["triangle A B C", "AB = AC"],
        gold_goal="angle ABC = angle BCA",
        acceptable_goals=["angle CBA = angle BCA"],
        lean_theorem_name="isosceles_base_angles",
    )
    assert example.category == "text-guided"
    assert example.vlm_fixture.endswith(".json")


def test_diagram_mark_is_frozen_dataclass():
    mark = DiagramMark(type="right_angle", points=["A", "C", "B"])
    assert mark.type == "right_angle"
    with pytest.raises(Exception):
        mark.type = "other"  # type: ignore[misc]


def test_diagram_reading_round_trip_via_dict():
    reading = DiagramReading(
        objects=["point A", "segment AB"],
        relations=["collinear A B C"],
        marks=[DiagramMark(type="right_angle", points=["A", "C", "B"])],
        raw_vlm_output={"source": "fixture"},
    )
    restored = DiagramReading.from_dict(reading.to_dict())
    assert restored == reading


def test_pipeline_result_captures_all_stages():
    reading = DiagramReading(objects=[], relations=[], marks=[], raw_vlm_output={})
    result = PipelineResult(
        reading=reading,
        candidate_assumptions=[],
        confirmed_assumptions=[],
        goal_candidates=["x = y"],
        selected_goal="x = y",
        lean_source="-- stub",
        lean_status=LeanStatus.OK,
        lean_stderr=None,
    )
    assert result.lean_status is LeanStatus.OK
    assert result.selected_goal == "x = y"


def test_load_benchmark_reads_expanded_schema(tmp_path: Path):
    payload = [
        {
            "id": "demo",
            "image": "img.png",
            "vlm_fixture": "fix.json",
            "category": "text-guided",
            "problem_text": "Prove x = y.",
            "objects": ["point A"],
            "diagram_marks": [],
            "gold_relations": [],
            "gold_assumptions": [],
            "gold_goal": "x = y",
            "acceptable_goals": [],
            "lean_theorem_name": "demo",
        }
    ]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload))
    examples = load_benchmark(path)
    assert len(examples) == 1
    assert examples[0].category == "text-guided"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_schema.py -q
```

Expected: failures because `DiagramMark`, `DiagramReading`, `PipelineResult`, `LeanStatus`, and the new `BenchmarkExample` fields do not exist yet.

- [ ] **Step 3: Rewrite `schema.py`**

Replace `src/diagram_theorem_assistant/schema.py` with:

```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
pytest tests/test_schema.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Run all tests to confirm no regressions**

Run:

```bash
pytest -q
```

Expected: everything still passes.

- [ ] **Step 6: Commit**

```bash
git add src/diagram_theorem_assistant/schema.py tests/test_schema.py && git commit -m "feat(schema): add DiagramMark, DiagramReading, PipelineResult, LeanStatus; extend BenchmarkExample"
```

---

### Task 3: Synthetic image generator, expanded benchmark, and VLM fixtures

**Files:**
- Create: `scripts/generate_synthetic.py`
- Create: `examples/images/synthetic/*.png` (5 files, generated)
- Create: `examples/vlm_fixtures/*.json` (5 files)
- Modify: `examples/benchmark.json`
- Modify: `.gitignore`

- [ ] **Step 1: Create the synthetic image generator**

Create `scripts/generate_synthetic.py`:

```python
"""Render deterministic matplotlib PNGs for benchmark examples.

Run: python scripts/generate_synthetic.py
Outputs: examples/images/synthetic/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "images" / "synthetic"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def isosceles_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.5, 0.9]), np.array([0.1, 0.2]), np.array([0.9, 0.2])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    # Equal-length tick marks on AB and AC
    for p1, p2 in [(a, b), (a, c)]:
        mid = (p1 + p2) / 2
        perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
        perp /= np.linalg.norm(perp) / 0.02
        ax.plot([mid[0] - perp[0], mid[0] + perp[0]], [mid[1] - perp[1], mid[1] + perp[1]], "k-", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "isosceles_triangle")


def parallel_lines() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0.1, 0.9], [0.7, 0.7], "k-", linewidth=2)
    ax.plot([0.1, 0.9], [0.3, 0.3], "k-", linewidth=2)
    ax.plot([0.2, 0.8], [0.1, 0.9], "k-", linewidth=2)
    ax.annotate("l", (0.9, 0.7), fontsize=14, xytext=(8, 0), textcoords="offset points")
    ax.annotate("m", (0.9, 0.3), fontsize=14, xytext=(8, 0), textcoords="offset points")
    ax.annotate("t", (0.8, 0.9), fontsize=14, xytext=(0, 8), textcoords="offset points")
    ax.annotate("1", (0.55, 0.73), fontsize=12)
    ax.annotate("2", (0.4, 0.27), fontsize=12)
    # Arrow marks denoting parallel lines
    for y in [0.7, 0.3]:
        ax.annotate("", xy=(0.55, y + 0.005), xytext=(0.45, y + 0.005),
                    arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _save(fig, "parallel_lines")


def ambiguous_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.5, 0.85]), np.array([0.15, 0.15]), np.array([0.8, 0.2])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "ambiguous_triangle")


def right_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.15, 0.85]), np.array([0.85, 0.15]), np.array([0.15, 0.15])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-12, 8), fontsize=14)
    # Right-angle box at C
    box = plt.Rectangle((0.15, 0.15), 0.05, 0.05, fill=False, edgecolor="black")
    ax.add_patch(box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "right_triangle")


def perpendicular_bisector() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, p = np.array([0.2, 0.3]), np.array([0.8, 0.3]), np.array([0.5, 0.85])
    ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=2)
    ax.plot([0.5, 0.5], [0.1, 0.95], "k--", linewidth=1.5)
    ax.plot([p[0], a[0]], [p[1], a[1]], "k-", linewidth=2)
    ax.plot([p[0], b[0]], [p[1], b[1]], "k-", linewidth=2)
    for point, label in [(a, "A"), (b, "B"), (p, "P")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    # Right-angle box at midpoint
    box = plt.Rectangle((0.5, 0.3), 0.04, 0.04, fill=False, edgecolor="black")
    ax.add_patch(box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "perpendicular_bisector")


def main() -> None:
    isosceles_triangle()
    parallel_lines()
    ambiguous_triangle()
    right_triangle()
    perpendicular_bisector()
    print(f"Wrote 5 synthetic PNGs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run:

```bash
cd "/Users/aadityab/Documents/New project" && source .venv/bin/activate && python scripts/generate_synthetic.py
```

Expected: `Wrote 5 synthetic PNGs to .../examples/images/synthetic` and 5 PNG files present.

Verify:

```bash
ls examples/images/synthetic/
```

Expected: `ambiguous_triangle.png isosceles_triangle.png parallel_lines.png perpendicular_bisector.png right_triangle.png`

- [ ] **Step 3: Hand-author VLM fixtures**

Create `examples/vlm_fixtures/isosceles_triangle_base_angles.json`:

```json
{
  "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
  "relations": ["triangle A B C"],
  "marks": [
    {"type": "equal_length", "segments": ["AB", "AC"]}
  ],
  "raw_vlm_output": {"source": "hand-authored-fixture"}
}
```

Create `examples/vlm_fixtures/parallel_lines_alternate_interior.json`:

```json
{
  "objects": ["line l", "line m", "line t", "angle 1", "angle 2"],
  "relations": ["parallel l m", "transversal t l m"],
  "marks": [
    {"type": "parallel", "lines": ["l", "m"]}
  ],
  "raw_vlm_output": {"source": "hand-authored-fixture"}
}
```

Create `examples/vlm_fixtures/ambiguous_triangle_no_text.json`:

```json
{
  "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
  "relations": ["triangle A B C"],
  "marks": [],
  "raw_vlm_output": {"source": "hand-authored-fixture"}
}
```

Create `examples/vlm_fixtures/right_triangle_pythagorean.json`:

```json
{
  "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
  "relations": ["triangle A B C", "perpendicular AC BC"],
  "marks": [
    {"type": "right_angle", "points": ["A", "C", "B"]}
  ],
  "raw_vlm_output": {"source": "hand-authored-fixture"}
}
```

Create `examples/vlm_fixtures/perpendicular_bisector_equidistant.json`:

```json
{
  "objects": ["point A", "point B", "point P", "line l", "segment AB", "segment PA", "segment PB"],
  "relations": ["collinear A midpoint_AB B"],
  "marks": [
    {"type": "perpendicular_bisector", "point": "P", "segment": "AB"}
  ],
  "raw_vlm_output": {"source": "hand-authored-fixture"}
}
```

- [ ] **Step 4: Rewrite `examples/benchmark.json`**

Replace the file content with:

```json
[
  {
    "id": "isosceles_triangle_base_angles",
    "image": "examples/images/synthetic/isosceles_triangle.png",
    "vlm_fixture": "examples/vlm_fixtures/isosceles_triangle_base_angles.json",
    "category": "text-guided",
    "problem_text": "In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
    "diagram_marks": [{"type": "equal_length", "segments": ["AB", "AC"]}],
    "gold_relations": ["triangle A B C"],
    "gold_assumptions": ["triangle A B C", "AB = AC"],
    "gold_goal": "angle ABC = angle BCA",
    "acceptable_goals": ["angle CBA = angle BCA"],
    "lean_theorem_name": "isosceles_base_angles"
  },
  {
    "id": "parallel_lines_alternate_interior",
    "image": "examples/images/synthetic/parallel_lines.png",
    "vlm_fixture": "examples/vlm_fixtures/parallel_lines_alternate_interior.json",
    "category": "text-guided",
    "problem_text": "Lines l and m are parallel and cut by transversal t. Prove angle 1 equals angle 2.",
    "objects": ["line l", "line m", "line t", "angle 1", "angle 2"],
    "diagram_marks": [{"type": "parallel", "lines": ["l", "m"]}],
    "gold_relations": ["parallel l m", "transversal t l m"],
    "gold_assumptions": ["parallel l m", "transversal t l m"],
    "gold_goal": "angle 1 = angle 2",
    "acceptable_goals": ["alternate interior angles are equal"],
    "lean_theorem_name": "alternate_interior_angles"
  },
  {
    "id": "ambiguous_triangle_no_text",
    "image": "examples/images/synthetic/ambiguous_triangle.png",
    "vlm_fixture": "examples/vlm_fixtures/ambiguous_triangle_no_text.json",
    "category": "diagram-only-ambiguous",
    "problem_text": "",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
    "diagram_marks": [],
    "gold_relations": ["triangle A B C"],
    "gold_assumptions": ["triangle A B C"],
    "gold_goal": "unknown without user selection",
    "acceptable_goals": ["AB = AC", "angle ABC = angle BCA", "area ABC > 0"],
    "lean_theorem_name": "ambiguous_triangle_user_selected_goal"
  },
  {
    "id": "right_triangle_pythagorean",
    "image": "examples/images/synthetic/right_triangle.png",
    "vlm_fixture": "examples/vlm_fixtures/right_triangle_pythagorean.json",
    "category": "text-guided",
    "problem_text": "Triangle ABC has a right angle at C. Prove AB^2 = AC^2 + BC^2.",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC", "right angle C"],
    "diagram_marks": [{"type": "right_angle", "points": ["A", "C", "B"]}],
    "gold_relations": ["triangle A B C", "perpendicular AC BC"],
    "gold_assumptions": ["triangle A B C", "right_angle A C B"],
    "gold_goal": "AB^2 = AC^2 + BC^2",
    "acceptable_goals": ["pythagorean theorem for triangle ABC"],
    "lean_theorem_name": "right_triangle_pythagorean"
  },
  {
    "id": "perpendicular_bisector_equidistant",
    "image": "examples/images/synthetic/perpendicular_bisector.png",
    "vlm_fixture": "examples/vlm_fixtures/perpendicular_bisector_equidistant.json",
    "category": "text-guided",
    "problem_text": "Point P lies on the perpendicular bisector of AB. Prove PA = PB.",
    "objects": ["point A", "point B", "point P", "line l", "segment AB", "segment PA", "segment PB"],
    "diagram_marks": [{"type": "perpendicular_bisector", "point": "P", "segment": "AB"}],
    "gold_relations": ["collinear A midpoint_AB B"],
    "gold_assumptions": ["P on perpendicular_bisector AB"],
    "gold_goal": "PA = PB",
    "acceptable_goals": ["distance P A = distance P B"],
    "lean_theorem_name": "perpendicular_bisector_equidistant"
  }
]
```

- [ ] **Step 5: Extend `.gitignore` for fetched datasets**

Append to `.gitignore`:

```
examples/images/geometry3k/
```

Use:

```bash
printf "examples/images/geometry3k/\n" >> .gitignore
```

- [ ] **Step 6: Run all tests, confirm green**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_synthetic.py examples/images/synthetic/ examples/vlm_fixtures/ examples/benchmark.json .gitignore && git commit -m "feat(benchmark): synthetic image generator, hand-authored VLM fixtures, expanded benchmark schema"
```

---

### Task 4: VLM base Protocol and `FixtureAdapter`

**Files:**
- Create: `src/diagram_theorem_assistant/vlm/__init__.py`
- Create: `src/diagram_theorem_assistant/vlm/base.py`
- Create: `src/diagram_theorem_assistant/vlm/fixture.py`
- Create: `tests/vlm/__init__.py`
- Create: `tests/vlm/test_fixture_adapter.py`

- [ ] **Step 1: Write failing tests for `FixtureAdapter`**

Create `tests/vlm/__init__.py` with:

```python
```

Create `tests/vlm/test_fixture_adapter.py`:

```python
from pathlib import Path

import pytest

from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter


def test_fixture_adapter_reads_hand_authored_fixture():
    adapter = FixtureAdapter(Path("examples/vlm_fixtures"))
    reading = adapter.read(Path("examples/images/synthetic/isosceles_triangle.png"),
                           fixture_path=Path("examples/vlm_fixtures/isosceles_triangle_base_angles.json"))
    assert "point A" in reading.objects
    assert any(mark.type == "equal_length" for mark in reading.marks)


def test_fixture_adapter_raises_when_fixture_missing(tmp_path: Path):
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(FixtureNotFoundError) as info:
        adapter.read(Path("anything.png"), fixture_path=tmp_path / "nope.json")
    assert "record_vlm_fixtures.py" in str(info.value)


def test_fixture_adapter_raises_vlm_error_on_malformed_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("this is not JSON")
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(VLMError):
        adapter.read(Path("img.png"), fixture_path=bad)
```

- [ ] **Step 2: Run tests to see them fail**

```bash
pytest tests/vlm/test_fixture_adapter.py -q
```

Expected: ImportError — `vlm` package does not exist.

- [ ] **Step 3: Implement `vlm/base.py` with the Protocol and exceptions**

Create `src/diagram_theorem_assistant/vlm/__init__.py`:

```python
"""Vision-language model adapters for diagram understanding."""
```

Create `src/diagram_theorem_assistant/vlm/base.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from diagram_theorem_assistant.schema import DiagramReading


class PipelineError(Exception):
    """Base for all pipeline-level failures."""


class VLMError(PipelineError):
    """Raised when a VLM adapter cannot produce a valid DiagramReading."""


class FixtureNotFoundError(PipelineError):
    """Raised when a benchmark expects a fixture JSON that does not exist."""


class DiagramUnderstander(Protocol):
    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading: ...
```

- [ ] **Step 4: Implement `vlm/fixture.py`**

Create `src/diagram_theorem_assistant/vlm/fixture.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError


class FixtureAdapter:
    """Deterministic VLM adapter backed by committed JSON fixtures."""

    def __init__(self, fixtures_root: Path) -> None:
        self.fixtures_root = fixtures_root

    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading:
        path = fixture_path or (self.fixtures_root / (image_path.stem + ".json"))
        if not path.exists():
            raise FixtureNotFoundError(
                f"No VLM fixture at {path}. "
                f"Record one with: python scripts/record_vlm_fixtures.py --image {image_path}"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VLMError(f"Fixture {path} is not valid JSON: {exc}") from exc
        return DiagramReading.from_dict(data)
```

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/vlm/test_fixture_adapter.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Run all tests**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/diagram_theorem_assistant/vlm/ tests/vlm/ && git commit -m "feat(vlm): add DiagramUnderstander Protocol and FixtureAdapter"
```

---

### Task 5: Gemini adapter (mocked tests, live test marker)

**Files:**
- Create: `src/diagram_theorem_assistant/vlm/prompts.py`
- Create: `src/diagram_theorem_assistant/vlm/gemini.py`
- Create: `tests/vlm/test_gemini_adapter.py`
- Create: `tests/test_gemini_live.py`

- [ ] **Step 1: Write failing tests for the Gemini adapter (all mocked)**

Create `tests/vlm/test_gemini_adapter.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter


def _response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _fake_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = [_response(text) for text in responses]
    return client


def test_gemini_adapter_sends_three_calls_and_merges(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        '{"objects": ["point A", "point B", "point C"]}',
        '{"relations": ["triangle A B C"]}',
        '{"marks": [{"type": "equal_length", "segments": ["AB", "AC"]}]}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)

    assert reading.objects == ["point A", "point B", "point C"]
    assert reading.relations == ["triangle A B C"]
    assert [mark.type for mark in reading.marks] == ["equal_length"]
    assert client.models.generate_content.call_count == 3


def test_gemini_adapter_retries_once_on_bad_json(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        "not JSON",
        '{"objects": ["point X"]}',
        '{"relations": []}',
        '{"marks": []}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)
    assert reading.objects == ["point X"]
    # 1 bad + 1 retry for stage 1, then stages 2 and 3 clean = 4 calls
    assert client.models.generate_content.call_count == 4


def test_gemini_adapter_raises_after_second_bad_json(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client(["still not JSON", "still not JSON"])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    with pytest.raises(VLMError):
        adapter.read(image)


def test_gemini_adapter_requires_api_key():
    with pytest.raises(VLMError):
        GeminiAdapter(api_key="", client=MagicMock(), model="gemini-test")


def test_gemini_adapter_defaults_missing_fields(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        '{"objects": ["point A"], "ignored": "extra"}',
        '{}',
        '{"marks": []}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)
    assert reading.objects == ["point A"]
    assert reading.relations == []
    assert reading.marks == []
```

Create `tests/test_gemini_live.py`:

```python
import os
from pathlib import Path

import pytest

from diagram_theorem_assistant.vlm.gemini import GeminiAdapter


@pytest.mark.live
def test_live_gemini_reads_synthetic_isosceles_triangle():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    image = Path("examples/images/synthetic/isosceles_triangle.png")
    assert image.exists(), "Run scripts/generate_synthetic.py first"

    adapter = GeminiAdapter(api_key=api_key)
    reading = adapter.read(image)

    # Minimal structural assertions — exact outputs vary by model
    assert isinstance(reading.objects, list)
    assert len(reading.objects) >= 3
```

- [ ] **Step 2: Run the mocked tests to see them fail**

```bash
pytest tests/vlm/test_gemini_adapter.py -q
```

Expected: ImportError — `gemini.py` and `prompts.py` do not exist.

- [ ] **Step 3: Create `vlm/prompts.py`**

Create `src/diagram_theorem_assistant/vlm/prompts.py`:

```python
"""Prompt templates for the three-stage multi-step VLM extraction.

Each call to Gemini is narrowly scoped to reduce hallucination: the model
answers one focused question at a time and returns JSON in a known shape.
"""
from __future__ import annotations

OBJECTS_PROMPT = """You are analyzing a geometry diagram.

List every NAMED geometric object you can see. Use lowercase kind followed
by the capitalized name. Examples: "point A", "segment AB", "line l",
"circle O", "angle 1".

Do NOT invent objects that are not labeled. If a point has no visible
label, do not include it.

Respond with ONLY valid JSON of the form:
{"objects": ["point A", "segment AB", ...]}
"""

RELATIONS_PROMPT = """You are analyzing a geometry diagram.

The following named objects have been identified:
{objects}

List the GEOMETRIC RELATIONS directly visible in the diagram. Use these
formats:
- "triangle A B C"                  (three points form a triangle)
- "collinear A B C"                 (three points on a line)
- "parallel l m"                    (two lines are parallel)
- "perpendicular AB AC"             (two segments/lines are perpendicular)
- "transversal t l m"               (t cuts lines l and m)
- "incidence P AB"                  (point P lies on segment AB)

Do NOT infer relations that are not visually indicated. If unsure, omit.

Respond with ONLY valid JSON of the form:
{"relations": ["triangle A B C", ...]}
"""

MARKS_PROMPT = """You are analyzing a geometry diagram.

List the VISUAL MARKS drawn on the diagram (tick marks, arrows, right-angle
boxes, etc). Use these shapes:

- {"type": "right_angle", "points": ["A", "C", "B"]}       (right-angle box at C)
- {"type": "equal_length", "segments": ["AB", "AC"]}       (tick marks showing equal segments)
- {"type": "parallel", "lines": ["l", "m"]}                (parallel arrows)
- {"type": "perpendicular_bisector", "point": "P", "segment": "AB"}

Only include marks that are actually drawn. Do not infer marks from object
positions.

Respond with ONLY valid JSON of the form:
{"marks": [...]}
"""
```

- [ ] **Step 4: Create `vlm/gemini.py`**

Create `src/diagram_theorem_assistant/vlm/gemini.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from diagram_theorem_assistant.schema import DiagramMark, DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError
from diagram_theorem_assistant.vlm.prompts import MARKS_PROMPT, OBJECTS_PROMPT, RELATIONS_PROMPT


def _default_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


class GeminiAdapter:
    """Real VLM adapter using Google Gemini via google-genai SDK.

    Makes three sequential calls per image (objects, relations, marks).
    Each response is parsed as JSON; one retry on malformed JSON with a
    stricter follow-up prompt.
    """

    def __init__(self, api_key: str, *, client: Any | None = None, model: str | None = None) -> None:
        if not api_key:
            raise VLMError(
                "GEMINI_API_KEY is empty. Set it in .env or pass api_key explicitly."
            )
        if client is None:
            from google import genai  # lazy import so tests can mock without SDK installed
            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading:
        image_bytes = Path(image_path).read_bytes()

        objects_json = self._call(OBJECTS_PROMPT, image_bytes, required_key="objects")
        relations_json = self._call(
            RELATIONS_PROMPT.format(objects=json.dumps(objects_json.get("objects", []))),
            image_bytes,
            required_key="relations",
        )
        marks_json = self._call(MARKS_PROMPT, image_bytes, required_key="marks")

        return DiagramReading(
            objects=list(objects_json.get("objects", [])),
            relations=list(relations_json.get("relations", [])),
            marks=[DiagramMark.from_dict(m) for m in marks_json.get("marks", [])],
            raw_vlm_output={
                "model": self._model,
                "objects_raw": objects_json,
                "relations_raw": relations_json,
                "marks_raw": marks_json,
            },
        )

    def _call(self, prompt: str, image_bytes: bytes, *, required_key: str) -> dict[str, Any]:
        response = self._invoke(prompt, image_bytes)
        parsed = _try_parse_json(response)
        if parsed is None:
            response = self._invoke(
                prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON, no prose.",
                image_bytes,
            )
            parsed = _try_parse_json(response)
            if parsed is None:
                raise VLMError(
                    f"Gemini returned non-JSON after retry. Raw output:\n{response}"
                )
        if required_key not in parsed:
            parsed[required_key] = []
        return parsed

    def _invoke(self, prompt: str, image_bytes: bytes) -> str:
        from google.genai import types  # lazy, matches real SDK
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
        except Exception:  # noqa: BLE001 — wrap any SDK/network error
            # One automatic retry with the same prompt for transient network issues
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        prompt,
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                raise VLMError(f"Gemini call failed: {exc}") from exc
        return response.text or ""


def _try_parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Handle ```json\n...\n``` fences
        lines = stripped.splitlines()
        inner = "\n".join(line for line in lines if not line.startswith("```"))
        stripped = inner
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None
```

Note: the test file uses `MagicMock` objects with `client.models.generate_content` returning responses that have `.text`. The adapter's lazy `from google.genai import types` will fail when running under pytest if `google-genai` is not installed — but since we installed it in Task 1, this works. For the mocked tests, `types.Part.from_bytes` is also called; we need the import to succeed. That's fine because the SDK is installed.

- [ ] **Step 5: Run tests and verify they pass**

```bash
pytest tests/vlm/test_gemini_adapter.py -q
```

Expected: `5 passed`.

- [ ] **Step 6: Run live test in opt-in mode (optional)**

```bash
pytest tests/test_gemini_live.py -q -m live
```

Expected: either passes (if API key valid and model reachable) or fails with a clear message you can triage. Default `pytest` runs skip this.

- [ ] **Step 7: Run all default tests**

```bash
pytest -q
```

Expected: all green, live test skipped.

- [ ] **Step 8: Commit**

```bash
git add src/diagram_theorem_assistant/vlm/ tests/vlm/test_gemini_adapter.py tests/test_gemini_live.py && git commit -m "feat(vlm): add Gemini adapter with multi-step prompting and retry"
```

---

### Task 6: Scaffold the `lean_project/` Lake project

**Files:**
- Create: `lean_project/lakefile.toml`
- Create: `lean_project/lean-toolchain`
- Create: `lean_project/DiagramTheorems.lean`
- Create: `lean_project/DiagramTheorems/Basic.lean`

- [ ] **Step 1: Create the toolchain file**

Create `lean_project/lean-toolchain`:

```
leanprover/lean4:v4.15.0
```

We pin to a mathlib-compatible release rather than using our locally-installed `4.30.0-rc2` because mathlib releases lag Lean slightly. `elan` downloads the correct toolchain automatically when `lake` runs in this directory.

- [ ] **Step 2: Create the `lakefile.toml`**

Create `lean_project/lakefile.toml`:

```toml
name = "DiagramTheorems"
defaultTargets = ["DiagramTheorems"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
version = "git#v4.15.0"

[[lean_lib]]
name = "DiagramTheorems"
```

- [ ] **Step 3: Create the top-level library file**

Create `lean_project/DiagramTheorems.lean`:

```lean
import DiagramTheorems.Basic
```

- [ ] **Step 4: Create the `Basic.lean` helpers file**

Create `lean_project/DiagramTheorems/Basic.lean`:

```lean
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Tactic

/-- Placeholder namespace for theorems generated by the diagram-to-Lean pipeline. -/
namespace DiagramTheorems

/-- Trivial sentinel used to confirm the Lake project builds cleanly. -/
theorem ok : True := by trivial

end DiagramTheorems
```

- [ ] **Step 5: Fetch the mathlib cache (~1.5 GB download, ~5-15 min)**

Run:

```bash
cd "/Users/aadityab/Documents/New project/lean_project" && lake update && lake exe cache get
```

Expected: downloads prebuilt mathlib olean files. No compilation happens.

If `lake update` complains about a missing manifest, that's normal on first run — it creates one.

- [ ] **Step 6: Verify the project builds**

```bash
cd "/Users/aadityab/Documents/New project/lean_project" && lake build DiagramTheorems.Basic
```

Expected: builds `Basic.lean` successfully (prints nothing or a short progress summary; exits 0).

- [ ] **Step 7: Commit**

```bash
cd "/Users/aadityab/Documents/New project" && git add lean_project/lakefile.toml lean_project/lean-toolchain lean_project/DiagramTheorems.lean lean_project/DiagramTheorems/Basic.lean && git commit -m "feat(lean): scaffold Lake project with mathlib dependency"
```

Note: `.lake/`, `lake-packages/`, and `build/` are already gitignored from the initial commit.

---

### Task 7: Rewrite `lean_export.py` to emit real mathlib geometry

**Files:**
- Modify: `src/diagram_theorem_assistant/lean_export.py`
- Rewrite: `tests/test_lean_export.py`

- [ ] **Step 1: Rewrite the failing tests**

Replace `tests/test_lean_export.py` with:

```python
from diagram_theorem_assistant.lean_export import theorem_from


def test_theorem_from_imports_basic_module_and_uses_namespace():
    source = theorem_from(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )
    assert source.startswith("import DiagramTheorems.Basic")
    assert "namespace DiagramTheorems.Generated" in source
    assert "theorem isosceles_base_angles" in source


def test_theorem_from_emits_assumption_and_goal_comments():
    source = theorem_from(
        name="demo",
        assumptions=["triangle A B C"],
        goal="area ABC > 0",
    )
    assert "-- assumption: triangle A B C" in source
    assert "-- goal: area ABC > 0" in source


def test_theorem_from_uses_sorry_for_unmapped_goals():
    source = theorem_from(
        name="demo",
        assumptions=[],
        goal="some unrecognized string",
    )
    assert ":= by" in source
    assert "sorry" in source


def test_theorem_from_escapes_comment_newlines():
    source = theorem_from(
        name="demo",
        assumptions=["a\nb"],
        goal="c",
    )
    assert "\\n" in source or "a b" in source  # either escaped or flattened
    assert "-- assumption:" in source


def test_theorem_from_sanitizes_identifier():
    source = theorem_from(name="bad name with spaces!", assumptions=[], goal="x")
    assert "theorem bad_name_with_spaces_" in source


def test_theorem_from_uses_true_placeholder_when_goal_is_empty():
    source = theorem_from(name="demo", assumptions=[], goal="")
    assert "theorem demo : True" in source
    assert ":= by trivial" in source
```

- [ ] **Step 2: Run tests to see them fail**

```bash
pytest tests/test_lean_export.py -q
```

Expected: failures — `theorem_from` does not exist (old function was `theorem_skeleton`).

- [ ] **Step 3: Rewrite `lean_export.py`**

Replace `src/diagram_theorem_assistant/lean_export.py` with:

```python
from __future__ import annotations

import re


def theorem_from(name: str, assumptions: list[str], goal: str) -> str:
    """Emit Lean 4 source for a generated theorem.

    The current scope is statement-level validity: imports resolve, the
    theorem declaration type-checks, and the proof is `sorry` or `trivial`.
    Future work can replace `True` / `sorry` with real Mathlib propositions
    as the formalization mapping matures.
    """
    identifier = _lean_identifier(name)
    assumption_lines = "\n".join(f"-- assumption: {_flatten(a)}" for a in assumptions)
    goal_line = f"-- goal: {_flatten(goal)}"

    if not goal.strip():
        body = f"theorem {identifier} : True := by trivial"
    else:
        body = f"theorem {identifier} : True := by\n  -- target: {_flatten(goal)}\n  sorry"

    return (
        "import DiagramTheorems.Basic\n"
        "\n"
        "namespace DiagramTheorems.Generated\n"
        "\n"
        f"{assumption_lines}\n"
        f"{goal_line}\n"
        f"{body}\n"
        "\n"
        "end DiagramTheorems.Generated\n"
    )


def _lean_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", name)
    if not cleaned or cleaned[0].isdigit():
        return "generated_" + cleaned
    return cleaned


def _flatten(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()
```

Note: we emit `theorem X : True := by sorry` even when a goal is recognized. This is deliberate — our spec scope is *statement-level* validity. Mapping arbitrary geometry goals to real mathlib propositions is out of scope; `sorry` lets the file still type-check.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_lean_export.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Run all tests**

```bash
pytest -q
```

Expected: the old `theorem_skeleton` callers now break. Specifically, `demo.py` imports `theorem_skeleton`. We'll fix `demo.py` in Task 13 — but we need `demo.py` to keep working now. Patch it with a temporary shim at the bottom of `lean_export.py`:

```python
# Temporary compatibility shim (removed in Task 13 when demo.py is rewritten)
def theorem_skeleton(name: str, assumptions: list[str], goal: str) -> str:
    return theorem_from(name=name, assumptions=assumptions, goal=goal)
```

Re-run:

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/diagram_theorem_assistant/lean_export.py tests/test_lean_export.py && git commit -m "feat(lean): emit real mathlib-imported Lean source from theorem_from"
```

---

### Task 8: Extend `assumption_extraction.py` to consume relations

**Files:**
- Modify: `src/diagram_theorem_assistant/assumption_extraction.py`
- Modify: `tests/test_assumption_extraction.py`

- [ ] **Step 1: Add failing tests for relation-based assumption extraction**

Append to `tests/test_assumption_extraction.py`:

```python
from diagram_theorem_assistant.assumption_extraction import (
    assumptions_from_relations,
    infer_assumptions,
)


def test_assumptions_from_relations_handles_known_relations():
    relations = [
        "triangle A B C",
        "perpendicular AC BC",
        "parallel l m",
        "collinear A B C",
    ]
    assumptions = assumptions_from_relations(relations)
    assert "triangle A B C" in assumptions
    assert "parallel l m" in assumptions
    assert "right_angle A C B" in assumptions  # from perpendicular at C
    assert "collinear A B C" in assumptions


def test_infer_assumptions_accepts_relations_argument():
    result = infer_assumptions(
        problem_text="",
        objects=["point A", "point B", "point C"],
        diagram_marks=[],
        confirmed_assumptions=None,
        relations=["triangle A B C"],
    )
    assert "triangle A B C" in result
```

- [ ] **Step 2: Run tests to see failures**

```bash
pytest tests/test_assumption_extraction.py -q
```

Expected: ImportError for `assumptions_from_relations`, and `infer_assumptions` TypeError for unexpected keyword `relations`.

- [ ] **Step 3: Extend `assumption_extraction.py`**

Replace `infer_assumptions` and add `assumptions_from_relations` in `src/diagram_theorem_assistant/assumption_extraction.py`. Apply the following edit:

Replace the existing `infer_assumptions` function with:

```python
def infer_assumptions(
    problem_text: str,
    objects: list[str],
    diagram_marks: list[dict[str, Any]],
    confirmed_assumptions: list[str] | None = None,
    relations: list[str] | None = None,
) -> list[str]:
    return merge_assumptions(
        confirmed_assumptions or [],
        assumptions_from_relations(relations or []),
        assumptions_from_objects(objects),
        assumptions_from_marks(diagram_marks),
        assumptions_from_text(problem_text),
    )
```

Add at the end of the file, before the helpers:

```python
def assumptions_from_relations(relations: list[str]) -> list[str]:
    assumptions: list[str] = []
    for relation in relations:
        tokens = relation.split()
        if not tokens:
            continue
        head = tokens[0]
        if head == "triangle" and len(tokens) == 4:
            assumptions.append(relation)
        elif head == "parallel" and len(tokens) == 3:
            assumptions.append(relation)
        elif head == "collinear" and len(tokens) == 4:
            assumptions.append(relation)
        elif head == "perpendicular" and len(tokens) == 3:
            # "perpendicular AC BC" → right angle at the shared vertex
            assumptions.append(relation)
            assumptions.extend(_right_angle_from_perpendicular(tokens[1], tokens[2]))
        elif head == "incidence" and len(tokens) == 3:
            assumptions.append(relation)
        elif head == "transversal" and len(tokens) == 4:
            assumptions.append(relation)
    return merge_assumptions(assumptions)


def _right_angle_from_perpendicular(seg_a: str, seg_b: str) -> list[str]:
    if len(seg_a) == 2 and len(seg_b) == 2:
        shared = set(seg_a) & set(seg_b)
        if len(shared) == 1:
            vertex = next(iter(shared))
            other_a = (set(seg_a) - {vertex}).pop()
            other_b = (set(seg_b) - {vertex}).pop()
            return [f"right_angle {other_a} {vertex} {other_b}"]
    return []
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_assumption_extraction.py -q
```

Expected: all tests pass including the two new ones.

- [ ] **Step 5: Run all tests**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/diagram_theorem_assistant/assumption_extraction.py tests/test_assumption_extraction.py && git commit -m "feat(extraction): consume VLM relations when inferring assumptions"
```

---

### Task 9: Create `lean_runner.py` with subprocess wrapper

**Files:**
- Create: `src/diagram_theorem_assistant/lean_runner.py`
- Create: `tests/test_lean_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_lean_runner.py`:

```python
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.base import PipelineError


def test_runner_reports_unavailable_when_lake_missing(tmp_path: Path):
    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value=None):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "Generated.lean")
        status, stderr = runner.typecheck("theorem demo : True := by trivial")
    assert status is LeanStatus.UNAVAILABLE
    assert stderr is None


def test_runner_writes_file_and_parses_success(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    generated = project / "DiagramTheorems" / "Generated.lean"

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", return_value=_FakeCompleted()):
        runner = LeanRunner(project_root=project, generated_path=generated)
        status, stderr = runner.typecheck("theorem demo : True := by trivial")

    assert generated.exists()
    assert status is LeanStatus.OK
    assert stderr == ""


def test_runner_parses_type_error(tmp_path: Path):
    class _FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "Generated.lean:5:0: error: unknown identifier 'foo'"

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", return_value=_FakeCompleted()):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean")
        status, stderr = runner.typecheck("bogus source")

    assert status is LeanStatus.TYPE_ERROR
    assert "unknown identifier" in stderr


def test_runner_handles_timeout(tmp_path: Path):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["lake", "build"], timeout=1)

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", side_effect=_raise):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean", timeout_sec=1)
        status, stderr = runner.typecheck("slow")
    assert status is LeanStatus.TYPE_ERROR
    assert "timed out" in stderr


def test_runner_raises_on_unexpected_failure(tmp_path: Path):
    def _raise(*args, **kwargs):
        raise OSError("lake crashed hard")

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", side_effect=_raise):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean")
        with pytest.raises(PipelineError):
            runner.typecheck("src")


@pytest.mark.lean
def test_runner_builds_trivial_theorem_with_real_lake():
    project_root = Path(__file__).resolve().parent.parent / "lean_project"
    generated = project_root / "DiagramTheorems" / "Generated.lean"
    runner = LeanRunner(project_root=project_root, generated_path=generated)
    source = (
        "import DiagramTheorems.Basic\n"
        "namespace DiagramTheorems.Generated\n"
        "theorem smoke : True := by trivial\n"
        "end DiagramTheorems.Generated\n"
    )
    status, stderr = runner.typecheck(source)
    assert status is LeanStatus.OK, f"lake build failed: {stderr}"
```

- [ ] **Step 2: Run mocked tests to see failures**

```bash
pytest tests/test_lean_runner.py -q -m "not lean"
```

Expected: ImportError — `lean_runner.py` does not exist.

- [ ] **Step 3: Create `lean_runner.py`**

Create `src/diagram_theorem_assistant/lean_runner.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.base import LeanError


class LeanRunner:
    """Run `lake build` against generated Lean source.

    Writes `lean_source` to `generated_path`, invokes `lake build` scoped
    to the generated module, and returns (status, stderr). Gracefully
    degrades to LeanStatus.UNAVAILABLE if `lake` is not on PATH.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        generated_path: Path,
        timeout_sec: int = 60,
        build_target: str = "DiagramTheorems.Generated",
    ) -> None:
        self.project_root = project_root
        self.generated_path = generated_path
        self.timeout_sec = timeout_sec
        self.build_target = build_target

    def typecheck(self, lean_source: str) -> tuple[LeanStatus, str | None]:
        lake = shutil.which("lake")
        if lake is None:
            return LeanStatus.UNAVAILABLE, None

        self.generated_path.parent.mkdir(parents=True, exist_ok=True)
        self.generated_path.write_text(lean_source, encoding="utf-8")

        try:
            completed = subprocess.run(  # noqa: S603 — controlled args
                [lake, "build", self.build_target],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return LeanStatus.TYPE_ERROR, "lake build timed out"
        except (OSError, subprocess.SubprocessError) as exc:
            raise LeanError(f"lake invocation failed: {exc}") from exc

        if completed.returncode == 0:
            return LeanStatus.OK, completed.stderr.strip()
        return LeanStatus.TYPE_ERROR, completed.stderr.strip()
```

- [ ] **Step 4: Run mocked tests**

```bash
pytest tests/test_lean_runner.py -q -m "not lean"
```

Expected: `5 passed` (the five mocked tests).

- [ ] **Step 5: Run the real-lake smoke test (requires mathlib cache from Task 6)**

```bash
pytest tests/test_lean_runner.py -q -m lean
```

Expected: one test passes. Takes 30-60 seconds depending on how much lake has cached.

- [ ] **Step 6: Run full default test suite**

```bash
pytest -q
```

Expected: all green; `lean` marker tests skipped by default.

- [ ] **Step 7: Commit**

```bash
git add src/diagram_theorem_assistant/lean_runner.py tests/test_lean_runner.py && git commit -m "feat(lean): add subprocess-based LeanRunner with graceful degradation"
```

---

### Task 10: Create pipeline orchestrator

**Files:**
- Create: `src/diagram_theorem_assistant/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.pipeline import run_pipeline
from diagram_theorem_assistant.schema import DiagramMark, DiagramReading, LeanStatus
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter

FIXTURES = Path("examples/vlm_fixtures")
IMAGES = Path("examples/images/synthetic")


@pytest.fixture()
def fake_lean_runner_ok():
    runner = MagicMock()
    runner.typecheck.return_value = (LeanStatus.OK, "")
    return runner


def test_pipeline_runs_isosceles_end_to_end(fake_lean_runner_ok):
    adapter = FixtureAdapter(FIXTURES)
    result = run_pipeline(
        image_path=IMAGES / "isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        confirmed_assumptions=None,
        vlm=adapter,
        vlm_fixture_path=FIXTURES / "isosceles_triangle_base_angles.json",
        lean_runner=fake_lean_runner_ok,
        theorem_name="isosceles_base_angles",
    )
    assert "triangle A B C" in result.confirmed_assumptions
    assert "AB = AC" in result.confirmed_assumptions
    assert result.selected_goal == "angle ABC = angle BCA"
    assert result.lean_status is LeanStatus.OK


def test_pipeline_diagram_only_returns_ranked_candidates(fake_lean_runner_ok):
    adapter = FixtureAdapter(FIXTURES)
    result = run_pipeline(
        image_path=IMAGES / "ambiguous_triangle.png",
        problem_text="",
        confirmed_assumptions=["triangle A B C"],
        vlm=adapter,
        vlm_fixture_path=FIXTURES / "ambiguous_triangle_no_text.json",
        lean_runner=fake_lean_runner_ok,
        theorem_name="ambiguous",
    )
    assert len(result.goal_candidates) >= 2
    assert result.selected_goal == result.goal_candidates[0]


def test_pipeline_surfaces_lean_unavailable_status():
    runner = MagicMock()
    runner.typecheck.return_value = (LeanStatus.UNAVAILABLE, None)
    adapter = FixtureAdapter(FIXTURES)
    result = run_pipeline(
        image_path=IMAGES / "right_triangle.png",
        problem_text="Triangle ABC has a right angle at C. Prove AB^2 = AC^2 + BC^2.",
        confirmed_assumptions=None,
        vlm=adapter,
        vlm_fixture_path=FIXTURES / "right_triangle_pythagorean.json",
        lean_runner=runner,
        theorem_name="pyth",
    )
    assert result.lean_status is LeanStatus.UNAVAILABLE
    assert result.lean_source.startswith("import DiagramTheorems.Basic")


def test_pipeline_returns_reading_with_vlm_marks(fake_lean_runner_ok):
    adapter = FixtureAdapter(FIXTURES)
    result = run_pipeline(
        image_path=IMAGES / "right_triangle.png",
        problem_text="Triangle ABC has a right angle at C. Prove AB^2 = AC^2 + BC^2.",
        confirmed_assumptions=None,
        vlm=adapter,
        vlm_fixture_path=FIXTURES / "right_triangle_pythagorean.json",
        lean_runner=fake_lean_runner_ok,
        theorem_name="pyth",
    )
    assert any(isinstance(m, DiagramMark) and m.type == "right_angle" for m in result.reading.marks)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_pipeline.py -q
```

Expected: ImportError — `pipeline.py` missing.

- [ ] **Step 3: Create `pipeline.py`**

Create `src/diagram_theorem_assistant/pipeline.py`:

```python
from __future__ import annotations

from pathlib import Path

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_from
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus, PipelineResult
from diagram_theorem_assistant.vlm.base import DiagramUnderstander


def run_pipeline(
    *,
    image_path: Path,
    problem_text: str,
    confirmed_assumptions: list[str] | None,
    vlm: DiagramUnderstander,
    vlm_fixture_path: Path | None = None,
    lean_runner: LeanRunner | None = None,
    theorem_name: str = "generated_theorem",
) -> PipelineResult:
    reading = vlm.read(image_path, fixture_path=vlm_fixture_path)

    candidate = infer_assumptions(
        problem_text=problem_text,
        objects=reading.objects,
        diagram_marks=[mark.to_dict() for mark in reading.marks],
        confirmed_assumptions=None,
        relations=reading.relations,
    )

    confirmed = confirmed_assumptions if confirmed_assumptions is not None else candidate

    extracted = extract_goal_from_text(problem_text)
    ranked = rank_goal_candidates(reading.objects, confirmed)
    goal_candidates = [extracted, *ranked] if extracted else ranked

    selected = extracted or (ranked[0] if ranked else "")

    lean_source = theorem_from(name=theorem_name, assumptions=confirmed, goal=selected)

    if lean_runner is not None:
        status, stderr = lean_runner.typecheck(lean_source)
    else:
        status, stderr = LeanStatus.UNAVAILABLE, None

    return PipelineResult(
        reading=reading,
        candidate_assumptions=candidate,
        confirmed_assumptions=confirmed,
        goal_candidates=goal_candidates,
        selected_goal=selected,
        lean_source=lean_source,
        lean_status=status,
        lean_stderr=stderr,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Run everything**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/diagram_theorem_assistant/pipeline.py tests/test_pipeline.py && git commit -m "feat(pipeline): orchestrate VLM → assumptions → goals → Lean"
```

---

### Task 11: Rename `evaluation.py` → `metrics.py` and add Lean validity + category breakdown

**Files:**
- Create: `src/diagram_theorem_assistant/metrics.py`
- Delete: `src/diagram_theorem_assistant/evaluation.py`
- Create: `tests/test_metrics.py`
- Delete: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for the new module**

Create `tests/test_metrics.py`:

```python
from diagram_theorem_assistant.metrics import (
    assumption_precision,
    assumption_recall,
    category_breakdown,
    lean_validity_rate,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.schema import (
    DiagramMark,
    DiagramReading,
    LeanStatus,
    PipelineResult,
)


def _result(status: LeanStatus, goal_candidates: list[str], selected: str = "") -> PipelineResult:
    return PipelineResult(
        reading=DiagramReading(objects=[], relations=[], marks=[]),
        candidate_assumptions=[],
        confirmed_assumptions=[],
        goal_candidates=goal_candidates,
        selected_goal=selected,
        lean_source="",
        lean_status=status,
        lean_stderr=None,
    )


def test_assumption_precision_and_recall_round_trip():
    predicted = ["a", "b", "c"]
    gold = ["a", "b", "d"]
    assert assumption_precision(predicted, gold) == 2 / 3
    assert assumption_recall(predicted, gold) == 2 / 3


def test_top_k_goal_accuracy_counts_position():
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=1) == 0.0
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=2) == 1.0
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=5) == 1.0


def test_lean_validity_rate_counts_ok_only():
    results = [
        _result(LeanStatus.OK, ["a"], "a"),
        _result(LeanStatus.TYPE_ERROR, ["a"], "a"),
        _result(LeanStatus.UNAVAILABLE, ["a"], "a"),
        _result(LeanStatus.OK, ["a"], "a"),
    ]
    assert lean_validity_rate(results) == 2 / 4


def test_lean_validity_rate_returns_zero_on_empty():
    assert lean_validity_rate([]) == 0.0


def test_category_breakdown_groups_and_aggregates():
    results_by_category = {
        "text-guided": [_result(LeanStatus.OK, ["x"], "x"), _result(LeanStatus.TYPE_ERROR, ["x"], "x")],
        "diagram-only-ambiguous": [_result(LeanStatus.OK, ["y"], "y")],
    }
    breakdown = category_breakdown(results_by_category)
    assert breakdown["text-guided"]["lean_validity_rate"] == 0.5
    assert breakdown["diagram-only-ambiguous"]["lean_validity_rate"] == 1.0
    assert breakdown["text-guided"]["count"] == 2
```

- [ ] **Step 2: Run, verify failure**

```bash
pytest tests/test_metrics.py -q
```

Expected: ImportError.

- [ ] **Step 3: Create `metrics.py`**

Create `src/diagram_theorem_assistant/metrics.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

from diagram_theorem_assistant.schema import LeanStatus, PipelineResult


def assumption_recall(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return len(set(predicted) & set(gold)) / len(set(gold))


def assumption_precision(predicted: list[str], gold: list[str]) -> float:
    if not predicted:
        return 1.0
    return len(set(predicted) & set(gold)) / len(set(predicted))


def top_k_goal_accuracy(ranked_goals: list[str], gold_goal: str, k: int) -> float:
    return 1.0 if gold_goal in ranked_goals[:k] else 0.0


def lean_validity_rate(results: Iterable[PipelineResult]) -> float:
    results_list = list(results)
    if not results_list:
        return 0.0
    ok = sum(1 for r in results_list if r.lean_status is LeanStatus.OK)
    return ok / len(results_list)


def category_breakdown(results_by_category: dict[str, list[PipelineResult]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for category, results in results_by_category.items():
        summary[category] = {
            "count": float(len(results)),
            "lean_validity_rate": lean_validity_rate(results),
        }
    return summary
```

- [ ] **Step 4: Delete old evaluation module and its test file**

```bash
git rm src/diagram_theorem_assistant/evaluation.py tests/test_evaluation.py
```

- [ ] **Step 5: Run all tests**

```bash
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/diagram_theorem_assistant/metrics.py tests/test_metrics.py && git commit -m "feat(metrics): rename from evaluation, add Lean validity rate and category breakdown"
```

---

### Task 12: Update `demo.py` to use the new pipeline

**Files:**
- Modify: `src/diagram_theorem_assistant/demo.py`
- Modify: `tests/test_demo.py`
- Modify: `src/diagram_theorem_assistant/lean_export.py` (remove shim)

- [ ] **Step 1: Rewrite `tests/test_demo.py`**

Replace `tests/test_demo.py` with:

```python
from pathlib import Path
from unittest.mock import patch

from diagram_theorem_assistant.demo import run_demo
from diagram_theorem_assistant.schema import LeanStatus


def test_run_demo_returns_isosceles_result():
    with patch("diagram_theorem_assistant.demo._lean_runner", return_value=None):
        result = run_demo()
    assert result["selected_goal"] == "angle ABC = angle BCA"
    assert "triangle A B C" in result["confirmed_assumptions"]
    assert "AB = AC" in result["confirmed_assumptions"]
    assert result["lean_status"] == LeanStatus.UNAVAILABLE.value
    assert result["lean_source"].startswith("import DiagramTheorems.Basic")


def test_run_demo_uses_real_lean_runner_when_lake_present(tmp_path: Path):
    from diagram_theorem_assistant.demo import run_demo
    with patch("diagram_theorem_assistant.demo._lean_runner") as mocked:
        fake = mocked.return_value
        fake.typecheck.return_value = (LeanStatus.OK, "")
        result = run_demo()
    assert result["lean_status"] == LeanStatus.OK.value
```

- [ ] **Step 2: Run tests to see them fail**

```bash
pytest tests/test_demo.py -q
```

Expected: failures (old `run_demo` signature, old return keys).

- [ ] **Step 3: Rewrite `demo.py`**

Replace `src/diagram_theorem_assistant/demo.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.pipeline import run_pipeline
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _lean_runner() -> LeanRunner | None:
    project_root = REPO_ROOT / "lean_project"
    if not project_root.exists():
        return None
    return LeanRunner(
        project_root=project_root,
        generated_path=project_root / "DiagramTheorems" / "Generated.lean",
    )


def run_demo() -> dict[str, Any]:
    adapter = FixtureAdapter(REPO_ROOT / "examples" / "vlm_fixtures")
    runner = _lean_runner()
    result = run_pipeline(
        image_path=REPO_ROOT / "examples" / "images" / "synthetic" / "isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        confirmed_assumptions=None,
        vlm=adapter,
        vlm_fixture_path=REPO_ROOT / "examples" / "vlm_fixtures" / "isosceles_triangle_base_angles.json",
        lean_runner=runner,
        theorem_name="isosceles_base_angles",
    )
    return {
        "candidate_assumptions": result.candidate_assumptions,
        "confirmed_assumptions": result.confirmed_assumptions,
        "goal_candidates": result.goal_candidates,
        "selected_goal": result.selected_goal,
        "lean_status": result.lean_status.value,
        "lean_source": result.lean_source,
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Remove the compatibility shim from `lean_export.py`**

Delete the `theorem_skeleton` shim at the bottom of `src/diagram_theorem_assistant/lean_export.py` (added in Task 7). The file should end at the `_flatten` function.

- [ ] **Step 5: Run all tests**

```bash
pytest -q
```

Expected: all tests green, including the two new demo tests.

- [ ] **Step 6: Manually run the demo script**

```bash
python -m diagram_theorem_assistant.demo
```

Expected: JSON printout showing confirmed assumptions, selected goal, and Lean source.

- [ ] **Step 7: Commit**

```bash
git add src/diagram_theorem_assistant/demo.py src/diagram_theorem_assistant/lean_export.py tests/test_demo.py && git commit -m "feat(demo): wire demo to new pipeline + optional LeanRunner"
```

---

### Task 13: Streamlit UI with interactive and benchmark pages

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py`**

Create `app.py` at the repo root:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_from
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.metrics import (
    assumption_precision,
    assumption_recall,
    category_breakdown,
    lean_validity_rate,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.pipeline import run_pipeline
from diagram_theorem_assistant.schema import LeanStatus, load_benchmark
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO_ROOT / "examples" / "benchmark.json"
FIXTURES_DIR = REPO_ROOT / "examples" / "vlm_fixtures"
IMAGES_DIR = REPO_ROOT / "examples" / "images" / "synthetic"
LEAN_PROJECT = REPO_ROOT / "lean_project"
GENERATED_LEAN = LEAN_PROJECT / "DiagramTheorems" / "Generated.lean"


def _lean_runner() -> LeanRunner | None:
    if not LEAN_PROJECT.exists():
        return None
    return LeanRunner(project_root=LEAN_PROJECT, generated_path=GENERATED_LEAN)


def _adapter(mode: str, api_key: str):
    if mode == "Gemini (live)":
        if not api_key:
            st.error("GEMINI_API_KEY missing. Set it in .env.")
            st.stop()
        return GeminiAdapter(api_key=api_key)
    return FixtureAdapter(FIXTURES_DIR)


def _status_badge(status: LeanStatus) -> str:
    if status is LeanStatus.OK:
        return ":green[✓ Type-checked by Lean]"
    if status is LeanStatus.TYPE_ERROR:
        return ":red[✗ Lean type error]"
    return ":orange[Lean unavailable]"


def interactive_page() -> None:
    st.header("Interactive: diagram → Lean theorem")

    examples = load_benchmark(BENCHMARK_PATH)
    choices = {e.id: e for e in examples}
    selection = st.selectbox("Benchmark example:", list(choices.keys()))
    example = choices[selection]

    image_path = REPO_ROOT / example.image
    st.image(str(image_path), caption=example.id, use_column_width=False, width=400)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    mode = st.sidebar.radio(
        "VLM adapter",
        options=["Fixture (deterministic)", "Gemini (live)"],
        help="Fixture mode uses the committed VLM fixture; live mode calls Gemini.",
    )
    st.sidebar.caption(f"API key set: {bool(api_key)}")

    st.subheader("Problem text")
    problem_text = st.text_area("Optional problem statement", value=example.problem_text, height=80)

    if st.button("Run VLM extraction"):
        try:
            adapter = _adapter(mode, api_key)
            fixture_path = REPO_ROOT / example.vlm_fixture if example.vlm_fixture else None
            reading = adapter.read(image_path, fixture_path=fixture_path)
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["reading"] = reading
        st.session_state["problem_text"] = problem_text
        st.session_state["example"] = example

    reading = st.session_state.get("reading")
    if reading is None:
        return

    st.subheader("VLM reading")
    st.json(reading.to_dict())

    candidate = infer_assumptions(
        problem_text=st.session_state.get("problem_text", ""),
        objects=reading.objects,
        diagram_marks=[mark.to_dict() for mark in reading.marks],
        confirmed_assumptions=None,
        relations=reading.relations,
    )

    st.subheader("Confirm assumptions")
    confirmed = [a for a in candidate if st.checkbox(a, value=True, key=f"chk-{a}")]
    extra = st.text_input("Additional assumption (optional)", key="extra-assumption")
    if extra.strip():
        confirmed.append(extra.strip())

    extracted_goal = extract_goal_from_text(st.session_state.get("problem_text", ""))
    ranked = rank_goal_candidates(reading.objects, confirmed)
    candidates = [extracted_goal, *ranked] if extracted_goal else ranked
    candidates = [c for c in candidates if c]

    st.subheader("Select goal")
    if not candidates:
        selected = st.text_input("Enter goal manually", key="manual-goal")
    else:
        selected = st.radio("Ranked goal candidates", candidates, key="goal-radio")

    if st.button("Generate Lean theorem") and selected:
        source = theorem_from(
            name=st.session_state["example"].lean_theorem_name,
            assumptions=confirmed,
            goal=selected,
        )
        runner = _lean_runner()
        if runner:
            status, stderr = runner.typecheck(source)
        else:
            status, stderr = LeanStatus.UNAVAILABLE, None

        st.subheader("Lean output")
        st.markdown(_status_badge(status))
        if stderr:
            st.code(stderr, language="text")
        st.code(source, language="lean")
        st.download_button(
            "Download .lean",
            data=source,
            file_name=f"{st.session_state['example'].lean_theorem_name}.lean",
        )


def benchmark_page() -> None:
    st.header("Benchmark: gold-confirmed run")

    examples = load_benchmark(BENCHMARK_PATH)
    adapter = FixtureAdapter(FIXTURES_DIR)
    runner = _lean_runner()

    rows = []
    per_category: dict[str, list] = {}
    for example in examples:
        fixture_path = REPO_ROOT / example.vlm_fixture
        try:
            result = run_pipeline(
                image_path=REPO_ROOT / example.image,
                problem_text=example.problem_text,
                confirmed_assumptions=example.gold_assumptions,
                vlm=adapter,
                vlm_fixture_path=fixture_path,
                lean_runner=runner,
                theorem_name=example.lean_theorem_name,
            )
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(f"{example.id}: {exc}")
            continue

        rows.append({
            "id": example.id,
            "category": example.category,
            "assumption_precision": assumption_precision(result.confirmed_assumptions, example.gold_assumptions),
            "assumption_recall": assumption_recall(result.confirmed_assumptions, example.gold_assumptions),
            "top_1": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 1),
            "top_3": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 3),
            "top_5": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 5),
            "lean_status": result.lean_status.value,
        })
        per_category.setdefault(example.category, []).append(result)

    st.subheader("Per-example")
    st.dataframe(rows)

    st.subheader("Aggregate")
    all_results = [r for group in per_category.values() for r in group]
    st.metric("Lean validity rate", f"{lean_validity_rate(all_results) * 100:.1f}%")
    st.json(category_breakdown(per_category))


def main() -> None:
    st.set_page_config(page_title="Diagram-to-Lean Theorem Assistant", layout="wide")
    page = st.sidebar.selectbox("Page", ["Interactive", "Benchmark"])
    if page == "Interactive":
        interactive_page()
    else:
        benchmark_page()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the UI locally and smoke-test**

```bash
cd "/Users/aadityab/Documents/New project" && source .venv/bin/activate && streamlit run app.py
```

Expected: browser opens at `http://localhost:8501`. The Interactive page loads, the benchmark selectbox has 5 entries, and clicking "Run VLM extraction" populates the assumptions section. Stop with Ctrl+C.

If Lean is installed and mathlib cache is warm (Task 6), clicking "Generate Lean theorem" produces a green "Type-checked by Lean" badge for at least the simplest examples.

- [ ] **Step 3: Run all tests**

```bash
pytest -q
```

Expected: all tests green. No tests for the UI itself (per spec).

- [ ] **Step 4: Commit**

```bash
git add app.py && git commit -m "feat(ui): add Streamlit app with Interactive and Benchmark pages"
```

---

### Task 14: Helper scripts — VLM fixture recorder and Geometry3K fetcher

**Files:**
- Create: `scripts/record_vlm_fixtures.py`
- Create: `scripts/fetch_geometry3k.py`

- [ ] **Step 1: Create `scripts/record_vlm_fixtures.py`**

Create `scripts/record_vlm_fixtures.py`:

```python
"""Refresh VLM fixtures by calling live Gemini.

Usage:
    python scripts/record_vlm_fixtures.py                      # refresh all
    python scripts/record_vlm_fixtures.py --id isosceles_...   # single example
    python scripts/record_vlm_fixtures.py --image path.png     # custom image

Writes JSON files into examples/vlm_fixtures/. Overwrites existing fixtures.
Costs live API calls — use sparingly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from diagram_theorem_assistant.schema import load_benchmark
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "examples" / "vlm_fixtures"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="example_id", help="Single benchmark example id to refresh")
    parser.add_argument("--image", dest="image_path", help="Arbitrary image path (no benchmark entry)")
    parser.add_argument("--out", dest="out_path", help="Output JSON path (required with --image)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set in .env", file=sys.stderr)
        return 1
    adapter = GeminiAdapter(api_key=api_key)

    if args.image_path:
        if not args.out_path:
            print("--out is required with --image", file=sys.stderr)
            return 1
        reading = adapter.read(Path(args.image_path))
        Path(args.out_path).write_text(json.dumps(reading.to_dict(), indent=2))
        print(f"Wrote {args.out_path}")
        return 0

    examples = load_benchmark(REPO_ROOT / "examples" / "benchmark.json")
    if args.example_id:
        examples = [e for e in examples if e.id == args.example_id]
        if not examples:
            print(f"No example with id {args.example_id}", file=sys.stderr)
            return 1

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for example in examples:
        image = REPO_ROOT / example.image
        if not image.exists():
            print(f"Skipping {example.id}: image {image} missing", file=sys.stderr)
            continue
        reading = adapter.read(image)
        out = REPO_ROOT / example.vlm_fixture if example.vlm_fixture else FIXTURES_DIR / f"{example.id}.json"
        out.write_text(json.dumps(reading.to_dict(), indent=2))
        print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create `scripts/fetch_geometry3k.py`**

Create `scripts/fetch_geometry3k.py`:

```python
"""Fetch a small curated set of Geometry3K diagrams for the realistic benchmark tier.

Geometry3K is published by Lu et al. (ACL 2021) at
https://github.com/lupantech/InterGPS. The full dataset is ~3,002 problems;
we download only the ~10 samples listed in CURATED_IDS to stay within
reasonable disk and license footprints.

Usage:
    python scripts/fetch_geometry3k.py

Writes images to examples/images/geometry3k/ (gitignored). If the upstream
layout changes, update BASE_URL and the list of sample IDs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "examples" / "images" / "geometry3k"
MANIFEST = REPO_ROOT / "examples" / "geometry3k_manifest.json"

BASE_URL = "https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k/test"

CURATED_IDS = [
    "2401", "2402", "2403", "2404", "2405",
    "2406", "2407", "2408", "2409", "2410",
]


def _fetch(example_id: str) -> tuple[bytes, str]:
    url = f"{BASE_URL}/{example_id}/img_diagram.png"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content, hashlib.sha256(response.content).hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for example_id in CURATED_IDS:
        try:
            content, digest = _fetch(example_id)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {example_id}: {exc}", file=sys.stderr)
            continue
        out = OUTPUT_DIR / f"{example_id}.png"
        out.write_bytes(content)
        manifest.append({"id": example_id, "path": str(out.relative_to(REPO_ROOT)), "sha256": digest})
        print(f"Saved {out.relative_to(REPO_ROOT)} ({len(content)} bytes)")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the fetcher runs (network-dependent; skip if offline)**

```bash
cd "/Users/aadityab/Documents/New project" && source .venv/bin/activate && python scripts/fetch_geometry3k.py
```

Expected: prints "Saved ..." for each successfully downloaded image. If upstream layout has changed, the script prints "Skipping ..." per ID — that's acceptable; update `BASE_URL` manually later.

The manifest file gets committed (it's metadata, not images).

- [ ] **Step 4: Commit**

```bash
git add scripts/record_vlm_fixtures.py scripts/fetch_geometry3k.py examples/geometry3k_manifest.json 2>/dev/null; git add scripts/record_vlm_fixtures.py scripts/fetch_geometry3k.py && git commit -m "feat(scripts): VLM fixture recorder and Geometry3K fetcher"
```

---

### Task 15: Update README with complete setup and usage

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README content**

Replace `README.md` with:

````markdown
# Diagram-to-Lean Theorem Assistant

Translate geometry diagrams and optional problem text into Lean 4 theorem candidates via a vision-language model. The system extracts named objects, relations, and visual marks from an input image, proposes candidate assumptions and ranked goal theorems, lets a human confirm, and emits a Lean source file that imports `Mathlib` and type-checks.

## Design and plan

- Design spec: `docs/superpowers/specs/2026-04-23-multimodal-math-diagram-to-lean-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-24-multimodal-math-vlm-pipeline-plan.md`
- Proposal: `docs/project-proposal.md`
- Evaluation plan: `docs/evaluation-plan.md`

## Setup

### 1. System tools

Install Homebrew (skip if present), Python 3.11, git, and Lean 4 via elan:

```bash
[ -x "$(command -v brew)" ] || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 git
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:stable
```

### 2. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. API key

Create `.env` in the repo root (gitignored):

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

Optionally override the model:

```bash
echo "GEMINI_MODEL=gemini-2.5-flash" >> .env
```

### 4. Lean project + mathlib cache

```bash
cd lean_project
lake update
lake exe cache get       # ~15 min first time, ~1.5 GB
cd ..
```

### 5. Synthetic images (if not already generated)

```bash
python scripts/generate_synthetic.py
```

## Running

### Streamlit UI

```bash
streamlit run app.py
```

- **Interactive page:** pick a benchmark example → review VLM reading → confirm assumptions → pick goal → generate Lean → type-check.
- **Benchmark page:** runs all 5 benchmark examples with gold confirmations via `FixtureAdapter`; shows precision, recall, top-k, Lean validity.

### Demo script (no UI)

```bash
python -m diagram_theorem_assistant.demo
```

### Tests

```bash
pytest                                 # fast unit + integration (< 10s)
pytest -m lean                         # real `lake build` smoke test
pytest -m live                         # real Gemini API call
pytest -m "not lean and not live"      # strict CI-friendly filter
```

## Refreshing VLM fixtures

To capture live Gemini outputs for benchmark examples:

```bash
python scripts/record_vlm_fixtures.py                      # all
python scripts/record_vlm_fixtures.py --id isosceles_...   # one
```

Costs API calls. Overwrites existing fixtures.

## Optional: Geometry3K tier

To add ~10 real-diagram samples:

```bash
python scripts/fetch_geometry3k.py
```

Images land in `examples/images/geometry3k/` (gitignored). Manifest (hashes and IDs) is committed.

## What this project is — and isn't

**Is:** a research prototype that produces *correctly initialized* Lean theorem statements from diagrams. Statement-level Lean validity is the deliverable.

**Isn't:** an automated theorem prover. Generated theorems close with `sorry`. Bridging that gap is open research.

See the design spec for the full scope, risks, and comparison with Lean Blueprint.
````

- [ ] **Step 2: Commit**

```bash
git add README.md && git commit -m "docs: overhaul README with full setup, usage, and test tiers"
```

---

## Self-Review

### Spec coverage

- §2 in scope: VLM adapter with both implementations → Tasks 4, 5. Real Lean 4 project → Task 6. Streamlit UI with two pages → Task 13. 5 synthetic + ~10 Geometry3K → Tasks 3, 14. Evaluation metrics → Task 11. Three-tier testing → Task 1 (markers) + tests in every task.
- §2 out of scope items: no implementation tasks exist for theorem proving, persistence, UI unit tests, LeanGeo, full Geometry3K. Correct.
- §3 architecture: orchestrator → Task 10. Subprocess Lean → Task 9. Pure-functional stages → Task 10 design.
- §4 VLM prompting (multi-step): Task 5 implements exactly this.
- §5 module layout: every module listed appears in a task.
- §6 core types: Task 2 defines all of them.
- §7 data flow: interactive + benchmark both implemented in Task 13.
- §8 error handling: `VLMError`, `FixtureNotFoundError`, `LeanError`, `LeanStatus.UNAVAILABLE` all wired in Tasks 4, 9, 10, 13.
- §9 testing strategy: three tiers via markers (Task 1); `@pytest.mark.lean` (Task 9); `@pytest.mark.live` (Task 5).
- §10 non-choices: `evaluation.py` → `metrics.py` done in Task 11; `lean_export.py` rewritten in Task 7 (with shim, shim removed in Task 12).

No gaps.

### Placeholder scan

No "TBD", "TODO", "fill in", "add validation", or "similar to Task N" markers. Every code block is complete.

### Type and name consistency

- `DiagramUnderstander.read(image_path, *, fixture_path=None)` — used identically in `FixtureAdapter.read` (Task 4), `GeminiAdapter.read` (Task 5), and `run_pipeline` (Task 10).
- `LeanRunner.typecheck(lean_source) -> tuple[LeanStatus, str | None]` — used identically in `run_pipeline` (Task 10), `demo.py` (Task 12), and `app.py` (Task 13).
- `PipelineResult` fields — referenced consistently in Tasks 10, 11, 13.
- `theorem_from(name, assumptions, goal)` — called identically in Tasks 10, 12, 13.
- `infer_assumptions(problem_text, objects, diagram_marks, confirmed_assumptions, relations)` — called identically in Tasks 10, 13. The `relations=` keyword added in Task 8 is used in Tasks 10 and 13.

No mismatches found.

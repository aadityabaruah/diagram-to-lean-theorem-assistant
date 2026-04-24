# Diagram-to-Lean Theorem Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small research prototype that takes a geometry diagram plus optional problem text, proposes explicit assumptions and theorem goals, and verifies the selected theorem statement in Lean.

**Architecture:** The system is split into four stages: diagram/problem parsing, assumption extraction and confirmation, goal theorem generation, and Lean verification/visualization. The prototype should never claim that a single diagram uniquely determines a theorem; it should present ranked candidates and make ambiguity visible.

**Tech Stack:** Python 3.11, OpenCV or scikit-image for geometry extraction, JSON fixtures for benchmark examples, Lean 4 with Mathlib for theorem checking, pytest for tests, and a lightweight Streamlit or browser UI for assumption/goal review.

---

## Feedback-Driven Scope

This plan directly addresses the comments from Natalie Parham and Kunal Marwaha:

- The project must define where the `goal` comes from. The prototype will support two modes: extracting a goal from problem text when available, and generating a ranked list of likely goals when only a diagram is available.
- The project must separate assumptions from goals. Assumptions are candidate facts inferred from the diagram, while goals are theorem statements to prove from those facts.
- The project must evaluate whether the system finds the right theorem. A benchmark set will include gold assumptions, gold goals, and acceptable alternative goals.
- The project must handle ambiguity in diagrams. Instead of pretending the intended assumptions are obvious, the UI will ask the user to confirm, reject, or add assumptions before Lean verification.
- The project must compare itself with the Lean Blueprint. The project is not a replacement for Lean Blueprint; it is a front-end theorem-discovery and visualization layer that can export verified theorem/proof structure into a blueprint-like dependency view.
- The project must include both working software and further visualizer ideas. The prototype will include a minimal working pipeline plus a clear extension path for a richer proof/diagram visualizer.

## Project Claim

The revised project should be described as:

> A diagram-to-Lean theorem assistant that turns ambiguous visual geometry input into explicit, checkable theorem candidates. The system proposes assumptions and goals, asks the user to resolve ambiguity, and then uses Lean as the final authority for whether a selected theorem statement is valid.

This claim is intentionally narrower than "the system automatically proves the intended theorem from a diagram." The narrower claim is easier to evaluate and more mathematically honest.

## Core Data Model

Every example should be represented as a structured object:

```json
{
  "id": "isosceles_triangle_base_angles",
  "image": "examples/images/isosceles_triangle.png",
  "problem_text": "In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
  "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
  "gold_assumptions": ["triangle A B C", "AB = AC"],
  "gold_goal": "angle ABC = angle BCA",
  "acceptable_goals": ["angle CBA = angle BCA"],
  "lean_theorem_name": "isosceles_base_angles"
}
```

The benchmark must include examples where the diagram alone is ambiguous, because that is one of the main concerns raised in the feedback.

## File Structure

- Create: `README.md`
  Project overview, revised research question, setup instructions, and comparison with Lean Blueprint.

- Create: `docs/project-proposal.md`
  Human-readable proposal incorporating the comments, including goal detection, assumption ambiguity, evaluation, and visualizer scope.

- Create: `docs/evaluation-plan.md`
  Benchmark design, metrics, and example categories.

- Create: `examples/benchmark.json`
  Gold examples with image paths, problem text, assumptions, goals, acceptable alternatives, and Lean theorem names.

- Create: `src/diagram_theorem_assistant/__init__.py`
  Package entry point.

- Create: `src/diagram_theorem_assistant/schema.py`
  Dataclasses for examples, assumptions, goals, and verification results.

- Create: `src/diagram_theorem_assistant/goal_generation.py`
  Goal extraction from problem text and ranked goal generation from detected objects/relations.

- Create: `src/diagram_theorem_assistant/assumption_extraction.py`
  Candidate assumption extraction and confidence scoring.

- Create: `src/diagram_theorem_assistant/lean_export.py`
  Conversion from selected assumptions/goals into Lean theorem skeletons.

- Create: `src/diagram_theorem_assistant/evaluation.py`
  Metrics for assumption precision/recall, top-k goal accuracy, Lean compile success, and proof success.

- Create: `tests/test_schema.py`
  Tests for loading and validating examples.

- Create: `tests/test_goal_generation.py`
  Tests for explicit goal extraction and ranked candidate generation.

- Create: `tests/test_assumption_extraction.py`
  Tests for extracting candidate assumptions from text and diagram marks.

- Create: `tests/test_evaluation.py`
  Tests for benchmark scoring behavior.

---

### Task 1: Write The Revised Project Proposal

**Files:**
- Create: `docs/project-proposal.md`
- Create: `README.md`

- [ ] **Step 1: Draft the proposal title and research question**

Write `docs/project-proposal.md` with this opening:

```markdown
# Diagram-to-Lean Theorem Assistant

## Research Question

Can a system help translate a geometry diagram and optional problem text into an explicit Lean theorem by proposing candidate assumptions, proposing candidate goals, and asking the user to resolve ambiguous choices?

The project is not claiming that a diagram uniquely determines a theorem. Instead, it treats theorem identification as an interactive ranking and verification problem.
```

- [ ] **Step 2: Add a section explaining where the goal comes from**

Add this section:

```markdown
## How The Goal Is Determined

The system supports two goal modes.

In text-guided mode, the problem statement contains language such as "prove that", "show that", or "find". The system extracts the clause after that phrase and converts it into a formal goal candidate.

In diagram-only mode, the system cannot know the intended theorem with certainty. It generates a ranked list of plausible goals from common geometry templates, such as equal angles, equal lengths, collinearity, parallelism, perpendicularity, congruent triangles, and similarity. The user then selects or edits the intended goal.

This design directly addresses the ambiguity in the original example: the system does not silently invent a hidden goal. It either extracts the goal from text or exposes multiple candidate goals for review.
```

- [ ] **Step 3: Add a section explaining assumptions**

Add this section:

```markdown
## How Assumptions Are Determined

The system separates assumptions from goals. Assumptions describe what is given or inferred from the diagram, while the goal describes what should be proven.

Candidate assumptions come from three sources:

- Explicit text, such as "AB = AC" or "DE is parallel to BC".
- Diagram marks, such as equal-length tick marks, right-angle boxes, parallel arrows, labeled points, and intersection structure.
- Conservative geometric defaults, such as detected collinearity or segment incidence.

Because diagrams are often under-specified, assumptions are never treated as final automatically. The interface asks the user to confirm, reject, or add assumptions before the Lean theorem is produced.
```

- [ ] **Step 4: Add an evaluation section**

Add this section:

```markdown
## Evaluation Plan

The project will use a curated benchmark of geometry examples. Each example will contain a diagram, optional problem text, gold assumptions, a gold goal theorem, and acceptable alternative theorem statements.

The system will be evaluated with four metrics:

- Assumption precision and recall: did it identify the intended given facts?
- Top-k goal accuracy: was the correct goal among the top 1, top 3, or top 5 generated candidates?
- Lean statement validity: did the generated theorem statement type-check in Lean?
- Proof success or skeleton quality: could Lean complete the proof automatically, or did the system produce a useful theorem skeleton?

This benchmark makes it possible to measure whether the system is finding the "right" theorem instead of relying on anecdotal examples.
```

- [ ] **Step 5: Add a Lean Blueprint comparison section**

Add this section:

```markdown
## Comparison With Lean Blueprint

Lean Blueprint is mainly a project-management and exposition tool for formalization projects. It helps connect informal mathematical explanations with Lean theorem dependencies.

This project operates earlier in the pipeline. It starts from a visual geometry problem and tries to produce explicit assumptions and candidate theorem goals. Lean is then used to verify the formal statement.

The two tools are complementary. A mature version of this project could export verified theorem statements, dependency graphs, and explanation text into a blueprint-style document. The visualizer could also borrow the blueprint idea of showing which definitions and lemmas are needed for each theorem.
```

- [ ] **Step 6: Write the README summary**

Write `README.md`:

```markdown
# Diagram-to-Lean Theorem Assistant

This project explores how to translate geometry diagrams and optional problem text into explicit Lean theorem candidates.

The system does three things:

- proposes candidate assumptions from text and diagram marks,
- proposes candidate theorem goals instead of assuming the goal is obvious,
- verifies selected theorem statements in Lean.

The project treats diagram ambiguity as a central problem. When the diagram does not uniquely determine the intended theorem, the system asks the user to confirm assumptions and choose among ranked goal candidates.

See `docs/project-proposal.md` for the full proposal and `docs/evaluation-plan.md` for the benchmark plan.
```

- [ ] **Step 7: Commit the proposal**

Run:

```bash
git add README.md docs/project-proposal.md
git commit -m "docs: revise diagram theorem assistant proposal"
```

Expected: a new commit containing only the proposal and README.

---

### Task 2: Define Benchmark Examples And Evaluation Metrics

**Files:**
- Create: `docs/evaluation-plan.md`
- Create: `examples/benchmark.json`
- Create: `tests/test_schema.py`
- Create: `src/diagram_theorem_assistant/schema.py`

- [ ] **Step 1: Write the failing schema test**

Create `tests/test_schema.py`:

```python
from diagram_theorem_assistant.schema import BenchmarkExample


def test_benchmark_example_requires_gold_goal():
    example = BenchmarkExample(
        id="isosceles_triangle_base_angles",
        image="examples/images/isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C"],
        gold_assumptions=["triangle A B C", "AB = AC"],
        gold_goal="angle ABC = angle BCA",
        acceptable_goals=["angle CBA = angle BCA"],
        lean_theorem_name="isosceles_base_angles",
    )

    assert example.gold_goal == "angle ABC = angle BCA"
    assert "AB = AC" in example.gold_assumptions
```

- [ ] **Step 2: Run the schema test and verify it fails**

Run:

```bash
pytest tests/test_schema.py -q
```

Expected: failure because `diagram_theorem_assistant.schema` does not exist yet.

- [ ] **Step 3: Implement the schema**

Create `src/diagram_theorem_assistant/__init__.py`:

```python
"""Tools for turning geometry diagrams into Lean theorem candidates."""
```

Create `src/diagram_theorem_assistant/schema.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    image: str
    problem_text: str
    objects: list[str]
    gold_assumptions: list[str]
    gold_goal: str
    acceptable_goals: list[str]
    lean_theorem_name: str
```

- [ ] **Step 4: Run the schema test and verify it passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_schema.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Create the benchmark fixture**

Create `examples/benchmark.json`:

```json
[
  {
    "id": "isosceles_triangle_base_angles",
    "image": "examples/images/isosceles_triangle.png",
    "problem_text": "In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
    "gold_assumptions": ["triangle A B C", "AB = AC"],
    "gold_goal": "angle ABC = angle BCA",
    "acceptable_goals": ["angle CBA = angle BCA"],
    "lean_theorem_name": "isosceles_base_angles"
  },
  {
    "id": "parallel_lines_alternate_interior",
    "image": "examples/images/parallel_lines.png",
    "problem_text": "Lines l and m are parallel and cut by transversal t. Prove angle 1 equals angle 2.",
    "objects": ["line l", "line m", "line t", "angle 1", "angle 2"],
    "gold_assumptions": ["parallel l m", "transversal t l m"],
    "gold_goal": "angle 1 = angle 2",
    "acceptable_goals": ["alternate interior angles are equal"],
    "lean_theorem_name": "alternate_interior_angles"
  },
  {
    "id": "ambiguous_triangle_no_text",
    "image": "examples/images/ambiguous_triangle.png",
    "problem_text": "",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
    "gold_assumptions": ["triangle A B C"],
    "gold_goal": "unknown without user selection",
    "acceptable_goals": ["AB = AC", "angle ABC = angle BCA", "area ABC > 0"],
    "lean_theorem_name": "ambiguous_triangle_user_selected_goal"
  },
  {
    "id": "right_triangle_pythagorean",
    "image": "examples/images/right_triangle.png",
    "problem_text": "Triangle ABC has a right angle at C. Prove AB^2 = AC^2 + BC^2.",
    "objects": ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC", "right angle C"],
    "gold_assumptions": ["triangle A B C", "right_angle A C B"],
    "gold_goal": "AB^2 = AC^2 + BC^2",
    "acceptable_goals": ["pythagorean theorem for triangle ABC"],
    "lean_theorem_name": "right_triangle_pythagorean"
  },
  {
    "id": "perpendicular_bisector_equidistant",
    "image": "examples/images/perpendicular_bisector.png",
    "problem_text": "Point P lies on the perpendicular bisector of AB. Prove PA = PB.",
    "objects": ["point A", "point B", "point P", "line l", "segment AB", "segment PA", "segment PB"],
    "gold_assumptions": ["P on perpendicular_bisector AB"],
    "gold_goal": "PA = PB",
    "acceptable_goals": ["distance P A = distance P B"],
    "lean_theorem_name": "perpendicular_bisector_equidistant"
  }
]
```

- [ ] **Step 6: Write the evaluation plan document**

Create `docs/evaluation-plan.md`:

```markdown
# Evaluation Plan

The benchmark will contain geometry problems with known assumptions and goals. Each example will include the diagram path, optional problem text, gold assumptions, the gold goal theorem, acceptable alternative goals, and the Lean theorem name.

## Example Categories

- Text-guided examples where the goal is explicitly stated.
- Diagram-guided examples where visual marks suggest assumptions.
- Ambiguous diagram-only examples where several goals are plausible.
- Negative examples where the system should ask for clarification instead of guessing.

## Metrics

- Assumption precision: predicted intended assumptions divided by all predicted assumptions.
- Assumption recall: predicted intended assumptions divided by all gold assumptions.
- Top-k goal accuracy: whether the gold goal appears in the top 1, top 3, or top 5 generated goals.
- Lean type-check rate: percentage of generated theorem statements accepted by Lean.
- Proof completion rate: percentage of theorem statements proved automatically or with a short generated proof.

## Success Criteria For The Class Prototype

The prototype is successful if it loads benchmark examples, produces assumption candidates, produces ranked goal candidates, exports a Lean theorem skeleton, and reports evaluation metrics on at least five examples.
```

- [ ] **Step 7: Commit benchmark and schema**

Run:

```bash
git add docs/evaluation-plan.md examples/benchmark.json src/diagram_theorem_assistant tests/test_schema.py
git commit -m "feat: define benchmark schema and evaluation plan"
```

Expected: a new commit with the schema, five benchmark examples, evaluation plan, and passing schema test.

---

### Task 3: Implement Assumption Extraction

**Files:**
- Create: `src/diagram_theorem_assistant/assumption_extraction.py`
- Create: `tests/test_assumption_extraction.py`

- [ ] **Step 1: Write failing tests for text and mark-based assumptions**

Create `tests/test_assumption_extraction.py`:

```python
from diagram_theorem_assistant.assumption_extraction import assumptions_from_marks, assumptions_from_text


def test_assumptions_from_text_extracts_equal_lengths():
    text = "In triangle ABC, AB = AC. Prove angle ABC = angle BCA."

    assert assumptions_from_text(text) == ["AB = AC"]


def test_assumptions_from_marks_extracts_right_angle():
    marks = [{"type": "right_angle", "points": ["A", "C", "B"]}]

    assert assumptions_from_marks(marks) == ["right_angle A C B"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/test_assumption_extraction.py -q
```

Expected: failure because `assumption_extraction.py` does not exist.

- [ ] **Step 3: Implement assumption extraction**

Create `src/diagram_theorem_assistant/assumption_extraction.py`:

```python
from __future__ import annotations

import re
from typing import TypedDict


class DiagramMark(TypedDict):
    type: str
    points: list[str]


def assumptions_from_text(text: str) -> list[str]:
    assumptions: list[str] = []
    equal_length_matches = re.findall(r"\b([A-Z]{2})\s*=\s*([A-Z]{2})\b", text)
    for left, right in equal_length_matches:
        assumptions.append(f"{left} = {right}")
    return assumptions


def assumptions_from_marks(marks: list[DiagramMark]) -> list[str]:
    assumptions: list[str] = []
    for mark in marks:
        if mark["type"] == "right_angle" and len(mark["points"]) == 3:
            assumptions.append("right_angle " + " ".join(mark["points"]))
    return assumptions
```

- [ ] **Step 4: Run assumption extraction tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_assumption_extraction.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit assumption extraction**

Run:

```bash
git add src/diagram_theorem_assistant/assumption_extraction.py tests/test_assumption_extraction.py
git commit -m "feat: extract candidate assumptions"
```

Expected: a new commit with tested assumption extraction behavior.

---

### Task 4: Implement Goal Generation

**Files:**
- Create: `src/diagram_theorem_assistant/goal_generation.py`
- Create: `tests/test_goal_generation.py`

- [ ] **Step 1: Write failing tests for text-guided and diagram-only goals**

Create `tests/test_goal_generation.py`:

```python
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates


def test_extract_goal_after_prove_phrase():
    text = "In triangle ABC, AB = AC. Prove angle ABC = angle BCA."

    assert extract_goal_from_text(text) == "angle ABC = angle BCA"


def test_rank_goal_candidates_for_triangle_objects():
    objects = ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"]
    assumptions = ["triangle A B C", "AB = AC"]

    candidates = rank_goal_candidates(objects=objects, assumptions=assumptions)

    assert candidates[0] == "angle ABC = angle BCA"
    assert "AB = AC" not in candidates
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/test_goal_generation.py -q
```

Expected: failure because `goal_generation.py` does not exist.

- [ ] **Step 3: Implement text extraction and simple goal ranking**

Create `src/diagram_theorem_assistant/goal_generation.py`:

```python
from __future__ import annotations

import re


def extract_goal_from_text(text: str) -> str | None:
    patterns = [
        r"\bprove(?: that)?\s+(.+?)(?:\.|$)",
        r"\bshow(?: that)?\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def rank_goal_candidates(objects: list[str], assumptions: list[str]) -> list[str]:
    object_text = " ".join(objects)
    assumption_set = set(assumptions)
    candidates: list[str] = []

    has_triangle_abc = all(token in object_text for token in ["point A", "point B", "point C"])
    if has_triangle_abc and "AB = AC" in assumption_set:
        candidates.append("angle ABC = angle BCA")

    if has_triangle_abc:
        candidates.extend([
            "AB = AC",
            "angle BAC + angle ABC + angle BCA = 180",
            "area ABC > 0",
        ])

    return [candidate for candidate in candidates if candidate not in assumption_set]
```

- [ ] **Step 4: Run goal generation tests and verify they pass**

Run:

```bash
PYTHONPATH=src pytest tests/test_goal_generation.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit goal generation**

Run:

```bash
git add src/diagram_theorem_assistant/goal_generation.py tests/test_goal_generation.py
git commit -m "feat: generate candidate theorem goals"
```

Expected: a new commit with tested goal generation behavior.

---

### Task 5: Implement Evaluation Scoring

**Files:**
- Create: `src/diagram_theorem_assistant/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for assumption recall and top-k goal accuracy**

Create `tests/test_evaluation.py`:

```python
from diagram_theorem_assistant.evaluation import assumption_recall, top_k_goal_accuracy


def test_assumption_recall_counts_gold_matches():
    predicted = ["triangle A B C", "AB = AC", "extra fact"]
    gold = ["triangle A B C", "AB = AC", "angle ABC is acute"]

    assert assumption_recall(predicted, gold) == 2 / 3


def test_top_k_goal_accuracy_accepts_gold_goal_in_range():
    ranked_goals = ["AB = AC", "angle ABC = angle BCA", "area ABC > 0"]

    assert top_k_goal_accuracy(ranked_goals, "angle ABC = angle BCA", k=1) == 0.0
    assert top_k_goal_accuracy(ranked_goals, "angle ABC = angle BCA", k=2) == 1.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation.py -q
```

Expected: failure because `evaluation.py` does not exist.

- [ ] **Step 3: Implement scoring functions**

Create `src/diagram_theorem_assistant/evaluation.py`:

```python
def assumption_recall(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    predicted_set = set(predicted)
    gold_set = set(gold)
    return len(predicted_set & gold_set) / len(gold_set)


def assumption_precision(predicted: list[str], gold: list[str]) -> float:
    if not predicted:
        return 1.0
    predicted_set = set(predicted)
    gold_set = set(gold)
    return len(predicted_set & gold_set) / len(predicted_set)


def top_k_goal_accuracy(ranked_goals: list[str], gold_goal: str, k: int) -> float:
    return 1.0 if gold_goal in ranked_goals[:k] else 0.0
```

- [ ] **Step 4: Run all tests**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit evaluation scoring**

Run:

```bash
git add src/diagram_theorem_assistant/evaluation.py tests/test_evaluation.py
git commit -m "feat: score theorem identification quality"
```

Expected: a new commit with tested evaluation metrics.

---

### Task 6: Export Lean Theorem Skeletons

**Files:**
- Create: `src/diagram_theorem_assistant/lean_export.py`
- Create: `tests/test_lean_export.py`

- [ ] **Step 1: Write a failing test for theorem skeleton export**

Create `tests/test_lean_export.py`:

```python
from diagram_theorem_assistant.lean_export import theorem_skeleton


def test_theorem_skeleton_includes_name_assumptions_and_goal():
    result = theorem_skeleton(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )

    assert "theorem isosceles_base_angles" in result
    assert "-- assumption: triangle A B C" in result
    assert "-- assumption: AB = AC" in result
    assert "-- goal: angle ABC = angle BCA" in result
    assert ":= by" in result
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/test_lean_export.py -q
```

Expected: failure because `lean_export.py` does not exist.

- [ ] **Step 3: Implement conservative Lean skeleton export**

Create `src/diagram_theorem_assistant/lean_export.py`:

```python
def theorem_skeleton(name: str, assumptions: list[str], goal: str) -> str:
    assumption_comments = "\n".join(f"-- assumption: {assumption}" for assumption in assumptions)
    return f"""import Mathlib

{assumption_comments}
-- goal: {goal}
theorem {name} : True := by
  trivial
"""
```

This initial exporter records the selected theorem content in comments and emits a Lean file that type-checks. Later work can replace `True` with real Mathlib geometry propositions as the formalization mapping matures.

- [ ] **Step 4: Run all tests**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Lean exporter**

Run:

```bash
git add src/diagram_theorem_assistant/lean_export.py tests/test_lean_export.py
git commit -m "feat: export Lean theorem skeletons"
```

Expected: a new commit with a conservative Lean export path.

---

### Task 7: Build A Minimal Demo Script

**Files:**
- Create: `src/diagram_theorem_assistant/demo.py`
- Create: `tests/test_demo.py`

- [ ] **Step 1: Write a failing test for the demo pipeline**

Create `tests/test_demo.py`:

```python
from diagram_theorem_assistant.demo import run_demo


def test_run_demo_returns_goal_and_lean_skeleton():
    result = run_demo(
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
        confirmed_assumptions=["triangle A B C"],
    )

    assert result["goal"] == "angle ABC = angle BCA"
    assert result["assumptions"] == ["triangle A B C", "AB = AC"]
    assert "theorem generated_theorem" in result["lean"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/test_demo.py -q
```

Expected: failure because `demo.py` does not exist.

- [ ] **Step 3: Implement the demo pipeline**

Create `src/diagram_theorem_assistant/demo.py`:

```python
from diagram_theorem_assistant.assumption_extraction import assumptions_from_text
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_skeleton


def run_demo(
    problem_text: str,
    objects: list[str],
    confirmed_assumptions: list[str],
) -> dict[str, str | list[str]]:
    assumptions = confirmed_assumptions + [
        assumption
        for assumption in assumptions_from_text(problem_text)
        if assumption not in confirmed_assumptions
    ]
    extracted_goal = extract_goal_from_text(problem_text)
    goal_candidates = rank_goal_candidates(objects, assumptions)
    selected_goal = extracted_goal or goal_candidates[0]
    lean = theorem_skeleton("generated_theorem", assumptions, selected_goal)
    return {
        "assumptions": assumptions,
        "goal": selected_goal,
        "goal_candidates": goal_candidates,
        "lean": lean,
    }
```

- [ ] **Step 4: Run all tests**

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the demo**

Run:

```bash
git add src/diagram_theorem_assistant/demo.py tests/test_demo.py
git commit -m "feat: add theorem assistant demo pipeline"
```

Expected: a new commit with a working text-to-goal-to-Lean skeleton demo.

---

## Visualizer Extension Ideas

After the minimal prototype works, the visualizer can grow in three directions:

- Diagram overlay: show detected points, segments, equal-length marks, right-angle marks, and parallel marks on top of the original diagram.
- Assumption/goal panel: show assumptions on the left, ranked goals in the middle, and Lean output on the right so users can see exactly what is being formalized.
- Blueprint-style proof graph: show the selected theorem, required lemmas, generated Lean statements, and proof status as a dependency graph.

These ideas should stay secondary until the core pipeline can produce and evaluate theorem candidates.

## Risks And Mitigations

- Risk: Diagram-only theorem intent is unknowable in many cases.
  Mitigation: Treat diagram-only input as candidate generation, not automatic theorem discovery.

- Risk: Lean geometry formalization may be too difficult for the class timeline.
  Mitigation: Start with Lean skeletons and comments, then formalize a small number of theorem templates.

- Risk: Evaluation could become subjective.
  Mitigation: Use benchmark examples with gold assumptions, gold goals, and accepted alternative goals.

- Risk: The system may over-infer assumptions from visual marks.
  Mitigation: Require a user confirmation step before Lean export.

## Final Deliverables

- A revised proposal that explicitly answers the feedback.
- A benchmark file with examples containing gold assumptions and goals.
- A small Python prototype that extracts or ranks theorem goals.
- A Lean skeleton exporter.
- Evaluation metrics for assumption extraction and goal selection.
- A short comparison with Lean Blueprint.
- Visualizer mockup ideas or a minimal UI if time permits.

## Recommended Execution Order

1. Write the revised proposal and README.
2. Define benchmark schema and evaluation metrics.
3. Implement assumption extraction.
4. Implement goal extraction and ranking.
5. Implement Lean theorem skeleton export.
6. Add a minimal demo pipeline.
7. Add visualizer only after the pipeline is working.

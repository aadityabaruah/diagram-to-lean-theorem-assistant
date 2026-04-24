"""Orchestration pipeline: VLM reading → assumptions → goals → Lean typecheck."""
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
    """Run the full diagram-theorem pipeline.

    Steps
    -----
    1. Read the diagram via *vlm*.
    2. Infer candidate assumptions from objects, marks, relations, and problem text.
    3. Use *confirmed_assumptions* if supplied; otherwise fall back to inferred candidates.
    4. Extract a goal from *problem_text*.
    5. Rank goal candidates from objects + confirmed assumptions.
    6. Build an ordered list of goal candidates (extracted first, then ranked).
    7. Select the primary goal (extracted > first ranked > empty string).
    8. Emit Lean source via :func:`theorem_from`.
    9. Typecheck with *lean_runner* when present; otherwise return UNAVAILABLE.
    10. Return a :class:`~diagram_theorem_assistant.schema.PipelineResult`.
    """
    # Step 1 — VLM reading
    reading = vlm.read(image_path, fixture_path=vlm_fixture_path)

    # Step 2 — infer candidate assumptions
    candidate = infer_assumptions(
        problem_text=problem_text,
        objects=reading.objects,
        diagram_marks=[m.to_dict() for m in reading.marks],
        confirmed_assumptions=None,
        relations=reading.relations,
    )

    # Step 3 — resolve confirmed assumptions
    confirmed = confirmed_assumptions if confirmed_assumptions is not None else candidate

    # Step 4 — extract goal from problem text
    extracted = extract_goal_from_text(problem_text)

    # Step 5 — rank goal candidates
    ranked = rank_goal_candidates(reading.objects, confirmed)

    # Step 6 — build ordered goal_candidates list
    goal_candidates: list[str] = [extracted, *ranked] if extracted else ranked

    # Step 7 — select primary goal
    selected = extracted or (ranked[0] if ranked else "")

    # Step 8 — emit Lean source
    lean_source = theorem_from(name=theorem_name, assumptions=confirmed, goal=selected)

    # Step 9 — typecheck
    if lean_runner is not None:
        status, stderr = lean_runner.typecheck(lean_source)
    else:
        status, stderr = LeanStatus.UNAVAILABLE, None

    # Step 10 — return result
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

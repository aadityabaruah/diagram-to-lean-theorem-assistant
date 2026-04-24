from __future__ import annotations

import json
from typing import Any

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_skeleton


def run_demo(
    problem_text: str,
    objects: list[str],
    diagram_marks: list[dict[str, Any]],
    confirmed_assumptions: list[str] | None = None,
) -> dict[str, Any]:
    assumptions = infer_assumptions(problem_text, objects, diagram_marks, confirmed_assumptions)
    extracted_goal = extract_goal_from_text(problem_text)
    goal_candidates = rank_goal_candidates(objects, assumptions)
    needs_clarification = extracted_goal is None and len(goal_candidates) != 1
    selected_goal = extracted_goal if extracted_goal is not None else _single_goal(goal_candidates)
    lean = theorem_skeleton("generated_theorem", assumptions, selected_goal) if selected_goal else None

    return {
        "assumptions": assumptions,
        "goal": selected_goal,
        "goal_candidates": goal_candidates,
        "needs_clarification": needs_clarification,
        "lean": lean,
    }


def main() -> None:
    result = run_demo(
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
        diagram_marks=[{"type": "equal_length", "segments": ["AB", "AC"]}],
        confirmed_assumptions=[],
    )
    print(json.dumps(result, indent=2))


def _single_goal(goal_candidates: list[str]) -> str | None:
    return goal_candidates[0] if len(goal_candidates) == 1 else None


if __name__ == "__main__":
    main()

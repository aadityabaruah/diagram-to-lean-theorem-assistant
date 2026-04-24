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

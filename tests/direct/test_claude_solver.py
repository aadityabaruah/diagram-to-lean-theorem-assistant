from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.direct.claude_solver import (
    ClaudeDirectSolver,
    _strip_fences,
    _weakened_goal_reason,
)
from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.base import VLMError


def _block(text):
    b = MagicMock()
    b.text = text
    return b


def _msg(text):
    m = MagicMock()
    m.content = [_block(text)]
    return m


GOOD_LEAN = """import DiagramTheorems.Basic
namespace DiagramTheorems.Generated
theorem demo : True := by trivial
end DiagramTheorems.Generated"""

LEAN_WITH_SORRY = """import DiagramTheorems.Basic
namespace DiagramTheorems.Generated
theorem demo : True := by sorry
end DiagramTheorems.Generated"""


def _runner(outcomes):
    r = MagicMock()
    r.typecheck.side_effect = outcomes
    return r


def test_solver_requires_api_key():
    with pytest.raises(VLMError):
        ClaudeDirectSolver(api_key="", client=MagicMock(), model="x")


def test_solver_returns_proof_on_first_attempt(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = MagicMock()
    client.messages.create.side_effect = [_msg(GOOD_LEAN)]
    runner = _runner([(LeanStatus.OK, None)])

    solver = ClaudeDirectSolver(api_key="fake", client=client, model="x")
    source, status, _ = solver.solve(
        image_path=image,
        problem_text="prove triv",
        theorem_name="demo",
        lean_runner=runner,
        max_attempts=2,
    )
    assert status is LeanStatus.OK
    assert "sorry" not in source
    assert "trivial" in source


def test_solver_retries_after_sorry_response(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = MagicMock()
    client.messages.create.side_effect = [_msg(LEAN_WITH_SORRY), _msg(GOOD_LEAN)]
    runner = _runner([(LeanStatus.OK, None), (LeanStatus.OK, None)])

    solver = ClaudeDirectSolver(api_key="fake", client=client, model="x")
    source, status, _ = solver.solve(
        image_path=image,
        problem_text="",
        theorem_name="demo",
        lean_runner=runner,
        max_attempts=3,
    )
    assert status is LeanStatus.OK
    assert "sorry" not in source
    assert client.messages.create.call_count == 2


def test_solver_retries_after_lake_error(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    bad = "import DiagramTheorems.Basic\nnamespace X\ntheorem demo : True := by foo\nend X"
    client = MagicMock()
    client.messages.create.side_effect = [_msg(bad), _msg(GOOD_LEAN)]
    runner = _runner([
        (LeanStatus.TYPE_ERROR, "unknown tactic 'foo'"),
        (LeanStatus.OK, None),
    ])

    solver = ClaudeDirectSolver(api_key="fake", client=client, model="x")
    source, status, _ = solver.solve(
        image_path=image,
        problem_text="",
        theorem_name="demo",
        lean_runner=runner,
        max_attempts=3,
    )
    assert status is LeanStatus.OK
    assert "trivial" in source
    assert client.messages.create.call_count == 2


def test_solver_returns_best_effort_after_max_attempts(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = MagicMock()
    client.messages.create.side_effect = [_msg(LEAN_WITH_SORRY), _msg(LEAN_WITH_SORRY)]
    runner = _runner([(LeanStatus.OK, None), (LeanStatus.OK, None)])

    solver = ClaudeDirectSolver(api_key="fake", client=client, model="x")
    source, status, _ = solver.solve(
        image_path=image,
        problem_text="",
        theorem_name="demo",
        lean_runner=runner,
        max_attempts=2,
    )
    # Last attempt still has sorry — returned for diagnostic purposes
    assert "sorry" in source


def test_strip_fences_handles_thinking_and_fences():
    raw = "<think>reason</think>\n```lean\ntheorem x : True := by trivial\n```"
    assert _strip_fences(raw) == "theorem x : True := by trivial"


def test_weakened_goal_flags_dist_when_problem_asks_for_angle():
    src = """theorem t (A B C : EuclideanSpace ℝ (Fin 2))
    (h : dist A B = dist A C) :
    dist B A = dist C A := by exact h"""
    reason = _weakened_goal_reason(src, "Prove angle ABC = angle BCA")
    assert reason is not None
    assert "angle" in reason.lower()


def test_weakened_goal_accepts_real_angle_goal():
    src = """theorem t (A B C : EuclideanSpace ℝ (Fin 2))
    (h : dist A B = dist A C) :
    ∠ A B C = ∠ B C A := by sorry"""
    assert _weakened_goal_reason(src, "Prove angle ABC = angle BCA") is None


def test_weakened_goal_flags_missing_square_in_pythagoras():
    src = """theorem t (A B C : EuclideanSpace ℝ (Fin 2)) (h : True) :
    dist A B = dist A C + dist B C := by sorry"""
    reason = _weakened_goal_reason(src, "Prove AB^2 = AC^2 + BC^2")
    assert reason is not None
    assert "squared" in reason.lower()

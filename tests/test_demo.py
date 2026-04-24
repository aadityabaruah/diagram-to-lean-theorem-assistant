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

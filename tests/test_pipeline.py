"""Tests for pipeline.py — Task 10 TDD flow."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.pipeline import run_pipeline
from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "examples" / "vlm_fixtures"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "examples" / "images" / "synthetic"


def _make_mock_lean_runner(status: LeanStatus = LeanStatus.OK, stderr: str | None = "") -> MagicMock:
    runner = MagicMock()
    runner.typecheck.return_value = (status, stderr)
    return runner


def test_pipeline_runs_isosceles_end_to_end():
    """Full pipeline: isosceles triangle fixture → assumptions → goal → Lean OK."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    lean_runner = _make_mock_lean_runner(LeanStatus.OK, "")

    result = run_pipeline(
        image_path=IMAGES_DIR / "isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove that angle ABC = angle BCA.",
        confirmed_assumptions=None,
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "isosceles_triangle_base_angles.json",
        lean_runner=lean_runner,
    )

    assert "triangle A B C" in result.confirmed_assumptions
    assert result.selected_goal != ""
    assert result.lean_status is LeanStatus.OK


def test_pipeline_diagram_only_returns_ranked_candidates():
    """Ambiguous diagram (no text goal) returns ranked candidates; selected_goal is first."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    lean_runner = _make_mock_lean_runner(LeanStatus.OK, "")

    result = run_pipeline(
        image_path=IMAGES_DIR / "ambiguous_triangle.png",
        problem_text="",  # no text — forces ranked-only path
        confirmed_assumptions=["triangle A B C"],
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "ambiguous_triangle_no_text.json",
        lean_runner=lean_runner,
    )

    assert result.confirmed_assumptions == ["triangle A B C"]
    assert len(result.goal_candidates) >= 1
    assert result.selected_goal == result.goal_candidates[0]


def test_pipeline_surfaces_lean_unavailable_status():
    """When lean_runner is None the pipeline returns UNAVAILABLE and valid Lean source."""
    vlm = FixtureAdapter(FIXTURES_DIR)

    result = run_pipeline(
        image_path=IMAGES_DIR / "isosceles_triangle.png",
        problem_text="Prove that AB = AC.",
        confirmed_assumptions=None,
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "isosceles_triangle_base_angles.json",
        lean_runner=None,
    )

    assert result.lean_status is LeanStatus.UNAVAILABLE
    assert result.lean_source.startswith("import DiagramTheorems.Basic")


def test_pipeline_returns_reading_with_vlm_marks():
    """Right-triangle fixture exposes a right_angle mark in the returned reading."""
    vlm = FixtureAdapter(FIXTURES_DIR)

    result = run_pipeline(
        image_path=IMAGES_DIR / "right_triangle.png",
        problem_text="Prove that AB^2 = AC^2 + BC^2.",
        confirmed_assumptions=None,
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "right_triangle_pythagorean.json",
        lean_runner=None,
    )

    mark_types = [m.type for m in result.reading.marks]
    assert "right_angle" in mark_types

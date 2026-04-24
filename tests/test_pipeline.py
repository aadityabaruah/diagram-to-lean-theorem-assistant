"""Tests for pipeline.py — run_pipeline + run_pipeline_from_reading."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from diagram_theorem_assistant.pipeline import (
    candidate_assumptions,
    run_pipeline,
    run_pipeline_from_reading,
)
from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "examples" / "vlm_fixtures"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "examples" / "images" / "synthetic"


def _make_mock_lean_runner(status: LeanStatus = LeanStatus.OK, stderr: str | None = None) -> MagicMock:
    runner = MagicMock()
    runner.typecheck.return_value = (status, stderr)
    return runner


def test_pipeline_runs_isosceles_end_to_end():
    """Full pipeline: isosceles triangle fixture → inferred candidates → goal → Lean OK."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    lean_runner = _make_mock_lean_runner(LeanStatus.OK)

    result = run_pipeline(
        image_path=IMAGES_DIR / "isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove that angle ABC = angle BCA.",
        confirmed_assumptions=["triangle A B C", "AB = AC"],
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "isosceles_triangle_base_angles.json",
        lean_runner=lean_runner,
    )

    assert "triangle A B C" in result.confirmed_assumptions
    assert "AB = AC" in result.confirmed_assumptions
    assert result.selected_goal != ""
    assert result.lean_status is LeanStatus.OK


def test_pipeline_diagram_only_returns_ranked_candidates():
    """Ambiguous diagram (no text goal) returns ranked candidates; selected_goal is first."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    lean_runner = _make_mock_lean_runner(LeanStatus.OK)

    result = run_pipeline(
        image_path=IMAGES_DIR / "ambiguous_triangle.png",
        problem_text="",
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
        confirmed_assumptions=["triangle A B C"],
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
        confirmed_assumptions=["triangle A B C", "right_angle A C B"],
        vlm=vlm,
        vlm_fixture_path=FIXTURES_DIR / "right_triangle_pythagorean.json",
        lean_runner=None,
    )

    mark_types = [m.type for m in result.reading.marks]
    assert "right_angle" in mark_types


def test_candidate_assumptions_matches_inferred_list():
    """The helper returns the same list the pipeline uses internally."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    reading = vlm.read(
        IMAGES_DIR / "isosceles_triangle.png",
        fixture_path=FIXTURES_DIR / "isosceles_triangle_base_angles.json",
    )
    problem_text = "In triangle ABC, AB = AC. Prove angle ABC = angle BCA."
    inferred = candidate_assumptions(reading, problem_text)
    assert "triangle A B C" in inferred
    assert "AB = AC" in inferred


def test_run_pipeline_from_reading_skips_vlm_read():
    """Callers with a pre-computed reading can run the pipeline without re-reading."""
    vlm = FixtureAdapter(FIXTURES_DIR)
    reading = vlm.read(
        IMAGES_DIR / "isosceles_triangle.png",
        fixture_path=FIXTURES_DIR / "isosceles_triangle_base_angles.json",
    )
    result = run_pipeline_from_reading(
        reading=reading,
        problem_text="Prove AB = AC.",
        confirmed_assumptions=[],
        lean_runner=None,
    )
    assert result.confirmed_assumptions == []
    assert result.reading is reading

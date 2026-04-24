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

from pathlib import Path

from diagram_theorem_assistant.schema import BenchmarkExample, load_benchmark


def test_benchmark_example_requires_gold_goal():
    example = BenchmarkExample(
        id="isosceles_triangle_base_angles",
        image="examples/images/isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C"],
        diagram_marks=[],
        gold_assumptions=["triangle A B C", "AB = AC"],
        gold_goal="angle ABC = angle BCA",
        acceptable_goals=["angle CBA = angle BCA"],
        lean_theorem_name="isosceles_base_angles",
    )

    assert example.gold_goal == "angle ABC = angle BCA"
    assert "AB = AC" in example.gold_assumptions


def test_load_benchmark_reads_all_examples():
    examples = load_benchmark(Path("examples/benchmark.json"))

    assert len(examples) == 5
    assert examples[0].id == "isosceles_triangle_base_angles"
    assert examples[0].diagram_marks == [{"type": "equal_length", "segments": ["AB", "AC"]}]

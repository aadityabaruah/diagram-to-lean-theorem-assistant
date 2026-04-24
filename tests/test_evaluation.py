from diagram_theorem_assistant.evaluation import (
    assumption_precision,
    assumption_recall,
    evaluate_example,
    evaluate_benchmark,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.schema import BenchmarkExample


def test_assumption_recall_counts_gold_matches():
    predicted = ["triangle A B C", "AB = AC", "extra fact"]
    gold = ["triangle A B C", "AB = AC", "angle ABC is acute"]

    assert assumption_recall(predicted, gold) == 2 / 3


def test_assumption_precision_counts_only_predicted_matches():
    predicted = ["triangle A B C", "AB = AC", "extra fact"]
    gold = ["triangle A B C", "AB = AC"]

    assert assumption_precision(predicted, gold) == 2 / 3


def test_top_k_goal_accuracy_accepts_gold_goal_in_range():
    ranked_goals = ["AB = AC", "angle ABC = angle BCA", "area ABC > 0"]

    assert top_k_goal_accuracy(ranked_goals, "angle ABC = angle BCA", k=1) == 0.0
    assert top_k_goal_accuracy(ranked_goals, "angle ABC = angle BCA", k=2) == 1.0


def test_evaluate_example_scores_assumptions_and_goals():
    example = BenchmarkExample(
        id="isosceles_triangle_base_angles",
        image="examples/images/isosceles_triangle.png",
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C"],
        diagram_marks=[],
        gold_assumptions=["triangle A B C", "AB = AC"],
        gold_goal="angle ABC = angle BCA",
        acceptable_goals=[],
        lean_theorem_name="isosceles_base_angles",
    )

    result = evaluate_example(
        example,
        predicted_assumptions=["triangle A B C", "AB = AC"],
        ranked_goals=["angle ABC = angle BCA"],
    )

    assert result.assumption_precision == 1.0
    assert result.assumption_recall == 1.0
    assert result.top_1_goal_accuracy == 1.0


def test_evaluate_benchmark_scores_all_examples():
    results = evaluate_benchmark("examples/benchmark.json")

    assert len(results) == 5
    assert results[0].example_id == "isosceles_triangle_base_angles"
    assert results[0].top_1_goal_accuracy == 1.0

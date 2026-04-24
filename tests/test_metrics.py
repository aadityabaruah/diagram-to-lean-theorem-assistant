from diagram_theorem_assistant.metrics import (
    assumption_precision,
    assumption_recall,
    category_breakdown,
    lean_validity_rate,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.schema import (
    DiagramMark,
    DiagramReading,
    LeanStatus,
    PipelineResult,
)


def _result(status: LeanStatus, goal_candidates: list[str], selected: str = "") -> PipelineResult:
    return PipelineResult(
        reading=DiagramReading(objects=[], relations=[], marks=[]),
        candidate_assumptions=[],
        confirmed_assumptions=[],
        goal_candidates=goal_candidates,
        selected_goal=selected,
        lean_source="",
        lean_status=status,
        lean_stderr=None,
    )


def test_assumption_precision_and_recall_round_trip():
    predicted = ["a", "b", "c"]
    gold = ["a", "b", "d"]
    assert assumption_precision(predicted, gold) == 2 / 3
    assert assumption_recall(predicted, gold) == 2 / 3


def test_top_k_goal_accuracy_counts_position():
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=1) == 0.0
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=2) == 1.0
    assert top_k_goal_accuracy(["a", "b", "c"], "b", k=5) == 1.0


def test_lean_validity_rate_counts_ok_only():
    results = [
        _result(LeanStatus.OK, ["a"], "a"),
        _result(LeanStatus.TYPE_ERROR, ["a"], "a"),
        _result(LeanStatus.UNAVAILABLE, ["a"], "a"),
        _result(LeanStatus.OK, ["a"], "a"),
    ]
    assert lean_validity_rate(results) == 2 / 4


def test_lean_validity_rate_returns_zero_on_empty():
    assert lean_validity_rate([]) == 0.0


def test_category_breakdown_groups_and_aggregates():
    results_by_category = {
        "text-guided": [_result(LeanStatus.OK, ["x"], "x"), _result(LeanStatus.TYPE_ERROR, ["x"], "x")],
        "diagram-only-ambiguous": [_result(LeanStatus.OK, ["y"], "y")],
    }
    breakdown = category_breakdown(results_by_category)
    assert breakdown["text-guided"]["lean_validity_rate"] == 0.5
    assert breakdown["diagram-only-ambiguous"]["lean_validity_rate"] == 1.0
    assert breakdown["text-guided"]["count"] == 2

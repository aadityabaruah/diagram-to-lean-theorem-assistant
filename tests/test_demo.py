from diagram_theorem_assistant.demo import run_demo


def test_run_demo_returns_goal_assumptions_and_lean_skeleton():
    result = run_demo(
        problem_text="In triangle ABC, AB = AC. Prove angle ABC = angle BCA.",
        objects=["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
        diagram_marks=[],
        confirmed_assumptions=[],
    )

    assert result["needs_clarification"] is False
    assert result["goal"] == "angle ABC = angle BCA"
    assert result["assumptions"] == ["triangle A B C", "AB = AC"]
    assert "theorem generated_theorem" in result["lean"]


def test_run_demo_flags_diagram_only_ambiguity():
    result = run_demo(
        problem_text="",
        objects=["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"],
        diagram_marks=[],
        confirmed_assumptions=[],
    )

    assert result["needs_clarification"] is True
    assert result["goal"] is None
    assert len(result["goal_candidates"]) >= 3

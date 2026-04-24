from diagram_theorem_assistant.assumption_extraction import (
    assumptions_from_marks,
    assumptions_from_objects,
    assumptions_from_text,
    merge_assumptions,
)


def test_assumptions_from_text_extracts_equal_lengths():
    text = "In triangle ABC, AB = AC. Prove angle ABC = angle BCA."

    assert assumptions_from_text(text) == ["triangle A B C", "AB = AC"]


def test_assumptions_from_marks_extracts_right_angle():
    marks = [{"type": "right_angle", "points": ["A", "C", "B"]}]

    assert assumptions_from_marks(marks) == ["right_angle A C B"]


def test_assumptions_from_objects_extracts_triangle():
    objects = ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"]

    assert assumptions_from_objects(objects) == ["triangle A B C"]


def test_merge_assumptions_preserves_order_and_removes_duplicates():
    merged = merge_assumptions(["triangle A B C", "AB = AC"], ["AB = AC", "triangle A B C"])

    assert merged == ["triangle A B C", "AB = AC"]


from diagram_theorem_assistant.assumption_extraction import (
    assumptions_from_relations,
    infer_assumptions,
)


def test_assumptions_from_relations_handles_known_relations():
    relations = [
        "triangle A B C",
        "perpendicular AC BC",
        "parallel l m",
        "collinear A B C",
    ]
    assumptions = assumptions_from_relations(relations)
    assert "triangle A B C" in assumptions
    assert "parallel l m" in assumptions
    assert "right_angle A C B" in assumptions  # from perpendicular at C
    assert "collinear A B C" in assumptions


def test_infer_assumptions_accepts_relations_argument():
    result = infer_assumptions(
        problem_text="",
        objects=["point A", "point B", "point C"],
        diagram_marks=[],
        confirmed_assumptions=None,
        relations=["triangle A B C"],
    )
    assert "triangle A B C" in result

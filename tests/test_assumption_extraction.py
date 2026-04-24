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

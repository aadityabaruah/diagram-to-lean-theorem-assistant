from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates


def test_extract_goal_after_prove_phrase():
    text = "In triangle ABC, AB = AC. Prove angle ABC = angle BCA."

    assert extract_goal_from_text(text) == "angle ABC = angle BCA"


def test_extract_goal_after_show_that_phrase():
    text = "Let D be the midpoint of AB. Show that AD = DB."

    assert extract_goal_from_text(text) == "AD = DB"


def test_rank_goal_candidates_for_isosceles_triangle_objects():
    objects = ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"]
    assumptions = ["triangle A B C", "AB = AC"]

    candidates = rank_goal_candidates(objects=objects, assumptions=assumptions)

    assert candidates[0] == "angle ABC = angle BCA"
    assert "AB = AC" not in candidates


def test_rank_goal_candidates_for_ambiguous_triangle_keeps_multiple_options():
    objects = ["point A", "point B", "point C", "segment AB", "segment AC", "segment BC"]
    assumptions = ["triangle A B C"]

    candidates = rank_goal_candidates(objects=objects, assumptions=assumptions)

    assert "AB = AC" in candidates
    assert "angle ABC = angle BCA" in candidates
    assert len(candidates) >= 3

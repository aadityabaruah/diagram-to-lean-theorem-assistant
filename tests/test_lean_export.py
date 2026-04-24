from diagram_theorem_assistant.lean_export import theorem_skeleton


def test_theorem_skeleton_includes_name_assumptions_and_goal():
    result = theorem_skeleton(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )

    assert "theorem isosceles_base_angles" in result
    assert "-- assumption: triangle A B C" in result
    assert "-- assumption: AB = AC" in result
    assert "-- goal: angle ABC = angle BCA" in result
    assert ":= by" in result

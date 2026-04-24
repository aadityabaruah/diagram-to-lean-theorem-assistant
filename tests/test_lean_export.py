from diagram_theorem_assistant.lean_export import theorem_from


def test_theorem_from_imports_basic_and_uses_namespace():
    source = theorem_from(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )
    assert source.startswith("import DiagramTheorems.Basic")
    assert "namespace DiagramTheorems.Generated" in source
    assert "theorem isosceles_base_angles" in source


def test_theorem_from_emits_real_hypotheses_for_isosceles():
    source = theorem_from(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )
    # Real point variables bound, not just mentioned in comments
    assert "(A B C : EuclideanSpace ℝ (Fin 2))" in source
    # Distance equality bound as a hypothesis, not a comment
    assert "(h0 : dist A B = dist A C)" in source
    # Goal is the real proposition, not `: True`
    assert "∠ A B C = ∠ B C A" in source
    assert ": True" not in source.split("by")[0]
    assert "sorry" in source  # proof still deferred


def test_theorem_from_emits_pythagorean_with_real_types():
    source = theorem_from(
        name="pythag",
        assumptions=["triangle A B C", "right_angle A C B"],
        goal="AB^2 = AC^2 + BC^2",
    )
    assert "(A B C : EuclideanSpace ℝ (Fin 2))" in source
    assert "(h0 : ∠ A C B = Real.pi / 2)" in source
    assert "dist A B ^ 2 = dist A C ^ 2 + dist B C ^ 2" in source
    assert ": True" not in source.split("by")[0]


def test_theorem_from_perpendicular_bisector_introduces_midpoint():
    source = theorem_from(
        name="perpbi",
        assumptions=["P on perpendicular_bisector AB"],
        goal="PA = PB",
    )
    # Introduces a midpoint variable M
    assert "midpoint" in source
    assert "dist P A = dist P B" in source


def test_theorem_from_angle_var_goal_uses_real_variables():
    source = theorem_from(
        name="altint",
        assumptions=["parallel l m", "transversal t l m"],
        goal="angle 1 = angle 2",
    )
    assert "(a1 a2 : ℝ)" in source
    assert "a1 = a2" in source
    # This case is provable via exact — no sorry
    assert "exact h_parallel_transversal" in source


def test_theorem_from_preserves_raw_assumption_comments():
    source = theorem_from(
        name="iso",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )
    assert "-- assumption: triangle A B C" in source
    assert "-- assumption: AB = AC" in source
    assert "-- goal: angle ABC = angle BCA" in source


def test_theorem_from_falls_back_to_true_for_unmapped_goal():
    source = theorem_from(
        name="demo",
        assumptions=["triangle A B C"],
        goal="some free-form goal we can't parse",
    )
    # Unknown goal falls back to True + sorry, but keeps the comment
    assert "theorem demo : True" in source
    assert "sorry" in source
    assert "-- goal: some free-form goal" in source


def test_theorem_from_empty_goal_uses_trivial():
    source = theorem_from(name="demo", assumptions=[], goal="")
    assert "theorem demo : True" in source
    assert ":= by trivial" in source


def test_theorem_from_sanitizes_identifier():
    source = theorem_from(name="bad name with spaces!", assumptions=[], goal="x")
    assert "theorem bad_name_with_spaces_" in source


def test_theorem_from_flattens_newlines_in_comments():
    source = theorem_from(name="demo", assumptions=["a\nb"], goal="c")
    assert "-- assumption: a b" in source

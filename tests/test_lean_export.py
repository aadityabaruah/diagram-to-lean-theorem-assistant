from diagram_theorem_assistant.lean_export import theorem_from


def test_theorem_from_imports_basic_module_and_uses_namespace():
    source = theorem_from(
        name="isosceles_base_angles",
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
    )
    assert source.startswith("import DiagramTheorems.Basic")
    assert "namespace DiagramTheorems.Generated" in source
    assert "theorem isosceles_base_angles" in source


def test_theorem_from_emits_assumption_and_goal_comments():
    source = theorem_from(
        name="demo",
        assumptions=["triangle A B C"],
        goal="area ABC > 0",
    )
    assert "-- assumption: triangle A B C" in source
    assert "-- goal: area ABC > 0" in source


def test_theorem_from_uses_sorry_for_unmapped_goals():
    source = theorem_from(
        name="demo",
        assumptions=[],
        goal="some unrecognized string",
    )
    assert ":= by" in source
    assert "sorry" in source


def test_theorem_from_escapes_comment_newlines():
    source = theorem_from(
        name="demo",
        assumptions=["a\nb"],
        goal="c",
    )
    assert "\\n" in source or "a b" in source  # either escaped or flattened
    assert "-- assumption:" in source


def test_theorem_from_sanitizes_identifier():
    source = theorem_from(name="bad name with spaces!", assumptions=[], goal="x")
    assert "theorem bad_name_with_spaces_" in source


def test_theorem_from_uses_true_placeholder_when_goal_is_empty():
    source = theorem_from(name="demo", assumptions=[], goal="")
    assert "theorem demo : True" in source
    assert ":= by trivial" in source

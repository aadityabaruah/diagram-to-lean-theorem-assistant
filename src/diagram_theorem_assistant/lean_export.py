from __future__ import annotations

import re


def theorem_from(name: str, assumptions: list[str], goal: str) -> str:
    """Emit Lean 4 source for a generated theorem.

    The current scope is statement-level validity: imports resolve, the
    theorem declaration type-checks, and the proof is `sorry` or `trivial`.
    Future work can replace `True` / `sorry` with real Mathlib propositions
    as the formalization mapping matures.
    """
    identifier = _lean_identifier(name)
    assumption_lines = "\n".join(f"-- assumption: {_flatten(a)}" for a in assumptions)
    goal_line = f"-- goal: {_flatten(goal)}"

    if not goal.strip():
        body = f"theorem {identifier} : True := by trivial"
    else:
        body = f"theorem {identifier} : True := by\n  -- target: {_flatten(goal)}\n  sorry"

    return (
        "import DiagramTheorems.Basic\n"
        "\n"
        "namespace DiagramTheorems.Generated\n"
        "\n"
        f"{assumption_lines}\n"
        f"{goal_line}\n"
        f"{body}\n"
        "\n"
        "end DiagramTheorems.Generated\n"
    )


def _lean_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", name)
    if not cleaned or cleaned[0].isdigit():
        return "generated_" + cleaned
    return cleaned


def _flatten(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()

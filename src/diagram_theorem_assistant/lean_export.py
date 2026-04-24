from __future__ import annotations

import re


def theorem_skeleton(name: str, assumptions: list[str], goal: str) -> str:
    assumption_comments = "\n".join(f"-- assumption: {assumption}" for assumption in assumptions)
    theorem_name = _lean_identifier(name)
    return f"""import Mathlib

{assumption_comments}
-- goal: {goal}
theorem {theorem_name} : True := by
  trivial
"""


def _lean_identifier(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", name)
    if not cleaned or cleaned[0].isdigit():
        return "generated_" + cleaned
    return cleaned

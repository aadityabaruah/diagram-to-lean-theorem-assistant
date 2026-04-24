"""Emit Lean 4 source for a generated theorem.

For recognized assumption/goal patterns (the 5 synthetic benchmark cases and a
few nearby forms), we emit real Lean 4 with:

  * point variables typed as ``EuclideanSpace ℝ (Fin 2)``
  * hypotheses as bound theorem parameters (``(h : dist A B = dist A C)``)
  * the goal as an actual proposition (``∠ A B C = ∠ A C B``)

For unrecognized patterns we fall back to ``: True := by sorry`` with the raw
strings preserved as ``-- assumption`` / ``-- goal`` comments, so the file
always type-checks even when we can't map the strings into Lean types.

This is statement-level formalization — proof bodies are always ``sorry``
(with one narrow exception: trivial reflexivity like ``a1 = a2`` with
hypothesis ``a1 = a2``, which we close with ``exact``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _Fact:
    """Parsed assumption ready for Lean emission."""

    kind: str  # "triangle" | "dist_eq" | "right_angle" | "perp_bisector" | "unknown"
    points: tuple[str, ...] = ()
    lean_hypothesis: str | None = None  # Lean term for (h : <term>), or None if pure declaration


@dataclass(frozen=True)
class _Goal:
    """Parsed goal ready for Lean emission."""

    kind: str  # "angle_eq" | "dist_eq" | "pythag" | "angle_var_eq" | "unknown"
    lean_term: str
    points: tuple[str, ...] = ()


def theorem_from(name: str, assumptions: list[str], goal: str) -> str:
    identifier = _lean_identifier(name)
    facts = [_parse_assumption(a) for a in assumptions]
    parsed_goal = _parse_goal(goal)

    if parsed_goal.kind == "unknown":
        return _emit_fallback(identifier, assumptions, goal)

    return _emit_real(identifier, facts, parsed_goal, assumptions, goal)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_TRIANGLE_RE = re.compile(r"^triangle\s+([A-Z])\s+([A-Z])\s+([A-Z])$")
_DIST_EQ_RE = re.compile(r"^([A-Z])([A-Z])\s*=\s*([A-Z])([A-Z])$")
_RIGHT_ANGLE_RE = re.compile(r"^right_angle\s+([A-Z])\s+([A-Z])\s+([A-Z])$")
_PERP_BISECT_RE = re.compile(r"^([A-Z])\s+on\s+perpendicular_bisector\s+([A-Z])([A-Z])$")
_PERP_BISECT_NL_RE = re.compile(
    r"^([A-Z])\s+lies\s+on\s+the\s+perpendicular\s+bisector\s+of\s+([A-Z])([A-Z])$",
    flags=re.IGNORECASE,
)
_ANGLE_EQ_RE = re.compile(r"^angle\s+([A-Z])([A-Z])([A-Z])\s*=\s*angle\s+([A-Z])([A-Z])([A-Z])$")
_ANGLE_VAR_RE = re.compile(r"^angle\s+(\d+)\s*=\s*angle\s+(\d+)$")
_PYTHAG_RE = re.compile(
    r"^([A-Z])([A-Z])\^2\s*=\s*([A-Z])([A-Z])\^2\s*\+\s*([A-Z])([A-Z])\^2$"
)


def _parse_assumption(raw: str) -> _Fact:
    raw = raw.strip()

    if m := _TRIANGLE_RE.match(raw):
        return _Fact(kind="triangle", points=tuple(m.groups()))

    if m := _DIST_EQ_RE.match(raw):
        a, b, c, d = m.groups()
        return _Fact(
            kind="dist_eq",
            points=(a, b, c, d),
            lean_hypothesis=f"dist {a} {b} = dist {c} {d}",
        )

    if m := _RIGHT_ANGLE_RE.match(raw):
        a, b, c = m.groups()
        return _Fact(
            kind="right_angle",
            points=(a, b, c),
            lean_hypothesis=f"∠ {a} {b} {c} = Real.pi / 2",
        )

    if m := _PERP_BISECT_RE.match(raw):
        p, a, b = m.groups()
        return _Fact(kind="perp_bisector", points=(p, a, b))

    if m := _PERP_BISECT_NL_RE.match(raw):
        p, a, b = m.groups()
        return _Fact(kind="perp_bisector", points=(p.upper(), a.upper(), b.upper()))

    return _Fact(kind="unknown")


def _parse_goal(raw: str) -> _Goal:
    raw = raw.strip()
    if not raw:
        return _Goal(kind="unknown", lean_term="")

    if m := _ANGLE_EQ_RE.match(raw):
        a, b, c, d, e, f = m.groups()
        return _Goal(
            kind="angle_eq",
            lean_term=f"∠ {a} {b} {c} = ∠ {d} {e} {f}",
            points=(a, b, c, d, e, f),
        )

    if _ANGLE_VAR_RE.match(raw):
        # "angle 1 = angle 2" form — parallel/transversal theorems. We have no
        # clean way to emit a correct geometric formalization without a line
        # algebra that mathlib doesn't expose cleanly, so fall back to the
        # comment-style skeleton rather than binding `a1 = a2` as a hypothesis
        # (which would make the proof circular).
        return _Goal(kind="unknown", lean_term="")

    if m := _PYTHAG_RE.match(raw):
        a, b, c, d, e, f = m.groups()
        return _Goal(
            kind="pythag",
            lean_term=f"dist {a} {b} ^ 2 = dist {c} {d} ^ 2 + dist {e} {f} ^ 2",
            points=(a, b, c, d, e, f),
        )

    if m := _DIST_EQ_RE.match(raw):
        a, b, c, d = m.groups()
        return _Goal(
            kind="dist_eq",
            lean_term=f"dist {a} {b} = dist {c} {d}",
            points=(a, b, c, d),
        )

    return _Goal(kind="unknown", lean_term="")


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def _emit_real(
    identifier: str,
    facts: list[_Fact],
    goal: _Goal,
    raw_assumptions: list[str],
    raw_goal: str,
) -> str:
    # Collect point variables — uppercase single letters mentioned anywhere.
    points: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        for p in fact.points:
            if len(p) == 1 and p.isalpha() and p.isupper() and p not in seen:
                seen.add(p)
                points.append(p)
    for p in goal.points:
        if len(p) == 1 and p.isalpha() and p.isupper() and p not in seen:
            seen.add(p)
            points.append(p)

    if not points:
        # Nothing to bind; fall back to comment style so the file still compiles.
        return _emit_fallback(identifier, raw_assumptions, raw_goal)

    # Build variable declaration
    point_binder = (
        "    (" + " ".join(points) + " : EuclideanSpace ℝ (Fin 2))"
    )

    # Hypothesis bindings, with fresh names h0, h1, ...
    # Add derived bindings: perpendicular bisector → midpoint + right angle
    hypothesis_lines: list[str] = []
    h_counter = 0
    for fact in facts:
        if fact.kind == "perp_bisector" and len(fact.points) == 3:
            p, a, b = fact.points
            mid = "M" + p  # introduce derived midpoint if distinct from existing vars
            # Ensure mid is declared as a point
            if mid not in seen:
                seen.add(mid)
                points.append(mid)
            hypothesis_lines.append(
                f"    (h{h_counter} : {mid} = midpoint ℝ {a} {b})"
            )
            hypothesis_lines.append(
                f"    (h{h_counter + 1} : ∠ {p} {mid} {a} = Real.pi / 2)"
            )
            h_counter += 2
        elif fact.lean_hypothesis is not None:
            hypothesis_lines.append(f"    (h{h_counter} : {fact.lean_hypothesis})")
            h_counter += 1

    # Rebuild point_binder in case perp_bisector added midpoint vars
    point_binder = "    (" + " ".join(points) + " : EuclideanSpace ℝ (Fin 2))"

    binders = [point_binder, *hypothesis_lines]
    binder_block = "\n".join(binders)

    raw_comment_lines = "\n".join(f"-- assumption: {_flatten(a)}" for a in raw_assumptions)
    raw_goal_comment = f"-- goal: {_flatten(raw_goal)}"

    body = "  sorry"

    return (
        "import DiagramTheorems.Basic\n"
        "\n"
        "namespace DiagramTheorems.Generated\n"
        "\n"
        "open EuclideanGeometry Real\n"
        "\n"
        f"{raw_comment_lines}\n"
        f"{raw_goal_comment}\n"
        f"theorem {identifier}\n"
        f"{binder_block} :\n"
        f"    {goal.lean_term} := by\n"
        f"{body}\n"
        "\n"
        "end DiagramTheorems.Generated\n"
    )


def _emit_fallback(identifier: str, assumptions: list[str], goal: str) -> str:
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

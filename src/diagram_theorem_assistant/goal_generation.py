from __future__ import annotations

import re


def extract_goal_from_text(text: str) -> str | None:
    patterns = [
        r"\bprove(?: that)?\s+(.+?)(?:\.|$)",
        r"\bshow(?: that)?\s+(.+?)(?:\.|$)",
        r"\bfind\s+(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_goal(match.group(1).strip())
    return None


def rank_goal_candidates(objects: list[str], assumptions: list[str]) -> list[str]:
    object_set = set(objects)
    assumption_set = set(assumptions)
    candidates: list[str] = []

    if "triangle A B C" in assumption_set and "AB = AC" in assumption_set:
        candidates.append("angle ABC = angle BCA")

    if "right_angle A C B" in assumption_set:
        candidates.append("AB^2 = AC^2 + BC^2")

    if "P on perpendicular_bisector AB" in assumption_set:
        candidates.append("PA = PB")

    if "parallel l m" in assumption_set and {"angle 1", "angle 2"} <= object_set:
        candidates.append("angle 1 = angle 2")

    if "triangle A B C" in assumption_set:
        candidates.extend(
            [
                "AB = AC",
                "angle ABC = angle BCA",
                "angle BAC + angle ABC + angle BCA = 180",
                "area ABC > 0",
            ]
        )

    return _dedupe(candidate for candidate in candidates if candidate not in assumption_set)


def _normalize_goal(goal: str) -> str:
    return re.sub(r"\s+equals\s+", " = ", goal, flags=re.IGNORECASE).strip()


def _dedupe(candidates: object) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped

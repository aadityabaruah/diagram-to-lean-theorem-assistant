from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def assumptions_from_text(text: str) -> list[str]:
    assumptions: list[str] = []
    triangle = _extract_triangle(text)

    if triangle is not None:
        assumptions.append(_triangle_assumption(triangle))

    for left, right in re.findall(r"\b([A-Z]{2})\s*=\s*([A-Z]{2})\b", text):
        assumptions.append(f"{left} = {right}")

    right_vertex = _extract_right_angle_vertex(text)
    if triangle is not None and right_vertex in triangle:
        others = [point for point in triangle if point != right_vertex]
        assumptions.append(f"right_angle {others[0]} {right_vertex} {others[1]}")

    perpendicular_bisector = re.search(
        r"\bPoint\s+([A-Z])\s+lies\s+on\s+the\s+perpendicular\s+bisector\s+of\s+([A-Z]{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if perpendicular_bisector:
        point, segment = perpendicular_bisector.groups()
        assumptions.append(f"{point.upper()} on perpendicular_bisector {segment.upper()}")

    parallel = re.search(
        r"\bLines?\s+([a-z])\s+and\s+([a-z])\s+are\s+parallel\b",
        text,
        flags=re.IGNORECASE,
    )
    transversal = re.search(r"\btransversal\s+([a-z])\b", text, flags=re.IGNORECASE)
    if parallel:
        line_a, line_b = parallel.groups()
        assumptions.append(f"parallel {line_a} {line_b}")
        if transversal:
            assumptions.append(f"transversal {transversal.group(1)} {line_a} {line_b}")

    return merge_assumptions(assumptions)


def assumptions_from_marks(marks: list[dict[str, Any]]) -> list[str]:
    assumptions: list[str] = []
    for mark in marks:
        mark_type = mark.get("type")
        if mark_type == "right_angle" and len(mark.get("points", [])) == 3:
            assumptions.append("right_angle " + " ".join(mark["points"]))
        elif mark_type == "equal_length" and len(mark.get("segments", [])) == 2:
            left, right = mark["segments"]
            assumptions.append(f"{left} = {right}")
        elif mark_type == "parallel" and len(mark.get("lines", [])) == 2:
            left, right = mark["lines"]
            assumptions.append(f"parallel {left} {right}")
        elif mark_type == "perpendicular_bisector" and mark.get("point") and mark.get("segment"):
            assumptions.append(f"{mark['point']} on perpendicular_bisector {mark['segment']}")
    return merge_assumptions(assumptions)


def assumptions_from_objects(objects: list[str]) -> list[str]:
    points = sorted(_object_suffixes(objects, "point"))
    segments = set(_object_suffixes(objects, "segment"))
    assumptions: list[str] = []

    if len(points) >= 3:
        for index, first in enumerate(points):
            for second in points[index + 1 :]:
                for third in points[index + 2 :]:
                    required = {first + second, first + third, second + third}
                    if required <= segments:
                        assumptions.append(_triangle_assumption([first, second, third]))

    return merge_assumptions(assumptions)


def infer_assumptions(
    problem_text: str,
    objects: list[str],
    diagram_marks: list[dict[str, Any]],
    confirmed_assumptions: list[str] | None = None,
    relations: list[str] | None = None,
) -> list[str]:
    return merge_assumptions(
        confirmed_assumptions or [],
        assumptions_from_relations(relations or []),
        assumptions_from_objects(objects),
        assumptions_from_marks(diagram_marks),
        assumptions_from_text(problem_text),
    )


def assumptions_from_relations(relations: list[str]) -> list[str]:
    assumptions: list[str] = []
    for relation in relations:
        tokens = relation.split()
        if not tokens:
            continue
        head = tokens[0]
        if head == "triangle" and len(tokens) == 4:
            assumptions.append(relation)
        elif head == "parallel" and len(tokens) == 3:
            assumptions.append(relation)
        elif head == "collinear" and len(tokens) == 4:
            assumptions.append(relation)
        elif head == "perpendicular" and len(tokens) == 3:
            # "perpendicular AC BC" → right angle at the shared vertex
            assumptions.append(relation)
            assumptions.extend(_right_angle_from_perpendicular(tokens[1], tokens[2]))
        elif head == "incidence" and len(tokens) == 3:
            assumptions.append(relation)
        elif head == "transversal" and len(tokens) == 4:
            assumptions.append(relation)
    return merge_assumptions(assumptions)


def _right_angle_from_perpendicular(seg_a: str, seg_b: str) -> list[str]:
    if len(seg_a) == 2 and len(seg_b) == 2:
        shared = set(seg_a) & set(seg_b)
        if len(shared) == 1:
            vertex = next(iter(shared))
            other_a = (set(seg_a) - {vertex}).pop()
            other_b = (set(seg_b) - {vertex}).pop()
            return [f"right_angle {other_a} {vertex} {other_b}"]
    return []


def merge_assumptions(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for assumption in group:
            if assumption not in seen:
                merged.append(assumption)
                seen.add(assumption)
    return merged


def _extract_triangle(text: str) -> list[str] | None:
    match = re.search(r"\btriangle\s+([A-Z]{3})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return list(match.group(1).upper())


def _extract_right_angle_vertex(text: str) -> str | None:
    match = re.search(r"\bright angle at\s+([A-Z])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _object_suffixes(objects: list[str], prefix: str) -> list[str]:
    marker = prefix + " "
    return [item.removeprefix(marker) for item in objects if item.startswith(marker)]


def _triangle_assumption(points: list[str]) -> str:
    return "triangle " + " ".join(points)

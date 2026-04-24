"""Prompt templates for the three-stage multi-step VLM extraction.

Each call to Gemini is narrowly scoped to reduce hallucination: the model
answers one focused question at a time and returns JSON in a known shape.
"""
from __future__ import annotations

OBJECTS_PROMPT = """You are analyzing a geometry diagram.

List every NAMED geometric object you can see. Use lowercase kind followed
by the capitalized name. Examples: "point A", "segment AB", "line l",
"circle O", "angle 1".

Do NOT invent objects that are not labeled. If a point has no visible
label, do not include it.

Respond with ONLY valid JSON of the form:
{"objects": ["point A", "segment AB", ...]}
"""

RELATIONS_PROMPT = """You are analyzing a geometry diagram.

The following named objects have been identified:
{objects}

List the GEOMETRIC RELATIONS directly visible in the diagram. Use these
formats:
- "triangle A B C"                  (three points form a triangle)
- "collinear A B C"                 (three points on a line)
- "parallel l m"                    (two lines are parallel)
- "perpendicular AB AC"             (two segments/lines are perpendicular)
- "transversal t l m"               (t cuts lines l and m)
- "incidence P AB"                  (point P lies on segment AB)

Do NOT infer relations that are not visually indicated. If unsure, omit.

Respond with ONLY valid JSON of the form:
{{"relations": ["triangle A B C", ...]}}
"""

MARKS_PROMPT = """You are analyzing a geometry diagram.

List the VISUAL MARKS drawn on the diagram (tick marks, arrows, right-angle
boxes, etc). Use these shapes:

- {"type": "right_angle", "points": ["A", "C", "B"]}       (right-angle box at C)
- {"type": "equal_length", "segments": ["AB", "AC"]}       (tick marks showing equal segments)
- {"type": "parallel", "lines": ["l", "m"]}                (parallel arrows)
- {"type": "perpendicular_bisector", "point": "P", "segment": "AB"}

Only include marks that are actually drawn. Do not infer marks from object
positions.

Respond with ONLY valid JSON of the form:
{"marks": [...]}
"""

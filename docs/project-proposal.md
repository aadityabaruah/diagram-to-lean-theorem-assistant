# Diagram-to-Lean Theorem Assistant

## Research Question

Can a system help translate a geometry diagram and optional problem text into an explicit Lean theorem by proposing candidate assumptions, proposing candidate goals, and asking the user to resolve ambiguous choices?

The project is not claiming that a diagram uniquely determines a theorem. Instead, it treats theorem identification as an interactive ranking and verification problem.

## How The Goal Is Determined

The system supports two goal modes.

In text-guided mode, the problem statement contains language such as "prove that", "show that", or "find". The system extracts the clause after that phrase and converts it into a formal goal candidate.

In diagram-only mode, the system cannot know the intended theorem with certainty. It generates a ranked list of plausible goals from common geometry templates, such as equal angles, equal lengths, collinearity, parallelism, perpendicularity, congruent triangles, and similarity. The user then selects or edits the intended goal.

This design directly addresses the ambiguity in the original example: the system does not silently invent a hidden goal. It either extracts the goal from text or exposes multiple candidate goals for review.

## How Assumptions Are Determined

The system separates assumptions from goals. Assumptions describe what is given or inferred from the diagram, while the goal describes what should be proven.

Candidate assumptions come from three sources:

- Explicit text, such as "AB = AC" or "DE is parallel to BC".
- Diagram marks, such as equal-length tick marks, right-angle boxes, parallel arrows, labeled points, and intersection structure.
- Conservative geometric defaults, such as detected collinearity or segment incidence.

Because diagrams are often under-specified, assumptions are never treated as final automatically. The interface asks the user to confirm, reject, or add assumptions before the Lean theorem is produced.

## Evaluation Plan

The project will use a curated benchmark of geometry examples. Each example will contain a diagram, optional problem text, gold assumptions, a gold goal theorem, and acceptable alternative theorem statements.

The system will be evaluated with four metrics:

- Assumption precision and recall: did it identify the intended given facts?
- Top-k goal accuracy: was the correct goal among the top 1, top 3, or top 5 generated candidates?
- Lean statement validity: did the generated theorem statement type-check in Lean?
- Proof success or skeleton quality: could Lean complete the proof automatically, or did the system produce a useful theorem skeleton?

This benchmark makes it possible to measure whether the system is finding the "right" theorem instead of relying on anecdotal examples.

## Comparison With Lean Blueprint

Lean Blueprint is mainly a project-management and exposition tool for formalization projects. It helps connect informal mathematical explanations with Lean theorem dependencies.

This project operates earlier in the pipeline. It starts from a visual geometry problem and tries to produce explicit assumptions and candidate theorem goals. Lean is then used to verify the formal statement.

The two tools are complementary. A mature version of this project could export verified theorem statements, dependency graphs, and explanation text into a blueprint-style document. The visualizer could also borrow the blueprint idea of showing which definitions and lemmas are needed for each theorem.

# Evaluation Plan

The benchmark contains geometry problems with known assumptions and goals. Each example includes the diagram path, optional problem text, gold assumptions, the gold goal theorem, acceptable alternative goals, and the Lean theorem name.

## Example Categories

- Text-guided examples where the goal is explicitly stated.
- Diagram-guided examples where visual marks suggest assumptions.
- Ambiguous diagram-only examples where several goals are plausible.
- Negative examples where the system should ask for clarification instead of guessing.

## Metrics

- Assumption precision: predicted intended assumptions divided by all predicted assumptions.
- Assumption recall: predicted intended assumptions divided by all gold assumptions.
- Top-k goal accuracy: whether the gold goal appears in the top 1, top 3, or top 5 generated goals.
- Lean type-check rate: percentage of generated theorem statements accepted by Lean.
- Proof completion rate: percentage of theorem statements proved automatically or with a short generated proof.

## Success Criteria For The Class Prototype

The prototype is successful if it loads benchmark examples, produces assumption candidates, produces ranked goal candidates, exports a Lean theorem skeleton, and reports evaluation metrics on at least five examples.

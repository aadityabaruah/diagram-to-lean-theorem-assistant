# Diagram-to-Lean Theorem Assistant

This project explores how to translate geometry diagrams and optional problem text into explicit Lean theorem candidates.

The prototype does three things:

- proposes candidate assumptions from text and diagram marks,
- proposes candidate theorem goals instead of assuming the goal is obvious,
- exports selected assumptions and goals as a Lean theorem skeleton.

The project treats diagram ambiguity as a central problem. When a diagram does not uniquely determine the intended theorem, the system exposes ranked goals and a clarification flag instead of silently inventing a hidden target.

## Quick Start

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m diagram_theorem_assistant.demo
```

See `docs/project-proposal.md` for the full proposal and `docs/evaluation-plan.md` for the benchmark plan.

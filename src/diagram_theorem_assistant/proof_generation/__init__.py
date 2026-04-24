"""LLM-based proof generation: replace `sorry` with a real Lean proof by
iterating `lake build` errors back to the model until it closes or max
attempts are exhausted.
"""

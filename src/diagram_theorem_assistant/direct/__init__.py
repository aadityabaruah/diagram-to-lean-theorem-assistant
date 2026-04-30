"""Direct end-to-end Claude solver.

Bypasses the multi-stage pipeline (VLM extraction → assumption inference →
goal generation → Lean emission → proof) and asks Claude — with vision —
to produce a complete Lean 4 file in one round, then verifies via lake
build with retry on errors.
"""

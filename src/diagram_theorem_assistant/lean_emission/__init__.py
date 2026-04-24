"""Fully LLM-based Lean 4 file emission.

Unlike the deterministic templated emitter in ``lean_export.py``, modules
here delegate the entire theorem-file generation to an LLM. The LLM chooses
the hypothesis types, the goal proposition, and the proof (or leaves
``sorry``) — producing real Lean rather than stub-with-comments. Each
attempt is verified by ``lake build``; errors are fed back to the LLM for
another round.
"""

"""ClaudeFormalizer — fully LLM-based Lean 4 file generation.

Given a ``DiagramReading`` and an ``Extraction``, asks Claude to produce a
complete Lean 4 file with real hypothesis types, a real goal proposition,
and a proof (or honest ``sorry``). The output is verified with ``lake
build``; on failure, the build error is fed back to Claude for another
round, up to ``max_attempts``.

This is the alternative to ``lean_export.theorem_from``'s deterministic
templates. Use it when you want the *statement* itself — not just the
proof — formalized by the model, so hypotheses land as bound Lean terms
instead of ``-- comments`` and the goal is a real proposition rather than
``: True``.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import DiagramReading, LeanStatus
from diagram_theorem_assistant.vlm.base import VLMError

RETRY_BACKOFF_SEC = 2.0

_RETRYABLE = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "TimeoutError",
        "RemoteDisconnected",
        "ServerError",
        "ServiceUnavailableError",
        "InternalServerError",
    }
)


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE


def _default_model() -> str:
    return os.environ.get("CLAUDE_FORMALIZER_MODEL", "claude-opus-4-7")


FORMALIZE_PROMPT = """You are a Lean 4 + mathlib4 expert. Produce a complete Lean
4 source file that formalizes the geometry problem described below.

The file must have:
1. `import DiagramTheorems.Basic` (which transitively imports
   `Mathlib.Geometry.Euclidean.Basic`, `Mathlib.Geometry.Euclidean.Triangle`,
   and `Mathlib.Tactic`). Add any other imports you need.
2. `namespace DiagramTheorems.Generated ... end DiagramTheorems.Generated`.
3. `open EuclideanGeometry Real` if you use angle notation or `π`.
4. A single theorem named `{name}` whose:
   - Hypotheses are REAL Lean bindings with mathlib-appropriate types, not
     `-- comments`. Examples:
       * Points: `(A B C : EuclideanSpace ℝ (Fin 2))`
       * A right angle at C: `(h : ∠ A C B = Real.pi / 2)`
       * Two equal segments: `(h : dist A B = dist A C)`
       * Collinearity of three points: `(h : Collinear ℝ ({{A, B, C}} : Set _))`
       * Midpoint: `(h : M = midpoint ℝ A B)`
   - Goal is a REAL Lean proposition matching the math — NEVER `: True`.
     Examples:
       * `∠ A B C = ∠ B C A`
       * `dist A B ^ 2 = dist A C ^ 2 + dist B C ^ 2`
       * `dist P A = dist P B`
5. A proof. Try the following strategies in order:
   a. Named mathlib lemmas (`EuclideanGeometry.law_cos`, `angle_comm`, etc.)
   b. Automated tactics: `simp_all`, `nlinarith`, `polyrith`, `linarith`,
      `ring`, `field_simp`, `norm_num`, `positivity`.
   c. Explicit constructions via `InnerProductGeometry.angle_eq_iff` or
      `dist_eq_norm_vsub`.
   d. If none work, leave a single `sorry` — but the theorem STATEMENT
      must still be faithful to the math.

ABSOLUTE RULES — violation makes the output worthless:

1. **Never use `: True` as the theorem type.** If the math says "prove
   angles 1 and 2 are equal", the goal must be an angle-equality
   proposition, not `True`.

2. **Never bind the goal (or anything definitionally equal to it) as a
   hypothesis.** A hypothesis `(h : ∠ A B C = ∠ D E F)` with goal
   `∠ A B C = ∠ D E F` is circular reasoning — the "proof" `exact h`
   assumes what it claims to prove. This is FORBIDDEN even if you cannot
   find the real geometric hypotheses.

3. **If you cannot honestly formalize the geometric givens** (e.g.,
   parallel lines without a line algebra, transversal relationships), do
   ONE of these — never both:
   a. State the theorem with whatever hypotheses you CAN formalize
      honestly (e.g., just declare points and use `sorry` for proof),
      accepting that the file may not correspond to the full problem.
   b. If even step (a) is impossible, return a theorem with NO
      hypotheses and goal `: True := by trivial`, and include an
      explanatory comment explaining what could not be formalized. This
      is the only time `: True` is acceptable.

4. Respond with ONLY the complete file source. No `<think>` blocks, no
   markdown fences, no commentary before or after. The entire response
   must be valid Lean 4 source that `lake build` can compile.

Self-check before responding: does your hypothesis list contain the goal
or any proposition that simplifies to the goal? If yes, delete it.

Problem details:
- Diagram objects:   {objects}
- Diagram relations: {relations}
- Diagram marks:     {marks}
- Assumptions (from VLM + text): {assumptions}
- Goal (from VLM + text):        {goal}
- Theorem name:                  {name}
"""


RETRY_PROMPT = """Your previous attempt produced this `lake build` error:

```
{error}
```

Previous attempt:

```lean
{previous}
```

Fix the file and return the COMPLETE corrected Lean source. Same rules as
before — no markdown fences, no commentary, no `<think>` blocks. If the
hypothesis types or imports are wrong, fix them. If a lemma name doesn't
exist, replace it with one that does (or with `sorry` if you genuinely
can't find the right lemma). Never fall back to `: True`.
"""


class ClaudeFormalizer:
    """Generate a complete Lean 4 file from a reading + extraction via LLM."""

    def __init__(
        self, api_key: str, *, client: Any | None = None, model: str | None = None
    ) -> None:
        if not api_key:
            raise VLMError("ANTHROPIC_API_KEY is empty for ClaudeFormalizer.")
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def formalize(
        self,
        *,
        reading: DiagramReading,
        extraction: Extraction,
        theorem_name: str,
        lean_runner: LeanRunner,
        max_attempts: int = 5,
    ) -> tuple[str, LeanStatus, str | None]:
        """Produce a verified Lean file.

        Returns ``(source, status, stderr)``. If any attempt builds cleanly
        (``LeanStatus.OK``), that source is returned. Otherwise the best
        candidate (the last attempt) is returned with its build status.
        """
        previous_source = ""
        previous_error = ""
        last_status = LeanStatus.UNAVAILABLE
        last_stderr: str | None = None

        for attempt in range(max_attempts):
            if attempt == 0:
                prompt = FORMALIZE_PROMPT.format(
                    name=theorem_name,
                    objects=json.dumps(reading.objects),
                    relations=json.dumps(reading.relations),
                    marks=json.dumps([m.to_dict() for m in reading.marks]),
                    assumptions=json.dumps(extraction.assumptions),
                    goal=json.dumps(extraction.goal),
                )
            else:
                prompt = RETRY_PROMPT.format(
                    error=previous_error[:2500],
                    previous=previous_source[:4500],
                )

            candidate = _strip_fences(self._invoke(prompt))
            if not candidate.strip() or "theorem" not in candidate:
                continue

            # Reject circular proofs before invoking lake build.
            if _is_circular(candidate):
                previous_source = candidate
                previous_error = (
                    "Your theorem binds the goal (or an equivalent proposition) "
                    "as a hypothesis and closes with `exact`. That is circular "
                    "reasoning. Remove that hypothesis. Use real geometric "
                    "hypotheses or fall back to an empty-hypothesis `: True` "
                    "theorem with an explanatory comment."
                )
                last_status = LeanStatus.TYPE_ERROR
                last_stderr = previous_error
                continue

            status, stderr = lean_runner.typecheck(candidate)
            previous_source = candidate
            previous_error = stderr or "(no stderr)"
            last_status = status
            last_stderr = stderr

            if status is LeanStatus.OK:
                return candidate, status, stderr

        return previous_source, last_status, last_stderr

    def _invoke(self, prompt: str) -> str:
        def _gen():
            return self._client.messages.create(
                model=self._model,
                max_tokens=20000,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            )

        try:
            msg = _gen()
        except Exception as first:  # noqa: BLE001
            if not _is_retryable(first):
                raise VLMError(f"Claude formalizer call failed: {first}") from first
            time.sleep(RETRY_BACKOFF_SEC)
            try:
                msg = _gen()
            except Exception as second:  # noqa: BLE001
                raise VLMError(
                    f"Claude formalizer call failed after retry: {second}"
                ) from second
        return msg.content[0].text if msg.content else ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_LEAN_RE = re.compile(r"```(?:lean4?|)\s*\n(.*?)\n```", re.DOTALL)

_THEOREM_HEAD_RE = re.compile(r"theorem\s+\S+", re.DOTALL)


def _is_circular(source: str) -> bool:
    """Detect whether a theorem binds its goal (or an α-equivalent) as a hypothesis.

    Parses the theorem header by depth-tracking parentheses and braces so
    that ``:`` inside binders doesn't confuse the goal/binder split. Then
    normalizes whitespace and compares each hypothesis type against the
    goal type.
    """
    match = _THEOREM_HEAD_RE.search(source)
    if not match:
        return False
    # Slice from after "theorem NAME" to the ":=" that opens the proof.
    tail = source[match.end():]
    body_end = tail.find(":=")
    if body_end < 0:
        return False
    header = tail[:body_end]
    # Walk the header and find the top-level ':' (depth 0) that separates
    # binders from the goal. There may be multiple ':' inside binders.
    depth = 0
    goal_colon_index = -1
    for i, ch in enumerate(header):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            goal_colon_index = i
    if goal_colon_index < 0:
        return False
    binders = header[:goal_colon_index]
    goal = header[goal_colon_index + 1:]
    goal_norm = re.sub(r"\s+", " ", goal).strip()
    if not goal_norm:
        return False
    # Pull out each top-level `(name : type)` binder (non-nested).
    for hyp_type in re.findall(r"\(\s*\w+(?:\s+\w+)*\s*:\s*([^()]+?)\)", binders):
        hyp_norm = re.sub(r"\s+", " ", hyp_type).strip()
        if hyp_norm == goal_norm:
            return True
    return False


def _strip_fences(text: str) -> str:
    cleaned = _THINK_RE.sub("", text).strip()
    match = _FENCED_LEAN_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        return "\n".join(line for line in lines if not line.startswith("```")).strip()
    return cleaned

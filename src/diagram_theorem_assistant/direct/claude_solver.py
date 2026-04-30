"""ClaudeDirectSolver — image → complete Lean file in one (iterated) call.

The simplest possible architecture: pass the diagram image plus optional
problem text directly to Claude (multimodal), ask for a complete Lean 4
file with hypotheses, goal, and proof, then verify with ``lake build``.
On failure, feed the error back and try again.

No regex extraction, no multi-provider consensus, no ``sorry`` fallback —
either Claude returns a verified proof or we surface the last error.
"""
from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any

from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus
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
    return os.environ.get("CLAUDE_DIRECT_MODEL", "claude-opus-4-7")


SOLVE_PROMPT = """You are a Lean 4 + mathlib4 expert with vision. Look at the
geometry diagram I'm sending and produce a COMPLETE, VERIFIED Lean 4 file
that formalizes the EXACT theorem stated below and proves it.

{problem_clause}

ABSOLUTE RULES (violation = unacceptable output):

1. The goal of the theorem MUST be the literal mathematical claim from the
   problem text — not a weaker, easier-to-prove restatement. For example:
   - If the problem says "prove ∠ABC = ∠BCA", the goal MUST be
     `∠ A B C = ∠ B C A` (an angle equality), NOT `dist B A = dist C A`
     (a distance equality), NOT something derivable trivially from the
     hypotheses by `dist_comm` or `rfl`.
   - If the problem says "prove AB² = AC² + BC²", the goal MUST involve
     squared distances, not norms or scalar identities.

2. The hypotheses must be the geometric givens from the problem — not
   the conclusion in disguise. Do not bind the goal (or anything
   definitionally equal to it) as a hypothesis. That's circular.

3. NEVER use `: True` as the theorem type. NEVER use `sorry` in the proof.
   If you cannot find a proof, return your best partial work but do NOT
   weaken the goal to `True` or to a trivial restatement.

4. Single theorem named `{theorem_name}`. Start with `import DiagramTheorems.Basic`,
   wrap in `namespace DiagramTheorems.Generated ... end`, `open EuclideanGeometry Real`.

Tactical hints:
- `EuclideanGeometry.law_cos` for triangle side/angle relations:
  `dist p1 p3 * dist p1 p3 = dist p1 p2 * dist p1 p2 + dist p3 p2 * dist p3 p2 -
  2 * dist p1 p2 * dist p3 p2 * Real.cos (∠ p1 p2 p3)`
- For angle equality: apply law_cos at both relevant vertices, substitute
  the hypotheses to get equal cos values, then conclude via `Real.injOn_cos`
  on `Set.Icc 0 π` with `EuclideanGeometry.angle_nonneg` and `angle_le_pi`.
- `linear_combination` for combining equations.
- `dist_eq_zero` for degenerate cases. Wrap in `by_cases` if needed.
- For Pythagoras: `law_cos` + `Real.cos_pi_div_two = 0` + `nlinarith`.

Respond with ONLY the complete Lean source. No `<think>` tags, no markdown
fences, no commentary, no explanation. Just the Lean code that compiles
under `lake build`.
"""


FORMALIZE_PROMPT = """You are a Lean 4 + mathlib4 expert with vision. Look at
the geometry diagram I'm sending and produce a Lean 4 file that FORMALIZES
the theorem statement faithfully — proper hypothesis types and a real
proposition for the goal. The proof body can be `sorry`; what matters is
that the theorem signature precisely captures the math.

{problem_clause}

ABSOLUTE RULES:

1. The goal of the theorem MUST be the literal mathematical claim from the
   problem text — not a weaker, trivially-derivable restatement.
   - "prove ∠ABC = ∠BCA" → goal MUST be `∠ A B C = ∠ B C A` (use `∠`)
   - "prove AB² = AC² + BC²" → goal MUST involve squared distances
2. Hypotheses must be the geometric givens. Never bind the goal as a
   hypothesis (circular). Skip a hypothesis if you cannot formalize it
   cleanly — but do not fabricate one that trivializes the proof.
3. NEVER use `: True` as the theorem type.
4. The proof body is `by sorry` — do not attempt the actual proof. We
   want the SIGNATURE right, the proof comes later.
5. Single theorem named `{theorem_name}`. `import DiagramTheorems.Basic`,
   `namespace DiagramTheorems.Generated ... end`, `open EuclideanGeometry Real`.

Type hints (use what fits):
- Points in 2D: `(A B C : EuclideanSpace ℝ (Fin 2))`
- Or fully general: `{{V P : Type*}} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  [MetricSpace P] [NormedAddTorsor V P]` then `(A B C : P)`
- Distance: `dist A B`. Angle: `∠ A B C`. Right angle: `Real.pi / 2`.
- Midpoint: `M = midpoint ℝ A B`.

Respond with ONLY the complete Lean source. No `<think>` blocks, no markdown
fences, no commentary.
"""


RETRY_PROMPT = """Your previous Lean file produced this `lake build` error:

```
{error}
```

Your previous attempt:

```lean
{previous}
```

Fix it. Same rules as before — return ONLY the complete Lean source. NEVER
fall back to `sorry`. NEVER use `: True`. NEVER bind the goal as a hypothesis.
"""


class ClaudeDirectSolver:
    """Image-to-verified-Lean in one pipeline stage."""

    def __init__(
        self, api_key: str, *, client: Any | None = None, model: str | None = None
    ) -> None:
        if not api_key:
            raise VLMError("ANTHROPIC_API_KEY is empty for ClaudeDirectSolver.")
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def solve(
        self,
        *,
        image_path: Path,
        problem_text: str,
        theorem_name: str,
        lean_runner: LeanRunner,
        max_attempts: int = 5,
        allow_sorry: bool = False,
    ) -> tuple[str, LeanStatus, str | None]:
        """Send image + prompt to Claude, verify with lake build, retry on errors.

        Returns ``(source, status, stderr)``. If any attempt builds cleanly,
        passes the goal-weakening guard, and (when ``allow_sorry=False``)
        contains no ``sorry``, that source is returned. Otherwise the last
        attempt is returned with its failing status.

        With ``allow_sorry=True`` we treat the task as "formalize the
        theorem statement faithfully" — Claude can leave ``sorry`` for the
        proof body. The hypothesis-and-goal correctness checks still apply.
        """
        image_bytes = Path(image_path).read_bytes()
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

        problem_clause = (
            f"Problem text (may help — if it conflicts with the image, trust the image):\n  "
            f"{problem_text.strip()}"
            if problem_text.strip()
            else "No problem text provided. Infer the theorem from the diagram alone."
        )

        previous_source = ""
        previous_error = ""
        last_status = LeanStatus.UNAVAILABLE
        last_stderr: str | None = None

        primary_prompt = (
            FORMALIZE_PROMPT if allow_sorry else SOLVE_PROMPT
        )

        for attempt in range(max_attempts):
            if attempt == 0:
                user_text = primary_prompt.format(
                    problem_clause=problem_clause,
                    theorem_name=theorem_name,
                )
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ]
            else:
                user_text = RETRY_PROMPT.format(
                    error=previous_error[:3000],
                    previous=previous_source[:6000],
                )
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ]

            candidate = _strip_fences(self._invoke(content))
            if not candidate.strip() or "theorem" not in candidate:
                previous_error = "Response did not contain a Lean theorem."
                continue

            status, stderr = lean_runner.typecheck(candidate)
            previous_source = candidate
            previous_error = stderr or "(empty stderr)"
            last_status = status
            last_stderr = stderr

            if status is LeanStatus.OK:
                # Whether or not sorry is allowed, the goal must not be weakened.
                weak_reason = _weakened_goal_reason(candidate, problem_text)
                if weak_reason is not None:
                    previous_error = (
                        f"Your previous response compiled but the goal was weakened: "
                        f"{weak_reason}. The goal MUST be the literal claim from "
                        f"the problem text, not a trivial restatement of the "
                        f"hypothesis. Re-read the problem and emit the actual theorem."
                    )
                    last_status = LeanStatus.TYPE_ERROR
                    last_stderr = previous_error
                    continue

                if allow_sorry or "sorry" not in candidate:
                    return candidate, status, stderr

                # allow_sorry is False and the candidate has sorry — nudge
                previous_error = (
                    "Your previous response compiled but contained `sorry`. "
                    "That violates the rules. Replace `sorry` with a real proof. "
                    "Use the tactical hints from the original instructions."
                )

        return previous_source, last_status, last_stderr

    def _invoke(self, content: list[dict[str, Any]]) -> str:
        def _gen():
            return self._client.messages.create(
                model=self._model,
                max_tokens=20000,
                messages=[{"role": "user", "content": content}],
            )

        try:
            msg = _gen()
        except Exception as first:  # noqa: BLE001
            if not _is_retryable(first):
                raise VLMError(f"Claude direct-solver call failed: {first}") from first
            time.sleep(RETRY_BACKOFF_SEC)
            try:
                msg = _gen()
            except Exception as second:  # noqa: BLE001
                raise VLMError(
                    f"Claude direct-solver call failed after retry: {second}"
                ) from second
        # Concatenate all text blocks (in case extended-thinking emits separate blocks).
        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_LEAN_RE = re.compile(r"```(?:lean4?|)\s*\n(.*?)\n```", re.DOTALL)


def _weakened_goal_reason(source: str, problem_text: str) -> str | None:
    """Heuristic check that the source's goal matches the problem's intent.

    Returns a human-readable reason string when the goal looks weakened,
    otherwise None. Triggers on:
      * problem mentions ``angle`` but Lean goal is just ``dist`` equality
      * problem mentions ``^2`` (Pythagoras) but Lean goal lacks ``^ 2``
    Conservative — only flags clear regressions, not stylistic differences.
    """
    pt = problem_text.lower()
    # Find the theorem goal: substring between the last top-level `:` and `:= by`
    m = re.search(r"theorem\s+\S+([\s\S]*?):=\s*by", source)
    if not m:
        return None
    header = m.group(1)
    depth = 0
    last_colon = -1
    for i, ch in enumerate(header):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            last_colon = i
    if last_colon < 0:
        return None
    goal = header[last_colon + 1:].lower()

    if ("angle" in pt or "∠" in problem_text) and "∠" not in source[m.start():]:
        if "dist" in goal:
            return (
                "the problem asks about angle equality but the goal only mentions `dist`. "
                "Use `∠` (EuclideanGeometry.angle) in the goal."
            )
    if ("^2" in pt or "² " in pt or "squared" in pt) and "^ 2" not in goal and "^2" not in goal:
        return (
            "the problem involves squared distances (Pythagoras-style) but the goal "
            "doesn't contain `^ 2`. Restore the squared form."
        )
    return None


def _strip_fences(text: str) -> str:
    cleaned = _THINK_RE.sub("", text).strip()
    match = _FENCED_LEAN_RE.search(cleaned)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        return "\n".join(line for line in lines if not line.startswith("```")).strip()
    return cleaned

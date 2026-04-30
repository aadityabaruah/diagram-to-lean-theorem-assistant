"""ClaudeProver — iteratively replace `sorry` with a real Lean 4 proof.

Takes a Lean source file (with `sorry` closing one or more theorems) and asks
Claude to produce a proof. Runs ``lake build`` against the returned source;
on failure, feeds the error back to Claude for another attempt. Stops when
the build succeeds or ``max_attempts`` is reached.

This is the one place the project lets a non-deterministic model touch
generated Lean — all other emission is deterministic. The subprocess
verification is the oracle that prevents hallucinated proofs from shipping.
"""
from __future__ import annotations

import os
import re
import time
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
    return os.environ.get("CLAUDE_PROVER_MODEL", "claude-opus-4-7")


INITIAL_PROMPT = """You are a Lean 4 + mathlib4 theorem-proving expert.

The file below contains a theorem whose proof is `sorry`. Write a complete,
correct proof. The file will be compiled against mathlib (version ~v4.15).

Output rules:
1. Return ONLY the complete file source. No markdown fences, no `<think>`
   blocks, no commentary. The entire response must be valid Lean 4 source.
2. Preserve all imports, namespace declarations, open statements, comments,
   theorem signatures, and hypothesis bindings EXACTLY as given.
3. Replace only the tactic block after `:= by` — you may use multiple lines.
4. If you truly cannot find a proof, leave `sorry`. Never fabricate a proof
   that references lemmas you are not sure exist.

Proof strategy hints:
- Start with `simp_all`, `linarith`, `nlinarith`, `polyrith`, `ring`,
  `field_simp`, `norm_num`, `positivity` — these handle many algebraic goals.
- For equalities involving real numbers derived from `dist`, expand via
  `dist_eq_norm_sub` and use the inner-product-space machinery.
- For geometry: relevant lemmas live in `Mathlib.Geometry.Euclidean.Angle.*`,
  `Mathlib.Geometry.Euclidean.Triangle`, `Mathlib.Analysis.InnerProductSpace.*`.
  Named lemmas you can try:
    * `EuclideanGeometry.angle_comm`, `angle_eq_zero_iff`
    * `EuclideanGeometry.dist_eq_norm_vsub`
    * `EuclideanGeometry.law_cos`
    * `InnerProductGeometry.angle_eq_iff`
    * `InnerProductGeometry.cos_angle`
- For the isosceles base-angles theorem, the law of cosines plus algebraic
  manipulation should give equality of `Real.cos` applied to the two angles,
  and then `Real.arccos` injectivity on [-1, 1] closes it.
- For Pythagoras with right angle at C: the fact `∠ A C B = π/2` implies
  `⟪A -ᵥ C, B -ᵥ C⟫ = 0`; combine with `dist_sq_eq_inner_sub_sq` or the
  parallelogram identity.

Lean file:

```lean
{source}
```
"""

RETRY_PROMPT = """Your previous proof attempt produced this Lean error:

```
{error}
```

Previous attempt:

```lean
{previous}
```

Fix the proof. Same rules as before: return ONLY the complete file source,
no commentary. If you still can't close the goal, leave `sorry`.
"""


class ClaudeProver:
    """Fills in Lean proofs by iterating build errors back to Claude."""

    def __init__(
        self, api_key: str, *, client: Any | None = None, model: str | None = None
    ) -> None:
        if not api_key:
            raise VLMError("ANTHROPIC_API_KEY is empty for ClaudeProver.")
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def prove(
        self,
        lean_source: str,
        *,
        lean_runner: LeanRunner,
        max_attempts: int = 3,
    ) -> tuple[str, LeanStatus, str | None]:
        """Attempt to close `sorry` in *lean_source* using *lean_runner* as oracle.

        Returns the final (source, status, stderr). If any attempt succeeds
        with status OK, the source has no `sorry`. Otherwise the original
        source is returned unchanged alongside the last failing status.
        """
        if "sorry" not in lean_source:
            status, stderr = lean_runner.typecheck(lean_source)
            return lean_source, status, stderr

        attempt_source = lean_source
        previous_error = ""
        previous_attempt = ""

        for attempt in range(max_attempts):
            if attempt == 0:
                prompt = INITIAL_PROMPT.format(source=attempt_source)
            else:
                prompt = RETRY_PROMPT.format(
                    error=previous_error[:3000], previous=previous_attempt[:6000]
                )

            candidate = self._invoke(prompt)
            candidate = _strip_fences(candidate)
            if not candidate.strip():
                continue

            status, stderr = lean_runner.typecheck(candidate)
            if status is LeanStatus.OK and "sorry" not in candidate:
                return candidate, LeanStatus.OK, stderr

            if status is LeanStatus.OK and "sorry" in candidate:
                # File compiles but the model gave up. Specifically nudge it.
                previous_error = (
                    "Your previous response compiled but still contains `sorry`. "
                    "You gave up. Try harder this time. Use the specific mathlib "
                    "lemma `EuclideanGeometry.law_cos`, the cancellation tactic "
                    "`linear_combination`, and `Real.injOn_cos` on `Set.Icc 0 π` "
                    "with `EuclideanGeometry.angle_nonneg` and "
                    "`EuclideanGeometry.angle_le_pi` to conclude angle equalities. "
                    "Handle degenerate cases (`dist a b = 0` ↔ `a = b`) with "
                    "`by_cases` and `dist_eq_zero`."
                )
            else:
                previous_error = stderr or "(no stderr)"
            previous_attempt = candidate

        # All attempts exhausted — return the original source with sorry intact,
        # build it once to get a stable status.
        status, stderr = lean_runner.typecheck(lean_source)
        return lean_source, status, stderr

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
                raise VLMError(f"Claude prover call failed: {first}") from first
            time.sleep(RETRY_BACKOFF_SEC)
            try:
                msg = _gen()
            except Exception as second:  # noqa: BLE001
                raise VLMError(f"Claude prover call failed after retry: {second}") from second
        return msg.content[0].text if msg.content else ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_LEAN_RE = re.compile(r"```(?:lean4?|)\s*\n(.*?)\n```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Extract Lean source from the model's response.

    Handles extended-thinking ``<think>`` blocks, fenced code blocks, and
    plain-text responses. Returns the first fenced Lean block if present,
    otherwise returns the remaining text with thinking blocks stripped.
    """
    # Drop any <think> blocks first
    cleaned = _THINK_RE.sub("", text).strip()

    # Prefer the first fenced Lean code block
    match = _FENCED_LEAN_RE.search(cleaned)
    if match:
        return match.group(1).strip()

    # If output starts with ``` but no match, strip fence lines manually
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        return "\n".join(line for line in lines if not line.startswith("```")).strip()

    return cleaned

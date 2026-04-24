"""JudgeLLM: semantic comparison and blame attribution for consensus pipeline.

Uses a Gemini model to:
- Compare two DiagramReadings and decide if they are equivalent.
- Compare two Extractions and detect issues such as circular reasoning.
- Attribute blame to a pipeline stage (vlm | extraction | emission) when
  Lean rejects generated source.

Expected Extraction shape (defined by the extraction subagent, mirrored here
for documentation purposes):

    @dataclass(frozen=True)
    class Extraction:
        assumptions: list[str]
        goal: str
        raw: dict
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError

_DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPARE_READINGS_PROMPT = '''You are comparing two VLM readings of the same geometry diagram.

Reading A:
{a_json}

Reading B:
{b_json}

Decide whether A and B describe the SAME diagram (even if phrased differently).
Differences that don't matter: token order, whitespace, synonyms like "triangle ABC" vs "triangle A B C".
Differences that DO matter: one side has an object/relation/mark the other doesn't, or contradictory marks.

Respond with ONLY JSON:
{{"equivalent": true|false, "preferred": "a"|"b"|"merge", "reason": "brief"}}
'''

COMPARE_EXTRACTIONS_PROMPT = '''You are comparing two extractions from the same geometry problem.

Each extraction has assumptions (given facts) and a goal (what to prove).

Extraction A:
  assumptions: {a_assumptions}
  goal: {a_goal}

Extraction B:
  assumptions: {b_assumptions}
  goal: {b_goal}

Critical checks:
- The goal must NOT appear in the assumption list (circular reasoning).
- Both extractions should have the same effective set of facts and the same goal.
- Prefer the shorter, non-circular extraction if they differ.

Respond with ONLY JSON:
{{"equivalent": true|false, "preferred": "a"|"b"|"merge", "reason": "brief"}}
'''

BLAME_PROMPT = '''You are diagnosing a failure in a diagram-to-Lean pipeline.

Pipeline stages:
- vlm: extracts objects/relations/marks from the image
- extraction: produces assumptions + goal from the VLM reading and problem text
- emission: formats the final Lean 4 source (deterministic string formatting)

VLM reading: {reading_json}
Extraction: {extraction_json}
Generated Lean (header only): {lean_header}

Lean error:
{lean_stderr}

Which stage is most likely to blame? Consider:
- "unknown identifier" errors → typically the vlm invented a term
- circular proof issues → the extraction included the goal as an assumption
- syntax errors → the emission stage has a bug

Respond with ONLY JSON:
{{"stage": "vlm"|"extraction"|"emission", "reason": "brief"}}
'''

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVerdict:
    """Result of a semantic comparison between two readings or extractions.

    Attributes:
        equivalent: True if both inputs describe the same information.
        preferred: Which input is preferred — ``"a"``, ``"b"``, or ``"merge"``.
        reason: Brief human-readable explanation from the judge model.
    """

    equivalent: bool
    preferred: str  # "a" | "b" | "merge"
    reason: str


@dataclass(frozen=True)
class BlameVerdict:
    """Result of pipeline failure attribution.

    Attributes:
        stage: The pipeline stage most likely responsible for the Lean error —
            ``"vlm"``, ``"extraction"``, or ``"emission"``.
        reason: Brief human-readable explanation from the judge model.
    """

    stage: str  # "vlm" | "extraction" | "emission"
    reason: str


# ---------------------------------------------------------------------------
# JudgeLLM
# ---------------------------------------------------------------------------


class JudgeLLM:
    """LLM-backed judge for semantic comparison and pipeline blame attribution.

    Parameters
    ----------
    api_key:
        Google Generative AI API key.  Must be non-empty.
    client:
        Optional pre-built ``google.genai.Client`` instance.  Injected in
        tests to avoid real network calls.
    model:
        Gemini model name.  Defaults to the ``JUDGE_MODEL`` environment
        variable, falling back to ``gemini-3.1-pro-preview``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        model: str | None = None,
    ) -> None:
        if not api_key:
            raise VLMError(
                "api_key is empty. Provide a valid Google Generative AI API key."
            )
        if client is None:
            from google import genai  # lazy so tests can mock without SDK installed

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model or _DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compare_readings(self, a: DiagramReading, b: DiagramReading) -> JudgeVerdict:
        """Semantically compare two :class:`DiagramReading` instances.

        Returns a :class:`JudgeVerdict` indicating whether the readings are
        equivalent and which (if any) is preferred.
        """
        prompt = COMPARE_READINGS_PROMPT.format(
            a_json=json.dumps(a.to_dict(), indent=2),
            b_json=json.dumps(b.to_dict(), indent=2),
        )
        data = self._call(prompt)
        return JudgeVerdict(
            equivalent=bool(data["equivalent"]),
            preferred=str(data["preferred"]),
            reason=str(data["reason"]),
        )

    def compare_extractions(self, a: Any, b: Any) -> JudgeVerdict:
        """Semantically compare two extraction objects (duck-typed).

        Each argument must expose ``.assumptions`` (list of str) and
        ``.goal`` (str) attributes.

        Returns a :class:`JudgeVerdict`.
        """
        prompt = COMPARE_EXTRACTIONS_PROMPT.format(
            a_assumptions=json.dumps(list(a.assumptions)),
            a_goal=str(a.goal),
            b_assumptions=json.dumps(list(b.assumptions)),
            b_goal=str(b.goal),
        )
        data = self._call(prompt)
        return JudgeVerdict(
            equivalent=bool(data["equivalent"]),
            preferred=str(data["preferred"]),
            reason=str(data["reason"]),
        )

    def attribute_blame(
        self,
        *,
        reading: DiagramReading,
        extraction: Any,
        lean_source: str,
        lean_stderr: str,
    ) -> BlameVerdict:
        """Attribute the blame for a Lean rejection to a pipeline stage.

        Parameters
        ----------
        reading:
            The VLM reading that fed into the pipeline.
        extraction:
            The extraction object (with ``.assumptions`` and ``.goal``).
        lean_source:
            The full generated Lean 4 source file.
        lean_stderr:
            The error output from the Lean checker.

        Returns a :class:`BlameVerdict` naming the guilty stage.
        """
        extraction_dict = {
            "assumptions": list(extraction.assumptions),
            "goal": str(extraction.goal),
        }
        # Provide only the first 30 lines of lean_source to keep the prompt concise.
        lean_header = "\n".join(lean_source.splitlines()[:30])

        prompt = BLAME_PROMPT.format(
            reading_json=json.dumps(reading.to_dict(), indent=2),
            extraction_json=json.dumps(extraction_dict, indent=2),
            lean_header=lean_header,
            lean_stderr=lean_stderr,
        )
        data = self._call(prompt)
        return BlameVerdict(
            stage=str(data["stage"]),
            reason=str(data["reason"]),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> dict[str, Any]:
        """Invoke the judge model, with one JSON-retry on parse failure.

        Raises :class:`VLMError` if the model returns unparseable JSON on
        both attempts.
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
        )
        text = response.text or ""
        parsed = _try_parse_json(text)
        if parsed is None:
            # Retry with a stricter instruction appended.
            retry_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON, no prose."
            response = self._client.models.generate_content(
                model=self._model,
                contents=[retry_prompt],
            )
            text = response.text or ""
            parsed = _try_parse_json(text)
            if parsed is None:
                raise VLMError(
                    f"JudgeLLM received non-JSON after retry. Raw output:\n{text}"
                )
        return parsed


# ---------------------------------------------------------------------------
# JSON parsing helper (mirrors vlm/gemini.py)
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse *text* as a JSON object; return ``None`` on failure."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(line for line in lines if not line.startswith("```"))
        stripped = inner
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None

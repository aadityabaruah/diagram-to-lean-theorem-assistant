from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from diagram_theorem_assistant.schema import DiagramMark, DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError
from diagram_theorem_assistant.vlm.prompts import MARKS_PROMPT, OBJECTS_PROMPT, RELATIONS_PROMPT

RETRY_BACKOFF_SEC = 2.0

_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "ConnectTimeout",
        "ConnectTimeoutError",
        "ReadTimeout",
        "ReadTimeoutError",
        "Timeout",
        "TimeoutError",
        "RemoteDisconnected",
        "ServerError",
        "ServiceUnavailableError",
        "InternalServerError",
    }
)


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


def _default_model() -> str:
    return os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")


class ClaudeAdapter:
    """Real VLM adapter using Anthropic Claude via the anthropic SDK.

    Makes three sequential calls per image (objects, relations, marks).

    Retry policy
    ------------
    * Transient network/server failures (connection reset, read timeout, 5xx):
      retried **once** after ``RETRY_BACKOFF_SEC`` seconds of sleep.
    * Authentication, model-not-found, and other client-side errors: propagated
      immediately as :class:`VLMError` with the original exception as ``__cause__``.
    * Non-JSON responses: one additional call with a stricter prompt asking for
      JSON only. If that also fails to parse, raises :class:`VLMError`.

    Worst-case call count per stage is three (transient retry -> JSON retry ->
    transient retry). Typical is one.
    """

    def __init__(self, api_key: str, *, client: Any | None = None, model: str | None = None) -> None:
        if not api_key:
            raise VLMError(
                "ANTHROPIC_API_KEY is empty. Set it in .env or pass api_key explicitly."
            )
        if client is None:
            import anthropic  # lazy so tests can mock without SDK installed
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading:
        image_bytes = Path(image_path).read_bytes()

        objects_json = self._call(OBJECTS_PROMPT, image_bytes, required_key="objects")
        relations_json = self._call(
            RELATIONS_PROMPT.format(objects=json.dumps(objects_json.get("objects", []))),
            image_bytes,
            required_key="relations",
        )
        marks_json = self._call(MARKS_PROMPT, image_bytes, required_key="marks")

        return DiagramReading(
            objects=list(objects_json.get("objects", [])),
            relations=list(relations_json.get("relations", [])),
            marks=[DiagramMark.from_dict(m) for m in marks_json.get("marks", [])],
            raw_vlm_output={
                "model": self._model,
                "objects_raw": objects_json,
                "relations_raw": relations_json,
                "marks_raw": marks_json,
            },
        )

    def _call(self, prompt: str, image_bytes: bytes, *, required_key: str) -> dict[str, Any]:
        response = self._invoke(prompt, image_bytes)
        parsed = _try_parse_json(response)
        if parsed is None:
            response = self._invoke(
                prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON, no prose.",
                image_bytes,
            )
            parsed = _try_parse_json(response)
            if parsed is None:
                raise VLMError(
                    f"Claude returned non-JSON after retry. Raw output:\n{response}"
                )
        if required_key not in parsed:
            parsed[required_key] = []
        return parsed

    def _invoke(self, prompt: str, image_bytes: bytes) -> str:
        encoded = base64.standard_b64encode(image_bytes).decode("utf-8")

        def _generate() -> Any:
            return self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

        try:
            response = _generate()
        except Exception as first_exc:  # noqa: BLE001 — classified below
            if not _is_retryable(first_exc):
                raise VLMError(f"Claude call failed: {first_exc}") from first_exc
            time.sleep(RETRY_BACKOFF_SEC)
            try:
                response = _generate()
            except Exception as second_exc:  # noqa: BLE001
                raise VLMError(f"Claude call failed after retry: {second_exc}") from second_exc
        return response.content[0].text or ""


def _try_parse_json(text: str) -> dict[str, Any] | None:
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

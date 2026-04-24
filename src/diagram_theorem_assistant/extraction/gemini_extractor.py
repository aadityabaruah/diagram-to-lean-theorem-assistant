from __future__ import annotations

import json
import os
import time
from typing import Any

from diagram_theorem_assistant.extraction.base import EXTRACTION_PROMPT, Extraction
from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError

RETRY_BACKOFF_SEC = 2.0

_RETRYABLE = frozenset({
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
})


def _default_model() -> str:
    return os.environ.get("GEMINI_EXTRACTOR_MODEL", "gemini-3.1-pro-preview")


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ in _RETRYABLE


class GeminiExtractor:
    def __init__(self, api_key: str, *, client: Any | None = None, model: str | None = None) -> None:
        if not api_key:
            raise VLMError("GEMINI_API_KEY is empty for GeminiExtractor.")
        if client is None:
            from google import genai  # lazy so tests can mock without SDK installed
            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model or _default_model()

    def extract(self, reading: DiagramReading, problem_text: str) -> Extraction:
        prompt = EXTRACTION_PROMPT.format(
            reading=json.dumps(reading.to_dict(), ensure_ascii=False),
            problem_text=problem_text,
        )
        parsed = self._call(prompt)
        return Extraction(
            assumptions=list(parsed.get("assumptions", [])),
            goal=str(parsed.get("goal", "")),
            raw={"model": self._model, "response": parsed},
        )

    def _call(self, prompt: str) -> dict[str, Any]:
        response = self._invoke(prompt)
        parsed = _try_parse_json(response)
        if parsed is None:
            response = self._invoke(
                prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON."
            )
            parsed = _try_parse_json(response)
            if parsed is None:
                raise VLMError(f"Gemini extractor returned non-JSON after retry:\n{response}")
        return parsed

    def _invoke(self, prompt: str) -> str:
        def _generate() -> Any:
            return self._client.models.generate_content(
                model=self._model,
                contents=[prompt],
            )

        try:
            response = _generate()
        except Exception as first:  # noqa: BLE001
            if not _is_retryable(first):
                raise VLMError(f"Gemini extractor call failed: {first}") from first
            time.sleep(RETRY_BACKOFF_SEC)
            try:
                response = _generate()
            except Exception as second:  # noqa: BLE001
                raise VLMError(f"Gemini extractor call failed after retry: {second}") from second
        return response.text or ""


def _try_parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None

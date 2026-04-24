from __future__ import annotations

import json
from pathlib import Path

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError


class FixtureAdapter:
    """Deterministic VLM adapter backed by committed JSON fixtures."""

    def __init__(self, fixtures_root: Path) -> None:
        self.fixtures_root = fixtures_root

    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading:
        path = fixture_path or (self.fixtures_root / (image_path.stem + ".json"))
        if not path.exists():
            raise FixtureNotFoundError(
                f"No VLM fixture at {path}. "
                f"Record one with: python scripts/record_vlm_fixtures.py --image {image_path}"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VLMError(f"Fixture {path} is not valid JSON: {exc}") from exc
        return DiagramReading.from_dict(data)

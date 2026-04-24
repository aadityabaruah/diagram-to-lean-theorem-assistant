from __future__ import annotations

from pathlib import Path
from typing import Protocol

from diagram_theorem_assistant.schema import DiagramReading


class PipelineError(Exception):
    """Base for all pipeline-level failures."""


class VLMError(PipelineError):
    """Raised when a VLM adapter cannot produce a valid DiagramReading."""


class FixtureNotFoundError(PipelineError):
    """Raised when a benchmark expects a fixture JSON that does not exist."""


class DiagramUnderstander(Protocol):
    def read(self, image_path: Path, *, fixture_path: Path | None = None) -> DiagramReading: ...

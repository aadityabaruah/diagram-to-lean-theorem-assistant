from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.base import LeanError


class LeanRunner:
    """Run `lake build` against generated Lean source.

    Writes `lean_source` to `generated_path`, invokes `lake build` scoped
    to the generated module, and returns (status, stderr). Gracefully
    degrades to LeanStatus.UNAVAILABLE if `lake` is not on PATH.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        generated_path: Path,
        timeout_sec: int = 60,
        build_target: str = "DiagramTheorems.Generated",
    ) -> None:
        self.project_root = project_root
        self.generated_path = generated_path
        self.timeout_sec = timeout_sec
        self.build_target = build_target

    def typecheck(self, lean_source: str) -> tuple[LeanStatus, str | None]:
        lake = shutil.which("lake")
        if lake is None:
            return LeanStatus.UNAVAILABLE, None

        self.generated_path.parent.mkdir(parents=True, exist_ok=True)
        self.generated_path.write_text(lean_source, encoding="utf-8")

        try:
            completed = subprocess.run(  # noqa: S603 — controlled args
                [lake, "build", self.build_target],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return LeanStatus.TIMEOUT, f"lake build exceeded {self.timeout_sec}s"
        except (OSError, subprocess.SubprocessError) as exc:
            raise LeanError(f"lake invocation failed: {exc}") from exc

        if completed.returncode == 0:
            return LeanStatus.OK, None
        return LeanStatus.TYPE_ERROR, completed.stderr.strip() or None

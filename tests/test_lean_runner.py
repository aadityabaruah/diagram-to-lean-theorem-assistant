import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus
from diagram_theorem_assistant.vlm.base import PipelineError


def test_runner_reports_unavailable_when_lake_missing(tmp_path: Path):
    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value=None):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "Generated.lean")
        status, stderr = runner.typecheck("theorem demo : True := by trivial")
    assert status is LeanStatus.UNAVAILABLE
    assert stderr is None


def test_runner_writes_file_and_parses_success(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    generated = project / "DiagramTheorems" / "Generated.lean"

    class _FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", return_value=_FakeCompleted()):
        runner = LeanRunner(project_root=project, generated_path=generated)
        status, stderr = runner.typecheck("theorem demo : True := by trivial")

    assert generated.exists()
    assert status is LeanStatus.OK
    assert stderr == ""


def test_runner_parses_type_error(tmp_path: Path):
    class _FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "Generated.lean:5:0: error: unknown identifier 'foo'"

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", return_value=_FakeCompleted()):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean")
        status, stderr = runner.typecheck("bogus source")

    assert status is LeanStatus.TYPE_ERROR
    assert "unknown identifier" in stderr


def test_runner_handles_timeout(tmp_path: Path):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["lake", "build"], timeout=1)

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", side_effect=_raise):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean", timeout_sec=1)
        status, stderr = runner.typecheck("slow")
    assert status is LeanStatus.TYPE_ERROR
    assert "timed out" in stderr


def test_runner_raises_on_unexpected_failure(tmp_path: Path):
    def _raise(*args, **kwargs):
        raise OSError("lake crashed hard")

    with patch("diagram_theorem_assistant.lean_runner.shutil.which", return_value="/fake/lake"), \
         patch("diagram_theorem_assistant.lean_runner.subprocess.run", side_effect=_raise):
        runner = LeanRunner(project_root=tmp_path, generated_path=tmp_path / "g.lean")
        with pytest.raises(PipelineError):
            runner.typecheck("src")


@pytest.mark.lean
def test_runner_builds_trivial_theorem_with_real_lake():
    project_root = Path(__file__).resolve().parent.parent / "lean_project"
    generated = project_root / "DiagramTheorems" / "Generated.lean"
    runner = LeanRunner(project_root=project_root, generated_path=generated)
    source = (
        "import DiagramTheorems.Basic\n"
        "namespace DiagramTheorems.Generated\n"
        "theorem smoke : True := by trivial\n"
        "end DiagramTheorems.Generated\n"
    )
    status, stderr = runner.typecheck(source)
    assert status is LeanStatus.OK, f"lake build failed: {stderr}"

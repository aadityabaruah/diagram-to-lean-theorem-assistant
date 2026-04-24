"""Orchestration pipeline: VLM reading → assumptions → goals → Lean typecheck.

Two public entry points:

* :func:`run_pipeline_from_reading` — the core stage composer. Callers supply
  an already-computed :class:`DiagramReading` (e.g. the UI, which reads once and
  then asks the user to review) plus the explicit confirmed-assumption list.

* :func:`run_pipeline` — convenience wrapper that reads the diagram via the
  supplied adapter and delegates to :func:`run_pipeline_from_reading`.

The helper :func:`candidate_assumptions` computes the inferred-but-unreviewed
assumption list from a reading + problem text so callers can show checkboxes.

``confirmed_assumptions`` is always an explicit :class:`list` — there is no
implicit fallback to the inferred candidates. Callers that want to accept the
candidates wholesale pass them in explicitly, so "caller forgot to confirm"
and "user confirmed nothing" are distinguishable.
"""
from __future__ import annotations

from pathlib import Path

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.consensus.judge import JudgeLLM
from diagram_theorem_assistant.extraction.base import AssumptionExtractor, Extraction
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_from
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import DiagramReading, LeanStatus, PipelineResult
from diagram_theorem_assistant.vlm.base import DiagramUnderstander


def candidate_assumptions(reading: DiagramReading, problem_text: str) -> list[str]:
    """Compute inferred (unreviewed) assumptions from a reading + problem text.

    UIs call this to populate the "confirm assumptions" checkbox list. The
    same function is used internally by the pipeline so the two agree.
    """
    return infer_assumptions(
        problem_text=problem_text,
        objects=reading.objects,
        diagram_marks=[mark.to_dict() for mark in reading.marks],
        confirmed_assumptions=None,
        relations=reading.relations,
    )


def run_pipeline_from_reading(
    *,
    reading: DiagramReading,
    problem_text: str,
    confirmed_assumptions: list[str],
    lean_runner: LeanRunner | None = None,
    theorem_name: str = "generated_theorem",
) -> PipelineResult:
    """Run the assumption → goal → Lean stages on a pre-computed reading."""
    candidate = candidate_assumptions(reading, problem_text)
    confirmed = list(confirmed_assumptions)

    extracted = extract_goal_from_text(problem_text)
    ranked = rank_goal_candidates(reading.objects, confirmed)
    goal_candidates: list[str] = [extracted, *ranked] if extracted is not None else ranked
    selected = extracted if extracted is not None else (ranked[0] if ranked else "")

    lean_source = theorem_from(name=theorem_name, assumptions=confirmed, goal=selected)

    if lean_runner is None:
        status: LeanStatus = LeanStatus.UNAVAILABLE
        stderr: str | None = None
    else:
        status, stderr = lean_runner.typecheck(lean_source)

    return PipelineResult(
        reading=reading,
        candidate_assumptions=candidate,
        confirmed_assumptions=confirmed,
        goal_candidates=goal_candidates,
        selected_goal=selected,
        lean_source=lean_source,
        lean_status=status,
        lean_stderr=stderr,
    )


def run_pipeline(
    *,
    image_path: Path,
    problem_text: str,
    confirmed_assumptions: list[str],
    vlm: DiagramUnderstander,
    vlm_fixture_path: Path | None = None,
    lean_runner: LeanRunner | None = None,
    theorem_name: str = "generated_theorem",
) -> PipelineResult:
    """Read the diagram and run the full pipeline.

    ``confirmed_assumptions`` is required — callers must decide what to confirm
    before calling. To accept the inferred candidates wholesale, first call
    :func:`candidate_assumptions` after a separate :meth:`DiagramUnderstander.read`
    and pass the result in.
    """
    reading = vlm.read(image_path, fixture_path=vlm_fixture_path)
    return run_pipeline_from_reading(
        reading=reading,
        problem_text=problem_text,
        confirmed_assumptions=confirmed_assumptions,
        lean_runner=lean_runner,
        theorem_name=theorem_name,
    )


def run_consensus_pipeline(
    *,
    image_path: Path,
    problem_text: str,
    vlm: DiagramUnderstander,
    extractor: AssumptionExtractor,
    judge: JudgeLLM | None = None,
    lean_runner: LeanRunner | None = None,
    theorem_name: str = "generated_theorem",
    max_retries: int = 3,
    vlm_fixture_path: Path | None = None,
) -> PipelineResult:
    """Run the pipeline with dual-provider consensus and judge-mediated retry.

    On Lean type-error, the judge attributes blame to vlm/extraction/emission
    and the pipeline re-runs only the blamed stage. VLM re-run implies
    extraction re-run (downstream dependency). Emission blame is terminal
    because it's a deterministic code bug.
    """
    reading: DiagramReading | None = None
    extraction: Extraction | None = None
    lean_source = ""
    status = LeanStatus.UNAVAILABLE
    stderr: str | None = None

    force_vlm = True
    force_extraction = True

    for _attempt in range(max_retries + 1):
        if force_vlm:
            reading = vlm.read(image_path, fixture_path=vlm_fixture_path)
            force_vlm = False
            force_extraction = True  # cascade: new reading → new extraction

        if force_extraction:
            assert reading is not None
            extraction = extractor.extract(reading, problem_text)
            force_extraction = False

        assert extraction is not None
        lean_source = theorem_from(
            name=theorem_name,
            assumptions=extraction.assumptions,
            goal=extraction.goal,
        )

        if lean_runner is None:
            status = LeanStatus.UNAVAILABLE
            stderr = None
            break

        status, stderr = lean_runner.typecheck(lean_source)
        if status is LeanStatus.OK:
            break
        if judge is None:
            break

        blame = judge.attribute_blame(
            reading=reading,
            extraction=extraction,
            lean_source=lean_source,
            lean_stderr=stderr or "",
        )
        if blame.stage == "vlm":
            force_vlm = True
            force_extraction = True
        elif blame.stage == "extraction":
            force_extraction = True
        else:
            # emission bug — can't recover by retry
            break

    assert reading is not None
    assert extraction is not None
    return PipelineResult(
        reading=reading,
        candidate_assumptions=list(extraction.assumptions),
        confirmed_assumptions=list(extraction.assumptions),
        goal_candidates=[extraction.goal] if extraction.goal else [],
        selected_goal=extraction.goal,
        lean_source=lean_source,
        lean_status=status,
        lean_stderr=stderr,
    )

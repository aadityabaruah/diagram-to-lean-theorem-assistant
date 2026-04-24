from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.consensus.judge import BlameVerdict
from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.pipeline import run_consensus_pipeline
from diagram_theorem_assistant.schema import DiagramMark, DiagramReading, LeanStatus


def _reading(objects=None):
    return DiagramReading(
        objects=list(objects or ["point A"]),
        relations=[],
        marks=[],
        raw_vlm_output={},
    )


def _vlm(readings):
    a = MagicMock()
    a.read.side_effect = readings
    return a


def _extractor(extractions):
    e = MagicMock()
    e.extract.side_effect = extractions
    return e


def _runner(statuses):
    r = MagicMock()
    r.typecheck.side_effect = statuses
    return r


def test_consensus_pipeline_ok_first_attempt(tmp_path: Path):
    vlm = _vlm([_reading(["point A", "point B"])])
    extractor = _extractor([Extraction(assumptions=["triangle A B C"], goal="x", raw={})])
    runner = _runner([(LeanStatus.OK, None)])
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove x",
        vlm=vlm,
        extractor=extractor,
        judge=None,
        lean_runner=runner,
        max_retries=3,
    )
    assert result.lean_status is LeanStatus.OK
    assert result.selected_goal == "x"
    assert vlm.read.call_count == 1
    assert extractor.extract.call_count == 1


def test_consensus_pipeline_retries_vlm_on_blame(tmp_path: Path):
    vlm = _vlm([
        _reading(["point bogus"]),
        _reading(["point A"]),
    ])
    extractor = _extractor([
        Extraction(assumptions=["bogus"], goal="x", raw={}),
        Extraction(assumptions=["triangle A B C"], goal="x", raw={}),
    ])
    runner = _runner([
        (LeanStatus.TYPE_ERROR, "unknown identifier bogus"),
        (LeanStatus.OK, None),
    ])
    judge = MagicMock()
    judge.attribute_blame.return_value = BlameVerdict(stage="vlm", reason="hallucinated")
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove x",
        vlm=vlm, extractor=extractor, judge=judge, lean_runner=runner,
    )
    assert result.lean_status is LeanStatus.OK
    assert vlm.read.call_count == 2
    assert extractor.extract.call_count == 2  # vlm re-run implies extraction re-run


def test_consensus_pipeline_retries_extraction_only_on_blame(tmp_path: Path):
    vlm = _vlm([_reading(["point A"])])
    extractor = _extractor([
        Extraction(assumptions=["PA = PB"], goal="PA = PB", raw={}),  # circular
        Extraction(assumptions=["P on perp bisector AB"], goal="PA = PB", raw={}),
    ])
    runner = _runner([
        (LeanStatus.TYPE_ERROR, "circular"),
        (LeanStatus.OK, None),
    ])
    judge = MagicMock()
    judge.attribute_blame.return_value = BlameVerdict(stage="extraction", reason="goal in assumptions")
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove PA = PB",
        vlm=vlm, extractor=extractor, judge=judge, lean_runner=runner,
    )
    assert result.lean_status is LeanStatus.OK
    assert vlm.read.call_count == 1  # VLM NOT re-run
    assert extractor.extract.call_count == 2


def test_consensus_pipeline_gives_up_on_emission_blame(tmp_path: Path):
    vlm = _vlm([_reading(["point A"])])
    extractor = _extractor([Extraction(assumptions=[], goal="x", raw={})])
    runner = _runner([(LeanStatus.TYPE_ERROR, "syntax error")])
    judge = MagicMock()
    judge.attribute_blame.return_value = BlameVerdict(stage="emission", reason="code bug")
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove x",
        vlm=vlm, extractor=extractor, judge=judge, lean_runner=runner,
    )
    assert result.lean_status is LeanStatus.TYPE_ERROR
    assert vlm.read.call_count == 1
    assert extractor.extract.call_count == 1


def test_consensus_pipeline_exhausts_retries(tmp_path: Path):
    vlm = _vlm([_reading() for _ in range(5)])
    extractor = _extractor([Extraction(assumptions=[], goal="x", raw={}) for _ in range(5)])
    runner = _runner([(LeanStatus.TYPE_ERROR, "err") for _ in range(5)])
    judge = MagicMock()
    judge.attribute_blame.return_value = BlameVerdict(stage="extraction", reason="keeps failing")
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove x",
        vlm=vlm, extractor=extractor, judge=judge, lean_runner=runner,
        max_retries=2,
    )
    assert result.lean_status is LeanStatus.TYPE_ERROR
    # initial + 2 retries = 3 total extractions
    assert extractor.extract.call_count == 3


def test_consensus_pipeline_no_runner_returns_unavailable(tmp_path: Path):
    vlm = _vlm([_reading()])
    extractor = _extractor([Extraction(assumptions=["x"], goal="y", raw={})])
    result = run_consensus_pipeline(
        image_path=tmp_path / "img.png",
        problem_text="prove x",
        vlm=vlm, extractor=extractor, judge=None, lean_runner=None,
    )
    assert result.lean_status is LeanStatus.UNAVAILABLE

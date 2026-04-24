from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.consensus.judge import JudgeVerdict
from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.extraction.consensus import ConsensusExtractor
from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError


def _reading():
    return DiagramReading(objects=[], relations=[], marks=[], raw_vlm_output={})


def _extractor(extraction):
    e = MagicMock()
    e.extract.return_value = extraction
    return e


def _judge(verdicts):
    j = MagicMock()
    j.compare_extractions.side_effect = verdicts
    return j


def test_consensus_returns_preferred_a():
    a = Extraction(assumptions=["triangle A B C"], goal="x", raw={})
    b = Extraction(assumptions=["triangle A B C"], goal="x", raw={})
    judge = _judge([JudgeVerdict(equivalent=True, preferred="a", reason="same")])
    cx = ConsensusExtractor(_extractor(a), _extractor(b), judge)
    assert cx.extract(_reading(), "text") is a


def test_consensus_returns_preferred_b():
    a = Extraction(assumptions=["X includes goal"], goal="X includes goal", raw={})
    b = Extraction(assumptions=["P on perpendicular_bisector AB"], goal="PA = PB", raw={})
    judge = _judge([JudgeVerdict(equivalent=True, preferred="b", reason="b is non-circular")])
    cx = ConsensusExtractor(_extractor(a), _extractor(b), judge)
    assert cx.extract(_reading(), "text") is b


def test_consensus_merges_intersection():
    a = Extraction(
        assumptions=["triangle A B C", "AB = AC", "extra_hallucination"],
        goal="angle ABC = angle BCA",
        raw={"model": "gemini"},
    )
    b = Extraction(
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
        raw={"model": "claude"},
    )
    judge = _judge([JudgeVerdict(equivalent=True, preferred="merge", reason="ok")])
    cx = ConsensusExtractor(_extractor(a), _extractor(b), judge)
    result = cx.extract(_reading(), "text")
    assert result.assumptions == ["triangle A B C", "AB = AC"]  # intersection
    assert result.goal == "angle ABC = angle BCA"
    assert "a" in result.raw and "b" in result.raw


def test_consensus_retries_on_disagreement():
    a = Extraction(assumptions=["X"], goal="g1", raw={})
    b = Extraction(assumptions=["Y"], goal="g2", raw={})
    primary = _extractor(a)
    secondary = _extractor(b)
    judge = _judge([
        JudgeVerdict(equivalent=False, preferred="a", reason="disagree"),
        JudgeVerdict(equivalent=True, preferred="a", reason="now agree"),
    ])
    cx = ConsensusExtractor(primary, secondary, judge, max_attempts=3)
    assert cx.extract(_reading(), "text") is a
    assert primary.extract.call_count == 2


def test_consensus_raises_when_exhausted():
    a = Extraction(assumptions=[], goal="a", raw={})
    b = Extraction(assumptions=[], goal="b", raw={})
    judge = _judge([
        JudgeVerdict(equivalent=False, preferred="a", reason="mismatch 1"),
        JudgeVerdict(equivalent=False, preferred="b", reason="mismatch 2"),
    ])
    cx = ConsensusExtractor(_extractor(a), _extractor(b), judge, max_attempts=2)
    with pytest.raises(VLMError) as info:
        cx.extract(_reading(), "text")
    assert "consensus" in str(info.value).lower()

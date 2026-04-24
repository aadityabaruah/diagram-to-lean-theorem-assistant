from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.consensus.judge import JudgeVerdict
from diagram_theorem_assistant.consensus.vlm_consensus import ConsensusVLMAdapter
from diagram_theorem_assistant.schema import DiagramMark, DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError


def _reading(objects=None, relations=None, marks=None):
    return DiagramReading(
        objects=list(objects or []),
        relations=list(relations or []),
        marks=list(marks or []),
        raw_vlm_output={},
    )


def _adapter(reading):
    a = MagicMock()
    a.read.return_value = reading
    return a


def _judge(verdicts):
    j = MagicMock()
    j.compare_readings.side_effect = verdicts
    return j


def test_consensus_returns_preferred_reading(tmp_path: Path):
    a = _reading(["point A"], ["triangle A B C"])
    b = _reading(["point A"], ["triangle A B C"])
    judge = _judge([JudgeVerdict(equivalent=True, preferred="a", reason="same")])
    adapter = ConsensusVLMAdapter(_adapter(a), _adapter(b), judge)
    result = adapter.read(tmp_path / "img.png")
    assert result is a


def test_consensus_returns_b_when_preferred_b(tmp_path: Path):
    a = _reading(["point A"])
    b = _reading(["point A", "point B"])
    judge = _judge([JudgeVerdict(equivalent=True, preferred="b", reason="b more complete")])
    adapter = ConsensusVLMAdapter(_adapter(a), _adapter(b), judge)
    result = adapter.read(tmp_path / "img.png")
    assert result is b


def test_consensus_merges_when_preferred_merge(tmp_path: Path):
    a = _reading(["point A", "point B"], ["triangle A B C"], [DiagramMark(type="right_angle", points=["A", "C", "B"])])
    b = _reading(["point B", "point C"], ["parallel l m"], [DiagramMark(type="equal_length", segments=["AB", "AC"])])
    judge = _judge([JudgeVerdict(equivalent=True, preferred="merge", reason="complementary")])
    adapter = ConsensusVLMAdapter(_adapter(a), _adapter(b), judge)
    result = adapter.read(tmp_path / "img.png")
    assert result.objects == ["point A", "point B", "point C"]  # dedup, order preserved
    assert set(result.relations) == {"triangle A B C", "parallel l m"}
    assert len(result.marks) == 2


def test_consensus_retries_on_disagreement(tmp_path: Path):
    a = _reading(["point A"])
    b = _reading(["point X"])
    primary = _adapter(a)
    secondary = _adapter(b)
    # first attempt disagree, second agree
    judge = _judge([
        JudgeVerdict(equivalent=False, preferred="a", reason="conflict"),
        JudgeVerdict(equivalent=True, preferred="a", reason="ok"),
    ])
    adapter = ConsensusVLMAdapter(primary, secondary, judge, max_attempts=3)
    result = adapter.read(tmp_path / "img.png")
    assert result is a
    assert primary.read.call_count == 2
    assert secondary.read.call_count == 2


def test_consensus_raises_when_attempts_exhausted(tmp_path: Path):
    a = _reading(["point A"])
    b = _reading(["point X"])
    judge = _judge([
        JudgeVerdict(equivalent=False, preferred="a", reason="mismatch 1"),
        JudgeVerdict(equivalent=False, preferred="b", reason="mismatch 2"),
    ])
    adapter = ConsensusVLMAdapter(_adapter(a), _adapter(b), judge, max_attempts=2)
    with pytest.raises(VLMError) as info:
        adapter.read(tmp_path / "img.png")
    assert "consensus" in str(info.value).lower()

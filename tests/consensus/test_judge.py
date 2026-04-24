from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.consensus.judge import JudgeLLM, JudgeVerdict, BlameVerdict
from diagram_theorem_assistant.vlm.base import VLMError


@dataclass(frozen=True)
class _Extraction:
    assumptions: list
    goal: str
    raw: dict


def _resp(text):
    r = MagicMock()
    r.text = text
    return r


def _fake_client(responses):
    c = MagicMock()
    c.models.generate_content.side_effect = [_resp(t) for t in responses]
    return c


def _reading(objects=None, relations=None):
    return DiagramReading(
        objects=list(objects or []),
        relations=list(relations or []),
        marks=[],
        raw_vlm_output={},
    )


def test_judge_requires_api_key():
    with pytest.raises(VLMError):
        JudgeLLM(api_key="", client=MagicMock(), model="x")


def test_judge_compare_readings_returns_verdict():
    client = _fake_client([
        '{"equivalent": true, "preferred": "a", "reason": "same objects"}'
    ])
    judge = JudgeLLM(api_key="fake", client=client, model="judge-test")
    a = _reading(["point A"], ["triangle A B C"])
    b = _reading(["point A"], ["triangle A B C"])
    v = judge.compare_readings(a, b)
    assert isinstance(v, JudgeVerdict)
    assert v.equivalent is True
    assert v.preferred == "a"


def test_judge_compare_extractions_returns_verdict():
    client = _fake_client([
        '{"equivalent": false, "preferred": "b", "reason": "a includes the goal"}'
    ])
    judge = JudgeLLM(api_key="fake", client=client, model="judge-test")
    a = _Extraction(assumptions=["PA = PB"], goal="PA = PB", raw={})
    b = _Extraction(assumptions=["P on perpendicular_bisector AB"], goal="PA = PB", raw={})
    v = judge.compare_extractions(a, b)
    assert v.equivalent is False
    assert v.preferred == "b"


def test_judge_attribute_blame_returns_stage():
    client = _fake_client([
        '{"stage": "vlm", "reason": "reading contains hallucinated point \'intersection\'"}'
    ])
    judge = JudgeLLM(api_key="fake", client=client, model="judge-test")
    r = _reading(["point intersection"], [])
    e = _Extraction(assumptions=[], goal="x", raw={})
    v = judge.attribute_blame(
        reading=r, extraction=e,
        lean_source="theorem t : True := by sorry",
        lean_stderr="unknown identifier 'intersection'",
    )
    assert isinstance(v, BlameVerdict)
    assert v.stage == "vlm"


def test_judge_handles_malformed_json_with_retry():
    client = _fake_client([
        "not JSON",
        '{"equivalent": true, "preferred": "a", "reason": "ok"}',
    ])
    judge = JudgeLLM(api_key="fake", client=client, model="judge-test")
    v = judge.compare_readings(_reading(), _reading())
    assert v.equivalent is True
    assert client.models.generate_content.call_count == 2


def test_judge_raises_on_second_bad_json():
    client = _fake_client(["not JSON", "still not JSON"])
    judge = JudgeLLM(api_key="fake", client=client, model="judge-test")
    with pytest.raises(VLMError):
        judge.compare_readings(_reading(), _reading())

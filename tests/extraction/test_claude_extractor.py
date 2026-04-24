from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.extraction.claude_extractor import ClaudeExtractor
from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError


def _block(text):
    b = MagicMock()
    b.text = text
    return b


def _msg(text):
    m = MagicMock()
    m.content = [_block(text)]
    return m


def _fake_client(responses):
    c = MagicMock()
    c.messages.create.side_effect = [_msg(t) for t in responses]
    return c


def _reading():
    return DiagramReading(
        objects=["point A", "point B", "point C"],
        relations=["triangle A B C"],
        marks=[],
        raw_vlm_output={},
    )


def test_extractor_requires_api_key():
    with pytest.raises(VLMError):
        ClaudeExtractor(api_key="", client=MagicMock(), model="x")


def test_extractor_returns_parsed_extraction():
    client = _fake_client([
        '{"assumptions": ["triangle A B C", "AB = AC"], "goal": "angle ABC = angle BCA"}'
    ])
    extractor = ClaudeExtractor(api_key="fake", client=client, model="ext-test")
    result = extractor.extract(_reading(), "In triangle ABC, AB = AC. Prove angle ABC = angle BCA.")
    assert isinstance(result, Extraction)
    assert "AB = AC" in result.assumptions
    assert result.goal == "angle ABC = angle BCA"


def test_extractor_retries_once_on_bad_json():
    client = _fake_client([
        "not JSON",
        '{"assumptions": ["triangle A B C"], "goal": "x"}',
    ])
    extractor = ClaudeExtractor(api_key="fake", client=client, model="ext-test")
    result = extractor.extract(_reading(), "text")
    assert result.goal == "x"
    assert client.messages.create.call_count == 2


def test_extractor_raises_after_second_bad_json():
    client = _fake_client(["not JSON", "still not JSON"])
    extractor = ClaudeExtractor(api_key="fake", client=client, model="ext-test")
    with pytest.raises(VLMError):
        extractor.extract(_reading(), "text")

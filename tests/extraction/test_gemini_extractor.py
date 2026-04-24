from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.extraction.gemini_extractor import GeminiExtractor
from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError


def _resp(text):
    r = MagicMock()
    r.text = text
    return r


def _fake_client(responses):
    c = MagicMock()
    c.models.generate_content.side_effect = [_resp(t) for t in responses]
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
        GeminiExtractor(api_key="", client=MagicMock(), model="x")


def test_extractor_returns_parsed_extraction():
    client = _fake_client([
        '{"assumptions": ["triangle A B C", "AB = AC"], "goal": "angle ABC = angle BCA"}'
    ])
    extractor = GeminiExtractor(api_key="fake", client=client, model="ext-test")
    result = extractor.extract(_reading(), "In triangle ABC, AB = AC. Prove angle ABC = angle BCA.")
    assert isinstance(result, Extraction)
    assert "triangle A B C" in result.assumptions
    assert "AB = AC" in result.assumptions
    assert result.goal == "angle ABC = angle BCA"


def test_extractor_retries_once_on_bad_json():
    client = _fake_client([
        "not JSON",
        '{"assumptions": ["triangle A B C"], "goal": "angle sum"}',
    ])
    extractor = GeminiExtractor(api_key="fake", client=client, model="ext-test")
    result = extractor.extract(_reading(), "some text")
    assert result.goal == "angle sum"
    assert client.models.generate_content.call_count == 2


def test_extractor_raises_after_second_bad_json():
    client = _fake_client(["not JSON", "still not JSON"])
    extractor = GeminiExtractor(api_key="fake", client=client, model="ext-test")
    with pytest.raises(VLMError):
        extractor.extract(_reading(), "text")


def test_extractor_fills_raw_field():
    payload = '{"assumptions": [], "goal": "x"}'
    client = _fake_client([payload])
    extractor = GeminiExtractor(api_key="fake", client=client, model="ext-test")
    result = extractor.extract(_reading(), "text")
    assert isinstance(result.raw, dict)
    assert "model" in result.raw or "response" in result.raw

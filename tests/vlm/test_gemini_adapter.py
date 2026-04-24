from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter


def _response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _fake_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = [_response(text) for text in responses]
    return client


def test_gemini_adapter_sends_three_calls_and_merges(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        '{"objects": ["point A", "point B", "point C"]}',
        '{"relations": ["triangle A B C"]}',
        '{"marks": [{"type": "equal_length", "segments": ["AB", "AC"]}]}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)

    assert reading.objects == ["point A", "point B", "point C"]
    assert reading.relations == ["triangle A B C"]
    assert [mark.type for mark in reading.marks] == ["equal_length"]
    assert client.models.generate_content.call_count == 3


def test_gemini_adapter_retries_once_on_bad_json(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        "not JSON",
        '{"objects": ["point X"]}',
        '{"relations": []}',
        '{"marks": []}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)
    assert reading.objects == ["point X"]
    # 1 bad + 1 retry for stage 1, then stages 2 and 3 clean = 4 calls
    assert client.models.generate_content.call_count == 4


def test_gemini_adapter_raises_after_second_bad_json(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client(["still not JSON", "still not JSON"])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    with pytest.raises(VLMError):
        adapter.read(image)


def test_gemini_adapter_requires_api_key():
    with pytest.raises(VLMError):
        GeminiAdapter(api_key="", client=MagicMock(), model="gemini-test")


def test_gemini_adapter_retries_on_transient_network_error(tmp_path: Path, monkeypatch):
    """Retryable exceptions trigger one retry with backoff; result then succeeds."""
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("diagram_theorem_assistant.vlm.gemini.time.sleep", lambda _: None)

    client = MagicMock()
    client.models.generate_content.side_effect = [
        ConnectionError("transient network blip"),
        _response('{"objects": ["point A"]}'),
        _response('{"relations": []}'),
        _response('{"marks": []}'),
    ]
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)
    assert reading.objects == ["point A"]
    # 1 transient + 1 retry on stage 1, 1 each on stages 2 and 3 = 4 calls
    assert client.models.generate_content.call_count == 4


def test_gemini_adapter_does_not_retry_on_non_retryable_error(tmp_path: Path):
    """Auth-style errors (e.g. 401) are propagated immediately without retry."""
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    class AuthError(Exception):
        """Not in the retryable name set."""

    client = MagicMock()
    client.models.generate_content.side_effect = [AuthError("401 unauthorized")]
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    with pytest.raises(VLMError):
        adapter.read(image)
    assert client.models.generate_content.call_count == 1


def test_gemini_adapter_raises_if_retry_also_fails(tmp_path: Path, monkeypatch):
    """Two transient failures in a row surface as VLMError — no third attempt."""
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr("diagram_theorem_assistant.vlm.gemini.time.sleep", lambda _: None)

    client = MagicMock()
    client.models.generate_content.side_effect = [
        ConnectionError("first"),
        ConnectionError("second"),
    ]
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    with pytest.raises(VLMError):
        adapter.read(image)
    assert client.models.generate_content.call_count == 2


def test_gemini_adapter_defaults_missing_fields(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    client = _fake_client([
        '{"objects": ["point A"], "ignored": "extra"}',
        '{}',
        '{"marks": []}',
    ])
    adapter = GeminiAdapter(api_key="fake", client=client, model="gemini-test")

    reading = adapter.read(image)
    assert reading.objects == ["point A"]
    assert reading.relations == []
    assert reading.marks == []

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.schema import DiagramReading
from diagram_theorem_assistant.vlm.base import VLMError
from diagram_theorem_assistant.vlm.claude import ClaudeAdapter


def _content_block(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    return block


def _message(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [_content_block(text)]
    return msg


def _fake_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [_message(t) for t in responses]
    return client


def test_claude_adapter_sends_three_calls_and_merges(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _fake_client([
        '{"objects": ["point A", "point B", "point C"]}',
        '{"relations": ["triangle A B C"]}',
        '{"marks": [{"type": "equal_length", "segments": ["AB", "AC"]}]}',
    ])
    adapter = ClaudeAdapter(api_key="fake", client=client, model="claude-test")
    reading = adapter.read(image)
    assert reading.objects == ["point A", "point B", "point C"]
    assert reading.relations == ["triangle A B C"]
    assert [m.type for m in reading.marks] == ["equal_length"]
    assert client.messages.create.call_count == 3


def test_claude_adapter_requires_api_key():
    with pytest.raises(VLMError):
        ClaudeAdapter(api_key="", client=MagicMock(), model="claude-test")


def test_claude_adapter_retries_on_transient_network_error(tmp_path: Path, monkeypatch):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr("diagram_theorem_assistant.vlm.claude.time.sleep", lambda _: None)
    client = MagicMock()
    client.messages.create.side_effect = [
        ConnectionError("blip"),
        _message('{"objects": ["point A"]}'),
        _message('{"relations": []}'),
        _message('{"marks": []}'),
    ]
    adapter = ClaudeAdapter(api_key="fake", client=client, model="claude-test")
    reading = adapter.read(image)
    assert reading.objects == ["point A"]
    assert client.messages.create.call_count == 4


def test_claude_adapter_raises_after_second_bad_json(tmp_path: Path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = _fake_client(["not JSON", "still not JSON"])
    adapter = ClaudeAdapter(api_key="fake", client=client, model="claude-test")
    with pytest.raises(VLMError):
        adapter.read(image)

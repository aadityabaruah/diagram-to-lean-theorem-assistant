from pathlib import Path

import pytest

from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter


def test_fixture_adapter_reads_hand_authored_fixture():
    adapter = FixtureAdapter(Path("examples/vlm_fixtures"))
    reading = adapter.read(Path("examples/images/synthetic/isosceles_triangle.png"),
                           fixture_path=Path("examples/vlm_fixtures/isosceles_triangle_base_angles.json"))
    assert "point A" in reading.objects
    assert any(mark.type == "equal_length" for mark in reading.marks)


def test_fixture_adapter_raises_when_fixture_missing(tmp_path: Path):
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(FixtureNotFoundError) as info:
        adapter.read(Path("anything.png"), fixture_path=tmp_path / "nope.json")
    assert "record_vlm_fixtures.py" in str(info.value)


def test_fixture_adapter_raises_vlm_error_on_malformed_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("this is not JSON")
    adapter = FixtureAdapter(tmp_path)
    with pytest.raises(VLMError):
        adapter.read(Path("img.png"), fixture_path=bad)

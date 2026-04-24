import os
from pathlib import Path

import pytest

from diagram_theorem_assistant.vlm.gemini import GeminiAdapter


@pytest.mark.live
def test_live_gemini_reads_synthetic_isosceles_triangle():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    image = Path("examples/images/synthetic/isosceles_triangle.png")
    assert image.exists(), "Run scripts/generate_synthetic.py first"

    adapter = GeminiAdapter(api_key=api_key)
    reading = adapter.read(image)

    # Minimal structural assertions — exact outputs vary by model
    assert isinstance(reading.objects, list)
    assert len(reading.objects) >= 3

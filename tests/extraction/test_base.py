from diagram_theorem_assistant.extraction.base import Extraction, EXTRACTION_PROMPT


def test_extraction_round_trip_via_dict():
    original = Extraction(
        assumptions=["triangle A B C", "AB = AC"],
        goal="angle ABC = angle BCA",
        raw={"source": "test"},
    )
    restored = Extraction.from_dict(original.to_dict())
    assert restored == original


def test_extraction_defaults_raw_to_empty_dict():
    e = Extraction(assumptions=[], goal="x")
    assert e.raw == {}


def test_extraction_prompt_mentions_goal_exclusion():
    assert "not appear in the assumption list" in EXTRACTION_PROMPT.lower() or \
           "never appear in the assumption list" in EXTRACTION_PROMPT.lower()
    assert "{reading}" in EXTRACTION_PROMPT
    assert "{problem_text}" in EXTRACTION_PROMPT

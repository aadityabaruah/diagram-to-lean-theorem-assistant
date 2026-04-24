from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.lean_emission.claude_formalizer import (
    ClaudeFormalizer,
    _is_circular,
    _strip_fences,
)
from diagram_theorem_assistant.schema import DiagramReading, LeanStatus
from diagram_theorem_assistant.vlm.base import VLMError


CIRCULAR_SOURCE = """import DiagramTheorems.Basic

namespace DiagramTheorems.Generated

theorem alternate_interior_angles
    (A B C D P Q : EuclideanSpace ℝ (Fin 2))
    (hPar : ∠ A P Q = ∠ D Q P) :
    ∠ A P Q = ∠ D Q P := by
  exact hPar

end DiagramTheorems.Generated
"""

HONEST_SORRY_SOURCE = """import DiagramTheorems.Basic

namespace DiagramTheorems.Generated

theorem isosceles_base_angles
    (A B C : EuclideanSpace ℝ (Fin 2))
    (hAB : dist A B = dist A C) :
    ∠ A B C = ∠ B C A := by
  sorry

end DiagramTheorems.Generated
"""

PROVED_SOURCE = """import DiagramTheorems.Basic

namespace DiagramTheorems.Generated

-- The hypothesis `a = b` is distinct from the goal `a + 1 = b + 1`.
theorem demo
    (a b : ℝ) (h : a = b) :
    a + 1 = b + 1 := by
  rw [h]

end DiagramTheorems.Generated
"""


def test_is_circular_catches_goal_as_hypothesis():
    assert _is_circular(CIRCULAR_SOURCE) is True


def test_is_circular_accepts_real_proof_where_hyp_differs_from_goal():
    assert _is_circular(PROVED_SOURCE) is False


def test_is_circular_accepts_honest_sorry():
    assert _is_circular(HONEST_SORRY_SOURCE) is False


def test_is_circular_ignores_colons_inside_binders():
    source = """theorem t
    (A : EuclideanSpace ℝ (Fin 2)) (h : True) :
    True := by trivial
"""
    # The goal is `True` and there's a hypothesis `(h : True)` — this IS circular.
    assert _is_circular(source) is True


def test_strip_fences_drops_think_blocks():
    raw = "<think>reasoning</think>\n```lean\ntheorem x : True := by trivial\n```"
    assert _strip_fences(raw) == "theorem x : True := by trivial"


def _block(text):
    b = MagicMock()
    b.text = text
    return b


def _msg(text):
    m = MagicMock()
    m.content = [_block(text)]
    return m


def _runner(outcomes):
    r = MagicMock()
    r.typecheck.side_effect = outcomes
    return r


def _reading():
    return DiagramReading(objects=[], relations=[], marks=[], raw_vlm_output={})


def _extraction():
    return Extraction(assumptions=["triangle A B C"], goal="angle ABC = angle BCA", raw={})


def test_formalizer_requires_api_key():
    with pytest.raises(VLMError):
        ClaudeFormalizer(api_key="", client=MagicMock(), model="x")


def test_formalizer_returns_first_successful_source():
    client = MagicMock()
    client.messages.create.side_effect = [_msg(HONEST_SORRY_SOURCE)]
    runner = _runner([(LeanStatus.OK, None)])
    formalizer = ClaudeFormalizer(api_key="fake", client=client, model="x")
    source, status, _ = formalizer.formalize(
        reading=_reading(),
        extraction=_extraction(),
        theorem_name="iso",
        lean_runner=runner,
        max_attempts=2,
    )
    assert status is LeanStatus.OK
    assert "isosceles_base_angles" in source


def test_formalizer_rejects_circular_and_retries():
    client = MagicMock()
    client.messages.create.side_effect = [
        _msg(CIRCULAR_SOURCE),
        _msg(HONEST_SORRY_SOURCE),
    ]
    runner = _runner([(LeanStatus.OK, None)])  # only the second candidate reaches lake
    formalizer = ClaudeFormalizer(api_key="fake", client=client, model="x")
    source, status, _ = formalizer.formalize(
        reading=_reading(),
        extraction=_extraction(),
        theorem_name="iso",
        lean_runner=runner,
        max_attempts=3,
    )
    assert status is LeanStatus.OK
    assert _is_circular(source) is False
    assert client.messages.create.call_count == 2


def test_formalizer_returns_best_effort_when_attempts_exhausted():
    client = MagicMock()
    client.messages.create.side_effect = [
        _msg(CIRCULAR_SOURCE),
        _msg(CIRCULAR_SOURCE),
    ]
    runner = _runner([])  # never called — circular check short-circuits first
    formalizer = ClaudeFormalizer(api_key="fake", client=client, model="x")
    source, status, stderr = formalizer.formalize(
        reading=_reading(),
        extraction=_extraction(),
        theorem_name="iso",
        lean_runner=runner,
        max_attempts=2,
    )
    # Returns the last attempt with its failing status/stderr
    assert status is LeanStatus.TYPE_ERROR
    assert "circular" in (stderr or "").lower()

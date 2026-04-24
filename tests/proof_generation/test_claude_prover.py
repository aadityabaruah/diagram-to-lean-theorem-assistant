from unittest.mock import MagicMock

import pytest

from diagram_theorem_assistant.proof_generation.claude_prover import ClaudeProver, _strip_fences
from diagram_theorem_assistant.schema import LeanStatus
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


def _runner(outcomes):
    r = MagicMock()
    r.typecheck.side_effect = outcomes
    return r


SOURCE_WITH_SORRY = """theorem demo : True := by sorry"""
SOURCE_WITH_PROOF = """theorem demo : True := by trivial"""


def test_prover_requires_api_key():
    with pytest.raises(VLMError):
        ClaudeProver(api_key="", client=MagicMock(), model="x")


def test_prover_short_circuits_if_no_sorry_and_source_ok():
    runner = _runner([(LeanStatus.OK, None)])
    prover = ClaudeProver(api_key="fake", client=MagicMock(), model="x")
    final, status, stderr = prover.prove("theorem demo : True := by trivial", lean_runner=runner)
    assert status is LeanStatus.OK
    assert "sorry" not in final


def test_prover_returns_successful_proof_on_first_attempt():
    client = _fake_client([SOURCE_WITH_PROOF])
    runner = _runner([(LeanStatus.OK, None)])
    prover = ClaudeProver(api_key="fake", client=client, model="x")
    final, status, _ = prover.prove(SOURCE_WITH_SORRY, lean_runner=runner, max_attempts=3)
    assert status is LeanStatus.OK
    assert "sorry" not in final
    assert "trivial" in final


def test_prover_retries_on_error_and_succeeds():
    client = _fake_client([
        "theorem demo : True := by nonsense_tactic",
        SOURCE_WITH_PROOF,
    ])
    runner = _runner([
        (LeanStatus.TYPE_ERROR, "unknown tactic 'nonsense_tactic'"),
        (LeanStatus.OK, None),
    ])
    prover = ClaudeProver(api_key="fake", client=client, model="x")
    final, status, _ = prover.prove(SOURCE_WITH_SORRY, lean_runner=runner, max_attempts=3)
    assert status is LeanStatus.OK
    assert "trivial" in final
    assert client.messages.create.call_count == 2


def test_prover_falls_back_to_original_after_max_attempts():
    client = _fake_client([SOURCE_WITH_SORRY, SOURCE_WITH_SORRY])
    runner = _runner([
        (LeanStatus.OK, None),      # both model attempts still contain sorry
        (LeanStatus.OK, None),
        (LeanStatus.OK, None),      # final run on the original
    ])
    prover = ClaudeProver(api_key="fake", client=client, model="x")
    final, status, _ = prover.prove(SOURCE_WITH_SORRY, lean_runner=runner, max_attempts=2)
    assert "sorry" in final  # no closure


def test_strip_fences_removes_markdown_blocks():
    raw = "```lean\ntheorem x : True := by trivial\n```"
    assert _strip_fences(raw) == "theorem x : True := by trivial"


def test_strip_fences_removes_thinking_blocks():
    raw = "<think>let me think</think>\n```lean\ntheorem x : True := by trivial\n```"
    assert _strip_fences(raw) == "theorem x : True := by trivial"


def test_strip_fences_returns_plain_text_as_is():
    raw = "theorem x : True := by trivial"
    assert _strip_fences(raw) == "theorem x : True := by trivial"

# Dual-Provider Consensus Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the current single-provider pipeline with a two-provider (Gemini + Claude) consensus architecture at both the VLM-reading and assumption-extraction stages, arbitrated by a judge LLM, with Lean as the final oracle and a retry loop that re-runs the blamed stage on failure.

**Architecture:** Each of the two "reasoning" stages runs Gemini and Claude in parallel; a judge LLM decides if their outputs are semantically equivalent. Disagreement triggers a stricter-prompt retry. `lake build` failures route through the judge again to blame a specific stage and re-run only that one. Max 3 retries per pipeline run.

**Tech Stack:** Python 3.11; `google-genai`, `anthropic` SDKs; existing Lean 4 project; Streamlit UI; pytest.

---

## File Structure

### New modules
- `src/diagram_theorem_assistant/vlm/claude.py` — `ClaudeAdapter` (same Protocol as `GeminiAdapter`)
- `src/diagram_theorem_assistant/consensus/__init__.py`
- `src/diagram_theorem_assistant/consensus/judge.py` — `JudgeLLM` with `compare_readings`, `compare_extractions`, `attribute_blame`
- `src/diagram_theorem_assistant/consensus/vlm_consensus.py` — `ConsensusVLMAdapter`
- `src/diagram_theorem_assistant/extraction/__init__.py`
- `src/diagram_theorem_assistant/extraction/base.py` — `AssumptionExtractor` Protocol + `Extraction` dataclass
- `src/diagram_theorem_assistant/extraction/gemini_extractor.py`
- `src/diagram_theorem_assistant/extraction/claude_extractor.py`
- `src/diagram_theorem_assistant/extraction/consensus.py` — `ConsensusExtractor`

### Refactored modules
- `src/diagram_theorem_assistant/pipeline.py` — accept `ConsensusVLMAdapter` + `ConsensusExtractor`, add retry loop
- `app.py` — consensus toggle, per-stage model pickers

### New tests
- `tests/vlm/test_claude_adapter.py`
- `tests/consensus/__init__.py`
- `tests/consensus/test_judge.py`
- `tests/consensus/test_vlm_consensus.py`
- `tests/extraction/__init__.py`
- `tests/extraction/test_gemini_extractor.py`
- `tests/extraction/test_claude_extractor.py`
- `tests/extraction/test_consensus_extractor.py`
- `tests/test_pipeline_consensus.py`

### New dependencies (pyproject.toml)
- `anthropic>=0.34` added to runtime dependencies

---

## Task waves

- **Wave 1** (sequential): pyproject dep + anthropic SDK install
- **Wave 2** (3 parallel): ClaudeAdapter, JudgeLLM, extraction/base.py
- **Wave 3** (2 parallel): GeminiExtractor, ClaudeExtractor (depend on extraction/base)
- **Wave 4** (2 parallel): VLM consensus, Extraction consensus
- **Wave 5** (1): pipeline.py refactor with retry + blame loop
- **Wave 6** (1): app.py UI for consensus controls
- **Wave 7** (1): end-to-end benchmark run with consensus mode

## Key design contracts

### Extraction dataclass

```python
@dataclass(frozen=True)
class Extraction:
    assumptions: list[str]   # minimal given facts, excluding the goal
    goal: str                # what should be proven
    raw: dict                # original LLM JSON for auditing
```

### Judge interface

```python
class JudgeLLM:
    def compare_readings(self, a: DiagramReading, b: DiagramReading) -> JudgeVerdict:
        """Returns {"equivalent": bool, "preferred": "a"|"b"|"merge", "reason": str}"""

    def compare_extractions(self, a: Extraction, b: Extraction) -> JudgeVerdict:
        """Same return shape."""

    def attribute_blame(self, stage_outputs: dict, lean_error: str) -> BlameVerdict:
        """Returns {"blamed_stage": "vlm"|"extraction"|"emission", "reason": str}"""
```

### Consensus adapter pattern

Both `ConsensusVLMAdapter` and `ConsensusExtractor` implement the same `.read` / `.extract` protocol as their single-provider counterparts, so `pipeline.py` treats them uniformly.

### Retry loop (pipeline.py)

```
max_attempts = 3
for attempt in range(max_attempts):
    reading = vlm.read(image)         # may itself loop if consensus fails
    extraction = extractor.extract(reading, problem_text)
    lean = theorem_from(...)
    status, stderr = lean_runner.typecheck(lean)
    if status is LeanStatus.OK:
        return PipelineResult(...)
    # status is TYPE_ERROR or TIMEOUT
    blame = judge.attribute_blame({"reading": reading, "extraction": extraction}, stderr)
    # Re-run ONLY the blamed stage on next attempt
    if blame.stage == "vlm":
        vlm_force_retry = True
    elif blame.stage == "extraction":
        extraction_force_retry = True
    else:
        break  # emission bug — not recoverable by re-running
return PipelineResult(...)  # last attempt's result + blame history
```

## Test strategy

Every new LLM-backed module is tested with **mocked clients** (same pattern as existing `test_gemini_adapter.py`). Live-API tests go under `@pytest.mark.live`. Real Lake builds stay `@pytest.mark.lean`.

For consensus modules specifically:
- Mock two sub-adapters returning known readings
- Mock judge returning known verdicts
- Assert consensus-aware merge/retry logic works
- Test the "max retries exhausted" path

## Non-goals

- Structured output (Gemini's JSON schema mode) — out of scope; we parse raw JSON
- Cost optimization — user explicitly said cost doesn't matter
- Replacing `theorem_from` with LLM — the string-formatting layer stays deterministic
- Proof generation (still `sorry`) — same scope as original project

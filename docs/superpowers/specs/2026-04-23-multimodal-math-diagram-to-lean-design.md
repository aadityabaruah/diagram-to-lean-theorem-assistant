# Multimodal Math: Diagram-to-Lean Theorem Assistant — Design Spec

**Date:** 2026-04-23
**Authors:** Aaditya Baruah (abb2237), Sahasra Kokkula (sk5652)
**Status:** Approved for implementation planning

## 1. Context

The submitted proposal ("Multimodal Math: Translating Visual Diagrams into Lean") describes a Python pipeline that converts geometry diagrams into Lean 4 theorem statements via a vision-language model. The existing repository contains a text-only scaffold prototype that addresses reviewer feedback *on paper* (modular schema, ranked goal candidates, assumption/goal separation, benchmark structure), but does not actually use a VLM, does not read images, and emits trivial `theorem foo : True := by trivial` Lean stubs.

Reviewer comments (Natalie Parham, Apr 9) require:
- A concrete mechanism for determining the *goal* theorem statement, not just assumptions.
- Evaluation against examples with known correct theorems.
- Explicit handling of diagram ambiguity, not silent inference.
- A comparison with Lean Blueprint.
- Both working software and visualizer ideas.

This spec lays out the target architecture for closing the gap between the text-only prototype and a working VLM→Lean pipeline end-to-end.

## 2. Scope

**In scope:**
- A pluggable VLM adapter layer with two implementations: a real Gemini adapter (multi-step prompting) and a deterministic fixture-based adapter for tests and reproducible evaluation.
- A real Lean 4 project (`lean_project/`) using mathlib4. Generated theorem files are type-checked via `lake build`.
- A Streamlit UI with two pages: interactive run (upload/pick image, confirm assumptions, pick goal) and benchmark (automated run over gold examples, metrics display).
- A benchmark with two tiers: 5 synthetic examples (matplotlib-generated, committed) and ~10 curated Geometry3K examples (fetched, gitignored). Each synthetic example has a committed VLM fixture.
- Evaluation metrics: assumption precision/recall, top-k goal accuracy (k=1,3,5), Lean statement validity rate, category-level breakdown.
- TDD-style test coverage with three tiers (fast default, `-m lean` smoke, `-m live` Gemini).

**Explicitly out of scope:**
- Actual theorem proving. Generated Lean files close with `:= by sorry`. Proposal is "correctly initialize the theorem," not "prove it."
- Persistent run history or a database. All state is files on disk or Streamlit session memory.
- Streamlit UI unit tests. UI layer stays thin; logic lives in tested modules.
- LeanGeo. Primary backend is mathlib4's `Mathlib.Geometry.Euclidean.*`. LeanGeo is noted as future work.
- Full Geometry3K or full academic geometry datasets. We use ~10 curated examples to show the system handles real diagrams.
- A growing Lean library. Each pipeline run overwrites `Generated.lean` — no accumulating theorem database.

## 3. Architecture

Four layers, each with clear boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                     │
│  Page 1: Interactive (upload → review → Lean)               │
│  Page 2: Benchmark   (run gold examples, show metrics)      │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  Pipeline orchestrator  (pipeline.py)                       │
│  Stage 1: Diagram Understanding                             │
│  Stage 2: Candidate Generation (assumptions + goals)        │
│  Stage 3: Validation + Lean Output                          │
└───────┬──────────────────┬───────────────────┬──────────────┘
        ▼                  ▼                   ▼
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ VLM adapter  │   │ assumption_      │   │ lean_export +    │
│ (protocol)   │   │  extraction +    │   │ lean_runner      │
│ Gemini /     │   │ goal_generation  │   │ (subprocess)     │
│ Fixture      │   │                  │   │ → lean_project/  │
└──────────────┘   └──────────────────┘   └──────────────────┘
        │                                          │
        └──────── reads ──────────▶   benchmark fixtures,
                                      gold data, images
```

**Key architectural properties:**

- **VLM adapter is an interface** (`DiagramUnderstander` Protocol). `GeminiAdapter` is real; `FixtureAdapter` returns pre-recorded JSON. Tests and benchmark evaluation use fixtures by default. The UI lets the user opt into the real adapter.
- **Lean runs in a subprocess.** Python never imports anything Lean. `lean_runner` writes `Generated.lean`, invokes `lake build`, parses exit code + stderr. Graceful degradation if `lake` is not on PATH.
- **Stages are pure functions at their boundaries.** No hidden state. Streamlit just wires user selections into the next call.
- **Benchmark JSON is the single source of truth for evaluation.** Extended from the existing `examples/benchmark.json` with image paths, VLM fixture paths, gold relations and marks, and category tags.

## 4. VLM prompting strategy

Three sequential calls per image — each narrowly scoped to reduce hallucination (reviewer concern: "VLM assumes symmetry without justification"):

1. **Objects** — "List named points, segments, lines, and circles visible in this diagram. Return JSON array."
2. **Relations** — Given those objects: "List visible geometric relations: collinearity, perpendicularity, parallelism, incidence, intersection. Return JSON array."
3. **Marks** — "List visual marks: equal-length tick marks, right-angle boxes, parallel arrows, labeled angles. Return JSON array."

The adapter merges all three into a `DiagramReading`. Each call gets a JSON schema via structured output where supported, falling back to strict-prompt + parse with one retry.

Model: `gemini-3.1-flash-lite-preview` (as specified by the user). If that name fails to resolve at runtime, the adapter falls back to a stable current model (configurable via `GEMINI_MODEL` env var).

## 5. Module layout

```
diagram-to-lean-theorem-assistant/
├── app.py                              # Streamlit entry point
├── pyproject.toml                      # deps pinned (google-genai, pillow, matplotlib, streamlit, python-dotenv, pytest, pytest-mock)
├── .env                                # gitignored; GEMINI_API_KEY
├── examples/
│   ├── benchmark.json                  # expanded with image paths, vlm fixtures, categories
│   ├── images/
│   │   ├── synthetic/*.png             # matplotlib-generated, committed
│   │   └── geometry3k/*.png            # fetched, gitignored
│   └── vlm_fixtures/*.json             # committed, deterministic VLM outputs
├── lean_project/                       # Lake project via `lake new`
│   ├── lakefile.lean
│   ├── lean-toolchain
│   └── DiagramTheorems/
│       ├── Basic.lean                  # imports + helpers, committed
│       └── Generated.lean              # overwritten per run, gitignored
├── scripts/
│   ├── generate_synthetic.py
│   ├── fetch_geometry3k.py
│   └── record_vlm_fixtures.py
├── src/diagram_theorem_assistant/
│   ├── __init__.py
│   ├── schema.py                       # BenchmarkExample, DiagramReading, PipelineResult, LeanStatus
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── base.py                     # DiagramUnderstander Protocol
│   │   ├── gemini.py                   # GeminiAdapter (multi-step)
│   │   ├── fixture.py                  # FixtureAdapter
│   │   └── prompts.py                  # three prompt templates
│   ├── assumption_extraction.py        # text + marks + relations → assumptions
│   ├── goal_generation.py              # text extraction + richer templates
│   ├── lean_export.py                  # emits real mathlib geometry source
│   ├── lean_runner.py                  # subprocess wrapper for `lake build`
│   ├── pipeline.py                     # orchestrator
│   ├── metrics.py                      # precision/recall/top-k/lean-validity
│   └── demo.py                         # smoke-test CLI (kept)
└── tests/
    ├── test_schema.py                  # expanded
    ├── test_assumption_extraction.py   # expanded
    ├── test_goal_generation.py         # expanded
    ├── test_lean_export.py             # expanded
    ├── test_metrics.py                 # renamed from test_evaluation.py
    ├── test_demo.py                    # kept
    ├── test_pipeline.py                # NEW — end-to-end w/ FixtureAdapter
    ├── test_lean_runner.py             # NEW — mocked + one real smoke test
    ├── test_gemini_live.py             # NEW — @pytest.mark.live
    └── vlm/
        ├── test_fixture_adapter.py
        └── test_gemini_adapter.py
```

## 6. Core data types

```python
@dataclass(frozen=True)
class DiagramMark:
    type: str               # "right_angle", "equal_length", "parallel", ...
    points: list[str]

@dataclass(frozen=True)
class DiagramReading:
    objects: list[str]      # ["point A", "segment AB", ...]
    relations: list[str]    # ["collinear A B C", "perpendicular AB AC", ...]
    marks: list[DiagramMark]
    raw_vlm_output: dict    # kept for auditing

@dataclass(frozen=True)
class PipelineResult:
    reading: DiagramReading
    candidate_assumptions: list[str]
    confirmed_assumptions: list[str]
    goal_candidates: list[str]
    selected_goal: str
    lean_source: str
    lean_status: LeanStatus
    lean_stderr: str | None

class LeanStatus(StrEnum):
    OK = "ok"
    TYPE_ERROR = "type_error"
    UNAVAILABLE = "unavailable"
```

## 7. Data flow

### Interactive (Streamlit page 1)

1. User picks image (benchmark example or upload).
2. VLM adapter returns `DiagramReading`. Gemini path makes three sequential calls; fixture path loads JSON.
3. `assumption_extraction` produces candidate assumptions from problem text, marks, and relations.
4. UI shows checkboxes — user confirms/rejects/adds.
5. `goal_generation` either extracts from text (if "prove"/"show"/"find" phrase present) or ranks templates based on objects and confirmed assumptions.
6. UI shows radio list — user picks goal.
7. `lean_export` emits source; `lean_runner` runs `lake build`; result rendered with status badge and download button.

### Benchmark (Streamlit page 2)

1. Load all `BenchmarkExample`s from JSON.
2. For each: use `FixtureAdapter` (deterministic, no API cost), auto-confirm `gold_assumptions`, auto-select `gold_goal`, run full pipeline.
3. `metrics.summarize` computes precision, recall, top-k accuracy, Lean validity rate, per-category breakdown.
4. Render overall table + per-example drill-down.

### VLM fixture capture (ad-hoc)

`scripts/record_vlm_fixtures.py` calls the real Gemini adapter on benchmark images that lack fixtures, serializes results to `examples/vlm_fixtures/`, commits them. Run explicitly; costs API calls.

## 8. Error handling

### Exception taxonomy

```python
class PipelineError(Exception): ...
class VLMError(PipelineError): ...
class FixtureNotFoundError(PipelineError): ...
class LeanError(PipelineError): ...
```

### Failure matrix

| Failure | Caught where | Response |
|---|---|---|
| `GEMINI_API_KEY` missing | `GeminiAdapter.__init__` | `VLMError` with setup hint. UI falls back to fixture-only mode. |
| Gemini network/timeout | `GeminiAdapter.read` | One retry with 2s backoff, then `VLMError`. |
| Gemini rate-limit (429) | `GeminiAdapter.read` | Raise `VLMError` — don't silent-wait. |
| Non-JSON response | `GeminiAdapter._parse` | Retry once with stricter prompt, then `VLMError` with raw text attached. |
| Wrong-schema JSON | `GeminiAdapter._parse` | Drop unknown keys, default missing lists to `[]`. Log warning. |
| Image missing/corrupt | `pipeline.run` entry | `FileNotFoundError` / `PIL.UnidentifiedImageError`. |
| Fixture missing | `FixtureAdapter.read` | `FixtureNotFoundError` with exact regeneration command. |
| `lake` not on PATH | `LeanRunner.__init__` | All calls return `LeanStatus.UNAVAILABLE`. Pipeline completes. |
| `lake build` type-error | `LeanRunner.typecheck` | **Expected result**, return `LeanStatus.TYPE_ERROR` + stderr. |
| `lake build` timeout (60s) | `LeanRunner.typecheck` | Kill subprocess, `TYPE_ERROR` with "build timed out". |
| `lake build` crashes (other) | `LeanRunner.typecheck` | Raise `LeanError`. |
| Empty object list from VLM | `pipeline.run` | Continue — empty candidates. UI offers manual entry. |
| User confirms zero assumptions | `pipeline.run` | Continue — valid for diagram-only examples. |
| No goal candidates | `pipeline.run` | Return result with empty `goal_candidates`, `selected_goal=""`. UI prompts for custom goal. |

### Principles

- **Always return `PipelineResult`** from `pipeline.run`, or raise a typed `PipelineError` subclass. Streamlit handles exceptions only at the top boundary.
- **Lean status is a result, not an exception.** Type errors are expected for many examples — that's what we're measuring.
- **Validate at system boundaries, trust inside.** VLM adapter defensively parses Gemini JSON. Pipeline trusts the `DiagramReading` it receives.

## 9. Testing strategy

Three tiers:

```
pytest                             # fast unit + integration (default, < 10s)
pytest -m lean                     # includes real `lake build` smoke tests
pytest -m live                     # includes real Gemini API calls
pytest -m "not lean and not live"  # strict CI filter
```

Per-module coverage:

- **Fast tier:** schema round-trips, assumption/goal/lean-export logic, metrics math, mocked Gemini adapter (verifies prompt sequence and parsing), fixture adapter, end-to-end pipeline on all 5 synthetic examples with `FixtureAdapter`, `lean_runner` with mocked subprocess.
- **`@pytest.mark.lean`:** one real `lake build` on a known-good generated `.lean`. Trusts Lean for the rest.
- **`@pytest.mark.live`:** one real Gemini call on a synthetic image; asserts response has expected fields. Ad-hoc, not in CI.

TDD discipline matches the existing plan: failing test → implement → green → refactor.

VLM fixtures do double duty: they power tests AND the Streamlit benchmark page. No forked test data.

Coverage target ~85%, measured with `pytest-cov`, not enforced as CI gate.

## 10. Deliberate non-choices

- **No database / persistence.** Everything is files on disk. Session state is ephemeral.
- **`Generated.lean` is disposable.** Every run overwrites. We do not accumulate theorems into a growing library.
- **No HTTP caching.** Fixtures are the cache; captured explicitly via `record_vlm_fixtures.py`.
- **`evaluation.py` → `metrics.py`.** Renamed and expanded. Old file removed.
- **`lean_export.py` is rewritten.** The current `:= by trivial` stub doesn't satisfy the proposal's claim; must emit real geometry.

## 11. Open risks

| Risk | Mitigation |
|---|---|
| `gemini-3.1-flash-lite-preview` model name may not resolve | `GEMINI_MODEL` env var override; fall back to a stable current model with a warning. |
| Gemini's JSON output is unreliable even with structured prompts | Multi-step prompting narrows scope of each call; one retry with stricter prompt; fixtures decouple tests from live variance. |
| mathlib4's geometry modules may lack primitives we need for non-trivial theorems | Spec limits scope to type-checking, not proving; `Generated.lean` closes with `sorry`. Missing definitions become a visible data point, not a blocker. |
| Lean build times (first mathlib build) | `lake exe cache get` downloads prebuilt mathlib (~15 min once). Documented in setup. Test timeout = 60s per call. |
| Geometry3K license / distribution | We fetch ~10 curated samples under the dataset's stated license; sample list + hashes committed, raw images gitignored. |
| Streamlit's state loss on refresh feels bad | Accepted; documented. A session persistence story is deferred. |

## 12. Comparison with Lean Blueprint

Lean Blueprint is a project-management tool for formalization projects — it links informal math writeups to Lean lemmas. This project operates *earlier* in the pipeline: it starts from a visual geometry problem and produces explicit assumption and theorem candidates. Lean is the final authority for whether a selected statement is valid.

They are complementary. A mature version of this project could export verified theorem statements and dependency graphs into a blueprint-style document.

## 13. Visualizer extensions (future)

After the core pipeline works, the UI can grow in three directions:

- **Diagram overlay:** draw detected points, segments, marks, and relations on top of the input image so the user sees what the VLM extracted.
- **Assumption/goal panel:** three-column layout — assumptions, ranked goals, Lean output — for at-a-glance review.
- **Proof-graph visualizer:** show the selected theorem's required lemmas and Lean dependency structure, à la Lean Blueprint.

These are secondary until the core pipeline can produce and evaluate theorem candidates reliably.

## 14. Final deliverables

- Revised proposal addressing reviewer feedback (already committed in `docs/project-proposal.md`; may receive minor updates).
- Working Python package with all modules in §5.
- Lean 4 project that type-checks generated files.
- Streamlit UI with interactive + benchmark pages.
- Benchmark JSON with ≥5 synthetic + ≥10 Geometry3K examples, each with gold assumptions, gold goal, acceptable alternatives.
- VLM fixtures committed for synthetic examples.
- Metrics reported for all benchmark examples.
- Setup documentation covering Lean, mathlib cache, Python venv, and API key handling.

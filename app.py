from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from diagram_theorem_assistant.consensus.judge import JudgeLLM
from diagram_theorem_assistant.consensus.vlm_consensus import ConsensusVLMAdapter
from diagram_theorem_assistant.extraction.base import Extraction
from diagram_theorem_assistant.extraction.claude_extractor import ClaudeExtractor
from diagram_theorem_assistant.extraction.consensus import ConsensusExtractor
from diagram_theorem_assistant.extraction.gemini_extractor import GeminiExtractor
from diagram_theorem_assistant.metrics import (
    assumption_precision,
    assumption_recall,
    category_breakdown,
    lean_validity_rate,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.pipeline import (
    candidate_assumptions,
    run_consensus_pipeline,
    run_pipeline_from_reading,
)
from diagram_theorem_assistant.direct.claude_solver import ClaudeDirectSolver
from diagram_theorem_assistant.proof_generation.claude_prover import ClaudeProver
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus, load_benchmark
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
from diagram_theorem_assistant.vlm.claude import ClaudeAdapter
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO_ROOT / "examples" / "benchmark.json"
FIXTURES_DIR = REPO_ROOT / "examples" / "vlm_fixtures"
IMAGES_DIR = REPO_ROOT / "examples" / "images" / "synthetic"
LEAN_PROJECT = REPO_ROOT / "lean_project"
GENERATED_LEAN = LEAN_PROJECT / "DiagramTheorems" / "Generated.lean"


def _lean_runner() -> LeanRunner | None:
    if not LEAN_PROJECT.exists():
        return None
    return LeanRunner(project_root=LEAN_PROJECT, generated_path=GENERATED_LEAN)


GEMINI_MODEL_CHOICES = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "Custom…",
]


def _adapter(mode: str, api_key: str, model: str):
    if mode == "Gemini (live)":
        if not api_key:
            st.error("GEMINI_API_KEY missing. Set it in .env.")
            st.stop()
        return GeminiAdapter(api_key=api_key, model=model)
    return FixtureAdapter(FIXTURES_DIR)


def _status_badge(status: LeanStatus) -> str:
    if status is LeanStatus.OK:
        return ":green[✓ Type-checked by Lean]"
    if status is LeanStatus.TYPE_ERROR:
        return ":red[✗ Lean type error]"
    if status is LeanStatus.TIMEOUT:
        return ":orange[⏱ Lean build timed out]"
    return ":orange[Lean unavailable]"


def interactive_page() -> None:
    st.header("Interactive: diagram → Lean theorem")

    examples = load_benchmark(BENCHMARK_PATH)
    choices = {e.id: e for e in examples}
    selection = st.selectbox("Benchmark example:", list(choices.keys()))
    example = choices[selection]

    image_path = REPO_ROOT / example.image
    st.image(str(image_path), caption=example.id, width=400)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    mode = st.sidebar.radio(
        "VLM adapter",
        options=["Fixture (deterministic)", "Gemini (live)"],
        help="Fixture mode uses the committed VLM fixture; live mode calls Gemini.",
    )
    st.sidebar.caption(f"API key set: {bool(api_key)}")

    default_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    default_index = (
        GEMINI_MODEL_CHOICES.index(default_model)
        if default_model in GEMINI_MODEL_CHOICES
        else len(GEMINI_MODEL_CHOICES) - 1  # Custom…
    )
    chosen = st.sidebar.selectbox(
        "Gemini model",
        options=GEMINI_MODEL_CHOICES,
        index=default_index,
        disabled=(mode != "Gemini (live)"),
        help="flash-lite = cheapest; pro = best accuracy on hand-drawn / real diagrams.",
    )
    if chosen == "Custom…":
        model = st.sidebar.text_input("Custom model name", value=default_model)
    else:
        model = chosen

    st.sidebar.divider()
    st.sidebar.subheader("Consensus mode (experimental)")

    consensus_enabled = st.sidebar.checkbox(
        "Enable dual-provider consensus",
        value=False,
        help="Run Gemini + Claude at each stage, judge LLM arbitrates. Requires both API keys.",
    )

    if consensus_enabled:
        claude_model = st.sidebar.selectbox(
            "Claude model",
            options=["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
            index=0,
        )
        judge_model = st.sidebar.selectbox(
            "Judge model",
            options=["gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview"],
            index=0,
        )
        max_retries = st.sidebar.slider("Max retries on Lean failure", 0, 5, 3)
        attempt_proof = st.sidebar.checkbox(
            "Attempt LLM-generated proof (experimental)",
            value=False,
            help="After generating the theorem statement, ask Claude to replace `sorry` with a real proof, iterating lake-build errors. Only closes trivial cases today — mathlib geometry automation is an open problem.",
        )
        prover_max_attempts = (
            st.sidebar.slider("Prover max attempts", 1, 6, 3)
            if attempt_proof
            else 0
        )
    else:
        claude_model = "claude-opus-4-7"
        judge_model = "gemini-3.1-pro-preview"
        max_retries = 0
        attempt_proof = False
        prover_max_attempts = 0

    # Standalone prover toggle for the deterministic Interactive flow —
    # available even when consensus mode is off.
    st.sidebar.divider()
    st.sidebar.subheader("Proof generation")
    prove_in_interactive = st.sidebar.checkbox(
        "Try to replace `sorry` with a real proof",
        value=False,
        help="After 'Generate Lean theorem', send the file to Claude Opus 4.7 and iterate "
             "lake-build errors up to N times. Geometry proofs are hard — works on Pythagoras "
             "and similar; isosceles base angles typically stays as sorry.",
    )
    interactive_prover_attempts = (
        st.sidebar.slider("Proof attempts", 1, 6, 4) if prove_in_interactive else 0
    )

    st.subheader("Problem text")
    problem_text = st.text_area("Optional problem statement", value=example.problem_text, height=80)

    # ---------- Direct Claude solver (image → Claude → verified Lean) ----------
    st.markdown("---")
    st.subheader("Direct Claude solve (image → Lean, no extraction stage)")
    st.caption(
        "Sends the image and problem text directly to Claude Opus 4.7, iterates "
        "`lake build` errors back to it. No regex extraction stage."
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        direct_attempts = st.slider("Direct-solve attempts", 1, 8, 5, key="direct-attempts")
    with col_b:
        statement_only = st.checkbox(
            "Statement only (sorry OK)",
            value=False,
            help="Ask Claude to formalize the theorem statement — proper hypothesis "
                 "types, real goal — and accept `sorry` for the proof body. Faster, "
                 "more reliable; turn off to demand a real proof.",
        )
    if st.button("Run direct Claude solve"):
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            st.error("ANTHROPIC_API_KEY missing in .env")
            st.stop()
        runner = _lean_runner()
        if runner is None:
            st.error("Lean project not found at lean_project/. Cannot verify the proof.")
            st.stop()
        spinner_msg = (
            f"Asking Claude Opus 4.7 to formalize the theorem statement "
            f"(up to {direct_attempts} rounds)…"
            if statement_only
            else f"Sending image to Claude Opus 4.7 + lake-build oracle (up to {direct_attempts} rounds)…"
        )
        with st.spinner(spinner_msg):
            try:
                solver = ClaudeDirectSolver(api_key=anthropic_key)
                source, status, stderr = solver.solve(
                    image_path=image_path,
                    problem_text=problem_text,
                    theorem_name=example.lean_theorem_name,
                    lean_runner=runner,
                    max_attempts=direct_attempts,
                    allow_sorry=statement_only,
                )
            except VLMError as exc:
                st.error(str(exc))
                st.stop()
        proved = status is LeanStatus.OK and "sorry" not in source
        compiles = status is LeanStatus.OK
        st.markdown(_status_badge(status))
        if statement_only and compiles:
            st.success("✓ Theorem formalized — statement type-checks. Proof body is `sorry` (by design).")
        elif proved:
            st.success("✓ Proof closed — `sorry`-free, verified by `lake build`.")
        elif compiles and "sorry" in source:
            st.warning("Compiles but contains `sorry` — toggle 'Statement only' off to demand a real proof.")
        else:
            st.warning("Did not compile cleanly. See stderr below.")
        if stderr:
            st.code(stderr, language="text")
        st.code(source, language="lean")
        st.download_button(
            "Download .lean",
            data=source,
            file_name=f"{example.lean_theorem_name}.lean",
            key="direct-download",
        )
        st.stop()  # don't render the deterministic flow below for this run
    st.markdown("---")

    if st.button("Run VLM extraction"):
        try:
            adapter = _adapter(mode, api_key, model)
            fixture_path = REPO_ROOT / example.vlm_fixture if example.vlm_fixture else None
            reading = adapter.read(image_path, fixture_path=fixture_path)
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["reading"] = reading
        st.session_state["problem_text"] = problem_text
        st.session_state["example"] = example
        st.session_state["vlm_mode"] = mode
        st.session_state["vlm_model"] = model if mode == "Gemini (live)" else "(fixture)"

    if consensus_enabled:
        if st.button("Run consensus pipeline (end-to-end)"):
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not gemini_key or not anthropic_key:
                st.error("Consensus mode requires both GEMINI_API_KEY and ANTHROPIC_API_KEY in .env")
                st.stop()

            with st.spinner("Running consensus pipeline — this may take 30-90 seconds..."):
                try:
                    gemini_vlm = GeminiAdapter(api_key=gemini_key, model=model)
                    claude_vlm = ClaudeAdapter(api_key=anthropic_key, model=claude_model)
                    judge = JudgeLLM(api_key=gemini_key, model=judge_model)
                    vlm = ConsensusVLMAdapter(gemini_vlm, claude_vlm, judge, max_attempts=3)

                    gemini_ext = GeminiExtractor(api_key=gemini_key, model=model)
                    claude_ext = ClaudeExtractor(api_key=anthropic_key, model=claude_model)
                    extractor = ConsensusExtractor(gemini_ext, claude_ext, judge, max_attempts=3)

                    prover = (
                        ClaudeProver(api_key=anthropic_key, model=claude_model)
                        if attempt_proof
                        else None
                    )
                    result = run_consensus_pipeline(
                        image_path=image_path,
                        problem_text=problem_text,
                        vlm=vlm,
                        extractor=extractor,
                        judge=judge,
                        lean_runner=_lean_runner(),
                        theorem_name=example.lean_theorem_name,
                        max_retries=max_retries,
                        prover=prover,
                        prover_max_attempts=prover_max_attempts,
                    )
                except (VLMError, FixtureNotFoundError) as exc:
                    st.error(str(exc))
                    st.stop()

            st.subheader("Consensus VLM reading")
            st.caption(f"Gemini: `{model}`  |  Claude: `{claude_model}`  |  Judge: `{judge_model}`")
            st.json(result.reading.to_dict())

            st.subheader("Consensus extraction")
            st.write("**Assumptions:**")
            for a in result.confirmed_assumptions:
                st.markdown(f"- `{a}`")
            st.write(f"**Goal:** `{result.selected_goal}`")

            st.subheader("Lean output")
            st.markdown(_status_badge(result.lean_status))
            if result.lean_status in (LeanStatus.TYPE_ERROR, LeanStatus.TIMEOUT) and result.lean_stderr:
                st.code(result.lean_stderr, language="text")
            st.code(result.lean_source, language="lean")
            st.download_button(
                "Download .lean",
                data=result.lean_source,
                file_name=f"{example.lean_theorem_name}.lean",
            )

    reading = st.session_state.get("reading")
    if reading is None:
        return

    st.subheader("VLM reading")
    if st.session_state.get("vlm_model"):
        st.caption(f"Source: {st.session_state['vlm_model']}")
    st.json(reading.to_dict())

    candidate = candidate_assumptions(reading, st.session_state.get("problem_text", ""))

    st.subheader("Confirm assumptions")
    confirmed = [a for a in candidate if st.checkbox(a, value=True, key=f"chk-{a}")]
    extra = st.text_input("Additional assumption (optional)", key="extra-assumption")
    if extra.strip():
        confirmed.append(extra.strip())

    if st.button("Generate Lean theorem"):
        runner = _lean_runner()
        result = run_pipeline_from_reading(
            reading=reading,
            problem_text=st.session_state.get("problem_text", ""),
            confirmed_assumptions=confirmed,
            lean_runner=runner,
            theorem_name=st.session_state["example"].lean_theorem_name,
        )

        # Optional: ask Claude to replace `sorry` with a real proof
        if (
            prove_in_interactive
            and runner is not None
            and "sorry" in result.lean_source
        ):
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not anthropic_key:
                st.warning("ANTHROPIC_API_KEY missing — skipping proof attempt.")
            else:
                with st.spinner(
                    f"Attempting proof via Claude Opus 4.7 (up to {interactive_prover_attempts} rounds)…"
                ):
                    try:
                        prover = ClaudeProver(api_key=anthropic_key)
                        new_source, new_status, new_stderr = prover.prove(
                            result.lean_source,
                            lean_runner=runner,
                            max_attempts=interactive_prover_attempts,
                        )
                        # Rebuild result with the prover's output
                        from dataclasses import replace
                        result = replace(
                            result,
                            lean_source=new_source,
                            lean_status=new_status,
                            lean_stderr=new_stderr,
                        )
                    except VLMError as exc:
                        st.error(f"Prover error: {exc}")

        st.subheader("Goal candidates")
        if result.goal_candidates:
            st.write(result.goal_candidates)
        st.write(f"**Selected:** `{result.selected_goal or '(none)'}`")
        if "sorry" in result.lean_source:
            st.info("Proof body is `sorry` — statement is real, proof not provided. Toggle 'Try to replace sorry' in the sidebar to attempt a real proof via Claude.")
        else:
            st.success("Proof closed — `sorry`-free, verified by `lake build`.")

        st.subheader("Lean output")
        st.markdown(_status_badge(result.lean_status))
        if result.lean_status in (LeanStatus.TYPE_ERROR, LeanStatus.TIMEOUT) and result.lean_stderr:
            st.code(result.lean_stderr, language="text")
        st.code(result.lean_source, language="lean")
        st.download_button(
            "Download .lean",
            data=result.lean_source,
            file_name=f"{st.session_state['example'].lean_theorem_name}.lean",
        )


def benchmark_page() -> None:
    st.header("Benchmark: gold-confirmed run")

    examples = load_benchmark(BENCHMARK_PATH)
    adapter = FixtureAdapter(FIXTURES_DIR)
    runner = _lean_runner()

    rows = []
    per_category: dict[str, list] = {}
    for example in examples:
        fixture_path = REPO_ROOT / example.vlm_fixture
        try:
            reading = adapter.read(REPO_ROOT / example.image, fixture_path=fixture_path)
            result = run_pipeline_from_reading(
                reading=reading,
                problem_text=example.problem_text,
                confirmed_assumptions=example.gold_assumptions,
                lean_runner=runner,
                theorem_name=example.lean_theorem_name,
            )
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(f"{example.id}: {exc}")
            continue

        rows.append(
            {
                "id": example.id,
                "category": example.category,
                "assumption_precision": assumption_precision(
                    result.confirmed_assumptions, example.gold_assumptions
                ),
                "assumption_recall": assumption_recall(
                    result.confirmed_assumptions, example.gold_assumptions
                ),
                "top_1": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 1),
                "top_3": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 3),
                "top_5": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 5),
                "lean_status": result.lean_status.value,
            }
        )
        per_category.setdefault(example.category, []).append(result)

    st.subheader("Per-example")
    st.dataframe(rows)

    st.subheader("Aggregate")
    all_results = [r for group in per_category.values() for r in group]
    st.metric("Lean validity rate", f"{lean_validity_rate(all_results) * 100:.1f}%")
    st.json(category_breakdown(per_category))


def main() -> None:
    st.set_page_config(
        page_title="Diagram-to-Lean Theorem Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    page = st.sidebar.selectbox("Page", ["Interactive", "Benchmark"])
    if page == "Interactive":
        interactive_page()
    else:
        benchmark_page()


if __name__ == "__main__":
    main()

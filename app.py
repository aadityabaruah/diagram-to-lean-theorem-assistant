from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from diagram_theorem_assistant.assumption_extraction import infer_assumptions
from diagram_theorem_assistant.goal_generation import extract_goal_from_text, rank_goal_candidates
from diagram_theorem_assistant.lean_export import theorem_from
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.metrics import (
    assumption_precision,
    assumption_recall,
    category_breakdown,
    lean_validity_rate,
    top_k_goal_accuracy,
)
from diagram_theorem_assistant.pipeline import run_pipeline
from diagram_theorem_assistant.schema import LeanStatus, load_benchmark
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
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


def _adapter(mode: str, api_key: str):
    if mode == "Gemini (live)":
        if not api_key:
            st.error("GEMINI_API_KEY missing. Set it in .env.")
            st.stop()
        return GeminiAdapter(api_key=api_key)
    return FixtureAdapter(FIXTURES_DIR)


def _status_badge(status: LeanStatus) -> str:
    if status is LeanStatus.OK:
        return ":green[✓ Type-checked by Lean]"
    if status is LeanStatus.TYPE_ERROR:
        return ":red[✗ Lean type error]"
    return ":orange[Lean unavailable]"


def interactive_page() -> None:
    st.header("Interactive: diagram → Lean theorem")

    examples = load_benchmark(BENCHMARK_PATH)
    choices = {e.id: e for e in examples}
    selection = st.selectbox("Benchmark example:", list(choices.keys()))
    example = choices[selection]

    image_path = REPO_ROOT / example.image
    st.image(str(image_path), caption=example.id, use_column_width=False, width=400)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    mode = st.sidebar.radio(
        "VLM adapter",
        options=["Fixture (deterministic)", "Gemini (live)"],
        help="Fixture mode uses the committed VLM fixture; live mode calls Gemini.",
    )
    st.sidebar.caption(f"API key set: {bool(api_key)}")

    st.subheader("Problem text")
    problem_text = st.text_area("Optional problem statement", value=example.problem_text, height=80)

    if st.button("Run VLM extraction"):
        try:
            adapter = _adapter(mode, api_key)
            fixture_path = REPO_ROOT / example.vlm_fixture if example.vlm_fixture else None
            reading = adapter.read(image_path, fixture_path=fixture_path)
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["reading"] = reading
        st.session_state["problem_text"] = problem_text
        st.session_state["example"] = example

    reading = st.session_state.get("reading")
    if reading is None:
        return

    st.subheader("VLM reading")
    st.json(reading.to_dict())

    candidate = infer_assumptions(
        problem_text=st.session_state.get("problem_text", ""),
        objects=reading.objects,
        diagram_marks=[mark.to_dict() for mark in reading.marks],
        confirmed_assumptions=None,
        relations=reading.relations,
    )

    st.subheader("Confirm assumptions")
    confirmed = [a for a in candidate if st.checkbox(a, value=True, key=f"chk-{a}")]
    extra = st.text_input("Additional assumption (optional)", key="extra-assumption")
    if extra.strip():
        confirmed.append(extra.strip())

    extracted_goal = extract_goal_from_text(st.session_state.get("problem_text", ""))
    ranked = rank_goal_candidates(reading.objects, confirmed)
    candidates = [extracted_goal, *ranked] if extracted_goal else ranked
    candidates = [c for c in candidates if c]

    st.subheader("Select goal")
    if not candidates:
        selected = st.text_input("Enter goal manually", key="manual-goal")
    else:
        selected = st.radio("Ranked goal candidates", candidates, key="goal-radio")

    if st.button("Generate Lean theorem") and selected:
        source = theorem_from(
            name=st.session_state["example"].lean_theorem_name,
            assumptions=confirmed,
            goal=selected,
        )
        runner = _lean_runner()
        if runner:
            status, stderr = runner.typecheck(source)
        else:
            status, stderr = LeanStatus.UNAVAILABLE, None

        st.subheader("Lean output")
        st.markdown(_status_badge(status))
        if stderr:
            st.code(stderr, language="text")
        st.code(source, language="lean")
        st.download_button(
            "Download .lean",
            data=source,
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
            result = run_pipeline(
                image_path=REPO_ROOT / example.image,
                problem_text=example.problem_text,
                confirmed_assumptions=example.gold_assumptions,
                vlm=adapter,
                vlm_fixture_path=fixture_path,
                lean_runner=runner,
                theorem_name=example.lean_theorem_name,
            )
        except (VLMError, FixtureNotFoundError) as exc:
            st.error(f"{example.id}: {exc}")
            continue

        rows.append({
            "id": example.id,
            "category": example.category,
            "assumption_precision": assumption_precision(result.confirmed_assumptions, example.gold_assumptions),
            "assumption_recall": assumption_recall(result.confirmed_assumptions, example.gold_assumptions),
            "top_1": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 1),
            "top_3": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 3),
            "top_5": top_k_goal_accuracy(result.goal_candidates, example.gold_goal, 5),
            "lean_status": result.lean_status.value,
        })
        per_category.setdefault(example.category, []).append(result)

    st.subheader("Per-example")
    st.dataframe(rows)

    st.subheader("Aggregate")
    all_results = [r for group in per_category.values() for r in group]
    st.metric("Lean validity rate", f"{lean_validity_rate(all_results) * 100:.1f}%")
    st.json(category_breakdown(per_category))


def main() -> None:
    st.set_page_config(page_title="Diagram-to-Lean Theorem Assistant", layout="wide")
    page = st.sidebar.selectbox("Page", ["Interactive", "Benchmark"])
    if page == "Interactive":
        interactive_page()
    else:
        benchmark_page()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from diagram_theorem_assistant.consensus.judge import JudgeLLM
from diagram_theorem_assistant.consensus.vlm_consensus import ConsensusVLMAdapter
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
    run_consensus_pipeline,
    run_pipeline_from_reading,
)
from diagram_theorem_assistant.lean_runner import LeanRunner
from diagram_theorem_assistant.schema import LeanStatus, load_benchmark
from diagram_theorem_assistant.vlm.base import FixtureNotFoundError, VLMError
from diagram_theorem_assistant.vlm.claude import ClaudeAdapter
from diagram_theorem_assistant.vlm.fixture import FixtureAdapter
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter

load_dotenv()

# On Streamlit Community Cloud, secrets come from st.secrets (not env vars).
# Mirror them into os.environ so the rest of the code stays env-var based.
for _k in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_MODEL", "CLAUDE_MODEL", "JUDGE_MODEL", "MAX_RETRIES"):
    if _k in os.environ:
        continue
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:  # noqa: BLE001 — secrets file may not exist locally
        break

REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO_ROOT / "examples" / "benchmark.json"
FIXTURES_DIR = REPO_ROOT / "examples" / "vlm_fixtures"
IMAGES_DIR = REPO_ROOT / "examples" / "images" / "synthetic"
LEAN_PROJECT = REPO_ROOT / "lean_project"
GENERATED_LEAN = LEAN_PROJECT / "DiagramTheorems" / "Generated.lean"

DEFAULT_PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
DEFAULT_SECONDARY_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite-preview")
DEFAULT_MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "1"))


def _lean_runner() -> LeanRunner | None:
    if not LEAN_PROJECT.exists():
        return None
    return LeanRunner(project_root=LEAN_PROJECT, generated_path=GENERATED_LEAN)


def _status_badge(status: LeanStatus) -> str:
    if status is LeanStatus.OK:
        return ":green[✓ Type-checked]"
    if status is LeanStatus.TYPE_ERROR:
        return ":red[✗ Type error]"
    if status is LeanStatus.TIMEOUT:
        return ":orange[⏱ Build timed out]"
    return ":orange[Type-checker unavailable]"


def _run_end_to_end(image_path: Path, problem_text: str, theorem_name: str):
    """Run the full pipeline silently. Picks the strongest available backend."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if gemini_key and anthropic_key:
        primary = GeminiAdapter(api_key=gemini_key, model=DEFAULT_PRIMARY_MODEL)
        secondary = ClaudeAdapter(api_key=anthropic_key, model=DEFAULT_SECONDARY_MODEL)
        judge = JudgeLLM(api_key=gemini_key, model=DEFAULT_JUDGE_MODEL)
        reader = ConsensusVLMAdapter(primary, secondary, judge, max_attempts=3)

        primary_ext = GeminiExtractor(api_key=gemini_key, model=DEFAULT_PRIMARY_MODEL)
        secondary_ext = ClaudeExtractor(api_key=anthropic_key, model=DEFAULT_SECONDARY_MODEL)
        extractor = ConsensusExtractor(primary_ext, secondary_ext, judge, max_attempts=3)

        return run_consensus_pipeline(
            image_path=image_path,
            problem_text=problem_text,
            vlm=reader,
            extractor=extractor,
            judge=judge,
            lean_runner=_lean_runner(),
            theorem_name=theorem_name,
            max_retries=DEFAULT_MAX_RETRIES,
        )

    if gemini_key:
        reader = GeminiAdapter(api_key=gemini_key, model=DEFAULT_PRIMARY_MODEL)
        extractor = GeminiExtractor(api_key=gemini_key, model=DEFAULT_PRIMARY_MODEL)
        return run_consensus_pipeline(
            image_path=image_path,
            problem_text=problem_text,
            vlm=reader,
            extractor=extractor,
            judge=None,
            lean_runner=_lean_runner(),
            theorem_name=theorem_name,
            max_retries=0,
        )

    reader = FixtureAdapter(FIXTURES_DIR)
    reading = reader.read(image_path, fixture_path=None)
    return run_pipeline_from_reading(
        reading=reading,
        problem_text=problem_text,
        confirmed_assumptions=None,
        lean_runner=_lean_runner(),
        theorem_name=theorem_name,
    )


def _sanitize_theorem_name(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", stem).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"user_{cleaned}" if cleaned else "user_theorem"
    return cleaned


def interactive_page() -> None:
    st.header("Diagram → Lean theorem")

    source = st.radio(
        "Diagram source",
        options=["Use example", "Upload your own"],
        horizontal=True,
    )

    image_path: Path | None = None
    caption = ""
    default_problem_text = ""
    theorem_name = "user_theorem"

    if source == "Use example":
        examples = load_benchmark(BENCHMARK_PATH)
        choices = {e.id: e for e in examples}
        selection = st.selectbox("Example diagram:", list(choices.keys()))
        example = choices[selection]
        image_path = REPO_ROOT / example.image
        caption = example.id
        default_problem_text = example.problem_text
        theorem_name = example.lean_theorem_name
    else:
        uploaded = st.file_uploader(
            "Upload a diagram",
            type=["png", "jpg", "jpeg", "webp"],
            help="PNG, JPG, JPEG, or WEBP.",
        )
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp.close()
            image_path = Path(tmp.name)
            caption = uploaded.name
            theorem_name = _sanitize_theorem_name(Path(uploaded.name).stem)

    if image_path is not None:
        st.image(str(image_path), caption=caption, width=400)

    st.subheader("Problem text")
    problem_text = st.text_area(
        "Optional problem statement", value=default_problem_text, height=80
    )

    can_run = image_path is not None
    if not can_run:
        st.info("Upload a diagram above to continue.")

    if st.button("Generate Lean theorem", type="primary", disabled=not can_run):
        with st.spinner("Generating theorem… this may take 30–60 seconds"):
            try:
                result = _run_end_to_end(
                    image_path=image_path,
                    problem_text=problem_text,
                    theorem_name=theorem_name,
                )
            except (VLMError, FixtureNotFoundError) as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 — surface in UI, don't kill server
                st.error(f"{type(exc).__name__}: {exc}")
                st.exception(exc)
                st.stop()

        st.subheader("Diagram analysis")
        st.json(result.reading.to_dict())

        st.subheader("Extracted assumptions")
        if result.confirmed_assumptions:
            for a in result.confirmed_assumptions:
                st.markdown(f"- `{a}`")
        else:
            st.caption("No assumptions extracted.")

        st.subheader("Goal")
        if result.goal_candidates:
            with st.expander("Alternative goals"):
                st.write(result.goal_candidates)
        st.write(f"`{result.selected_goal or '(none)'}`")

        st.subheader("Lean output")
        st.markdown(_status_badge(result.lean_status))
        if result.lean_status in (LeanStatus.TYPE_ERROR, LeanStatus.TIMEOUT) and result.lean_stderr:
            st.code(result.lean_stderr, language="text")
        st.code(result.lean_source, language="lean")
        st.download_button(
            "Download .lean",
            data=result.lean_source,
            file_name=f"{theorem_name}.lean",
        )


def benchmark_page() -> None:
    st.header("Benchmark")

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
    st.set_page_config(page_title="Diagram-to-Lean Theorem Assistant", layout="wide")
    page = st.sidebar.selectbox("Page", ["Interactive", "Benchmark"])
    if page == "Interactive":
        interactive_page()
    else:
        benchmark_page()


if __name__ == "__main__":
    main()

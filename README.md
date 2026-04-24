# Diagram-to-Lean Theorem Assistant

Translate geometry diagrams and optional problem text into Lean 4 theorem candidates via a vision-language model. The system extracts named objects, relations, and visual marks from an input image, proposes candidate assumptions and ranked goal theorems, lets a human confirm, and emits a Lean 4 source file that imports a Mathlib-backed helper module and declares the theorem statement. Proof bodies are left as `sorry` — statement-level validity (the theorem declaration type-checks) is the deliverable.

## Design and plan

- Design spec: `docs/superpowers/specs/2026-04-23-multimodal-math-diagram-to-lean-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-24-multimodal-math-vlm-pipeline-plan.md`
- Proposal: `docs/project-proposal.md`
- Evaluation plan: `docs/evaluation-plan.md`

## Setup

### 1. System tools

Install Homebrew (skip if present), Python 3.11, git, and Lean 4 via elan:

```bash
[ -x "$(command -v brew)" ] || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11 git
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:stable
```

### 2. Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. API key

Create `.env` in the repo root (gitignored):

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

Optionally override the model:

```bash
echo "GEMINI_MODEL=gemini-2.5-flash" >> .env
```

### 4. Lean project + mathlib cache

```bash
cd lean_project
lake update
lake exe cache get       # ~15 min first time, ~1.5 GB
cd ..
```

### 5. Synthetic images (if not already generated)

```bash
python scripts/generate_synthetic.py
```

## Running

### Streamlit UI

```bash
streamlit run app.py
```

- **Interactive page:** pick a benchmark example → review VLM reading → confirm assumptions → pick goal → generate Lean → type-check.
- **Benchmark page:** runs all 5 benchmark examples with gold confirmations via `FixtureAdapter`; shows precision, recall, top-k, Lean validity.

### Demo script (no UI)

```bash
python -m diagram_theorem_assistant.demo
```

### Tests

```bash
pytest                                 # fast unit + integration (< 10s)
pytest -m lean                         # real `lake build` smoke test
pytest -m live                         # real Gemini API call
pytest -m "not lean and not live"      # strict CI-friendly filter
```

## Refreshing VLM fixtures

To capture live Gemini outputs for benchmark examples:

```bash
python scripts/record_vlm_fixtures.py                      # all
python scripts/record_vlm_fixtures.py --id isosceles_...   # one
```

Costs API calls. Overwrites existing fixtures.

## Optional: Geometry3K tier

To add ~10 real-diagram samples:

```bash
python scripts/fetch_geometry3k.py
```

Images land in `examples/images/geometry3k/` (gitignored). Manifest (hashes and IDs) is committed.

## What this project is — and isn't

**Is:** a research prototype that produces *correctly initialized* Lean theorem statements from diagrams. Statement-level Lean validity is the deliverable.

**Isn't:** an automated theorem prover. Generated theorems close with `sorry`. Bridging that gap is open research.

See the design spec for the full scope, risks, and comparison with Lean Blueprint.

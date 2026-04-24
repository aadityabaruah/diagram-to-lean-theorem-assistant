"""Refresh VLM fixtures by calling live Gemini.

Usage:
    python scripts/record_vlm_fixtures.py                      # refresh all
    python scripts/record_vlm_fixtures.py --id isosceles_...   # single example
    python scripts/record_vlm_fixtures.py --image path.png     # custom image

Writes JSON files into examples/vlm_fixtures/. Overwrites existing fixtures.
Costs live API calls — use sparingly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from diagram_theorem_assistant.schema import load_benchmark
from diagram_theorem_assistant.vlm.gemini import GeminiAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "examples" / "vlm_fixtures"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="example_id", help="Single benchmark example id to refresh")
    parser.add_argument("--image", dest="image_path", help="Arbitrary image path (no benchmark entry)")
    parser.add_argument("--out", dest="out_path", help="Output JSON path (required with --image)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set in .env", file=sys.stderr)
        return 1
    adapter = GeminiAdapter(api_key=api_key)

    if args.image_path:
        if not args.out_path:
            print("--out is required with --image", file=sys.stderr)
            return 1
        reading = adapter.read(Path(args.image_path))
        Path(args.out_path).write_text(json.dumps(reading.to_dict(), indent=2))
        print(f"Wrote {args.out_path}")
        return 0

    examples = load_benchmark(REPO_ROOT / "examples" / "benchmark.json")
    if args.example_id:
        examples = [e for e in examples if e.id == args.example_id]
        if not examples:
            print(f"No example with id {args.example_id}", file=sys.stderr)
            return 1

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for example in examples:
        image = REPO_ROOT / example.image
        if not image.exists():
            print(f"Skipping {example.id}: image {image} missing", file=sys.stderr)
            continue
        reading = adapter.read(image)
        out = REPO_ROOT / example.vlm_fixture if example.vlm_fixture else FIXTURES_DIR / f"{example.id}.json"
        out.write_text(json.dumps(reading.to_dict(), indent=2))
        print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

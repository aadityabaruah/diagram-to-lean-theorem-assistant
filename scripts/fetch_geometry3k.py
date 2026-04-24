"""Fetch a small curated set of Geometry3K diagrams for the realistic benchmark tier.

Uses the HuggingFace `hiyouga/geometry3k` dataset — a curated repackaging of
Geometry3K (Lu et al., ACL 2021) that bundles each problem with its diagram
image, problem text, and gold answer. The full set is ~3,002 problems across
train/validation/test splits; this script downloads only `--count` samples
(default 10) from the chosen split to keep the footprint small.

Usage:
    python scripts/fetch_geometry3k.py                 # 10 from test split
    python scripts/fetch_geometry3k.py --count 25
    python scripts/fetch_geometry3k.py --split train --count 5

Writes images to examples/images/geometry3k/ (gitignored). Writes a manifest
of ids, paths, SHA256 hashes, problem text, and gold answers to
examples/geometry3k_manifest.json (committed) so the selected sample set is
reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "examples" / "images" / "geometry3k"
MANIFEST = REPO_ROOT / "examples" / "geometry3k_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--dataset", default="hiyouga/geometry3k")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "The `datasets` library is required. Install with: pip install -e \".[dev]\"",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.dataset} ({args.split} split)...")
    ds = load_dataset(args.dataset, split=args.split)
    count = min(args.count, len(ds))
    print(f"Selecting first {count} of {len(ds)} examples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for index in range(count):
        example = ds[index]
        try:
            images = example["images"]
            image = images[0] if isinstance(images, list) else images
            problem = example.get("problem", "")
            answer = example.get("answer", "")
        except (KeyError, IndexError, TypeError) as exc:
            print(
                f"  skipping index {index}: unexpected schema ({type(exc).__name__}: {exc}). "
                f"Available keys: {list(example.keys()) if hasattr(example, 'keys') else 'unknown'}",
                file=sys.stderr,
            )
            continue

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        digest = hashlib.sha256(png_bytes).hexdigest()

        example_id = f"{args.split}_{index:04d}"
        out_path = OUTPUT_DIR / f"{example_id}.png"
        out_path.write_bytes(png_bytes)

        manifest.append(
            {
                "id": example_id,
                "path": str(out_path.relative_to(REPO_ROOT)),
                "sha256": digest,
                "problem": problem,
                "answer": answer,
            }
        )
        print(f"  saved {out_path.relative_to(REPO_ROOT)} ({len(png_bytes)} bytes)")

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest to {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

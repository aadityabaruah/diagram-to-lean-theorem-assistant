"""Fetch a small curated set of Geometry3K diagrams for the realistic benchmark tier.

Geometry3K is published by Lu et al. (ACL 2021) at
https://github.com/lupantech/InterGPS. The full dataset is ~3,002 problems;
we download only the ~10 samples listed in CURATED_IDS to stay within
reasonable disk and license footprints.

Usage:
    python scripts/fetch_geometry3k.py

Writes images to examples/images/geometry3k/ (gitignored). If the upstream
layout changes, update BASE_URL and the list of sample IDs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "examples" / "images" / "geometry3k"
MANIFEST = REPO_ROOT / "examples" / "geometry3k_manifest.json"

BASE_URL = "https://raw.githubusercontent.com/lupantech/InterGPS/main/data/geometry3k/test"

CURATED_IDS = [
    "2401", "2402", "2403", "2404", "2405",
    "2406", "2407", "2408", "2409", "2410",
]


def _fetch(example_id: str) -> tuple[bytes, str]:
    url = f"{BASE_URL}/{example_id}/img_diagram.png"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content, hashlib.sha256(response.content).hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for example_id in CURATED_IDS:
        try:
            content, digest = _fetch(example_id)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {example_id}: {exc}", file=sys.stderr)
            continue
        out = OUTPUT_DIR / f"{example_id}.png"
        out.write_bytes(content)
        manifest.append({"id": example_id, "path": str(out.relative_to(REPO_ROOT)), "sha256": digest})
        print(f"Saved {out.relative_to(REPO_ROOT)} ({len(content)} bytes)")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

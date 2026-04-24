"""Render deterministic matplotlib PNGs for benchmark examples.

Run: python scripts/generate_synthetic.py
Outputs: examples/images/synthetic/*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples" / "images" / "synthetic"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def isosceles_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.5, 0.9]), np.array([0.1, 0.2]), np.array([0.9, 0.2])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    # Equal-length tick marks on AB and AC
    for p1, p2 in [(a, b), (a, c)]:
        mid = (p1 + p2) / 2
        perp = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
        perp /= np.linalg.norm(perp) / 0.02
        ax.plot([mid[0] - perp[0], mid[0] + perp[0]], [mid[1] - perp[1], mid[1] + perp[1]], "k-", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "isosceles_triangle")


def parallel_lines() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0.1, 0.9], [0.7, 0.7], "k-", linewidth=2)
    ax.plot([0.1, 0.9], [0.3, 0.3], "k-", linewidth=2)
    ax.plot([0.2, 0.8], [0.1, 0.9], "k-", linewidth=2)
    ax.annotate("l", (0.9, 0.7), fontsize=14, xytext=(8, 0), textcoords="offset points")
    ax.annotate("m", (0.9, 0.3), fontsize=14, xytext=(8, 0), textcoords="offset points")
    ax.annotate("t", (0.8, 0.9), fontsize=14, xytext=(0, 8), textcoords="offset points")
    ax.annotate("1", (0.55, 0.73), fontsize=12)
    ax.annotate("2", (0.4, 0.27), fontsize=12)
    # Arrow marks denoting parallel lines
    for y in [0.7, 0.3]:
        ax.annotate("", xy=(0.55, y + 0.005), xytext=(0.45, y + 0.005),
                    arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _save(fig, "parallel_lines")


def ambiguous_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.5, 0.85]), np.array([0.15, 0.15]), np.array([0.8, 0.2])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "ambiguous_triangle")


def right_triangle() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, c = np.array([0.15, 0.85]), np.array([0.85, 0.15]), np.array([0.15, 0.15])
    triangle = plt.Polygon([a, b, c], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)
    for point, label in [(a, "A"), (b, "B"), (c, "C")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-12, 8), fontsize=14)
    # Right-angle box at C
    box = plt.Rectangle((0.15, 0.15), 0.05, 0.05, fill=False, edgecolor="black")
    ax.add_patch(box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "right_triangle")


def perpendicular_bisector() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    a, b, p = np.array([0.2, 0.3]), np.array([0.8, 0.3]), np.array([0.5, 0.85])
    ax.plot([a[0], b[0]], [a[1], b[1]], "k-", linewidth=2)
    ax.plot([0.5, 0.5], [0.1, 0.95], "k--", linewidth=1.5)
    ax.plot([p[0], a[0]], [p[1], a[1]], "k-", linewidth=2)
    ax.plot([p[0], b[0]], [p[1], b[1]], "k-", linewidth=2)
    for point, label in [(a, "A"), (b, "B"), (p, "P")]:
        ax.scatter(*point, color="black", zorder=5)
        ax.annotate(label, point, textcoords="offset points", xytext=(-10, 10), fontsize=14)
    # Right-angle box at midpoint
    box = plt.Rectangle((0.5, 0.3), 0.04, 0.04, fill=False, edgecolor="black")
    ax.add_patch(box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "perpendicular_bisector")


def main() -> None:
    isosceles_triangle()
    parallel_lines()
    ambiguous_triangle()
    right_triangle()
    perpendicular_bisector()
    print(f"Wrote 5 synthetic PNGs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

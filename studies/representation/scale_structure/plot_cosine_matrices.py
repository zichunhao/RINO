#!/usr/bin/env python3
"""Plot cross-scale cosine-similarity matrices from ``cosine_similarity_stats.json``.

Draws one annotated heatmap per ``--panel`` on a shared colour scale. Each
panel reads the mean per-jet cosine of every scale pair from a stats file
written by ``compute_scale_cosines.py`` (or by
``dino/diagnostics/scale_invariance_analysis.py``); no values are embedded.

A panel is a list of ``key=value`` tokens:
    stats=PATH      stats JSON (required)
    space=NAME      backbone (default) | head_output | head_bottleneck
    title=TEXT      first title line
    subtitle=TEXT   second title line
    teacher=LIST    comma-separated subjet counts of the teacher scales, e.g.
                    6,8,16; tick labels are then coloured by role
                    (teacher red, student blue). Omit for no role colouring.

Usage (App. "Scale Structure", 2x2 figure):
    python studies/representation/scale_structure/plot_cosine_matrices.py \
        --panel stats=<mpmv2>/cosine_similarity_stats.json \
                title="MPMv2 backbone" subtitle="Scale-ignorant" \
        --panel stats=<jetclr-scale>/cosine_similarity_stats.json \
                title="JetCLR-scale backbone" subtitle="Scale-invariant" \
        --panel stats=<rino>/cosine_similarity_stats.json teacher=6,8,16 \
                title="RINO backbone" subtitle="Scale-aware" \
        --panel stats=<rino>/cosine_similarity_stats.json space=head_output \
                teacher=6,8,16 title="RINO backbone + projection head" \
                subtitle="Scale-invariant" \
        --ncols 2 --colorbar bottom --output scale_structure.pdf
"""

import argparse
import json
import re
import shlex
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEACHER_COLOR = "#cb181d"
STUDENT_COLOR = "#2171b5"
DEFAULT_ORDER = ["subjet2", "subjet3", "subjet4", "subjet6", "subjet8", "subjet16", "ALL"]
SPACE_KEYS = {
    "backbone": "pairwise_cosine",
    "head_output": "head_output_pairwise_cosine",
    "head_bottleneck": "head_bottleneck_pairwise_cosine",
}
PANEL_KEYS = {"stats", "space", "title", "subtitle", "teacher"}


def parse_panel(tokens: list[str]) -> dict[str, str]:
    """Parse ``key=value`` tokens of one ``--panel``."""
    panel: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or key not in PANEL_KEYS:
            raise ValueError(
                f"Bad panel token {token!r}; expected key=value with key in {sorted(PANEL_KEYS)}"
            )
        panel[key] = value
    if "stats" not in panel:
        raise ValueError(f"Panel {' '.join(map(shlex.quote, tokens))} has no stats=PATH")
    panel.setdefault("space", "backbone")
    if panel["space"] not in SPACE_KEYS:
        raise ValueError(f"space must be one of {sorted(SPACE_KEYS)}, got {panel['space']!r}")
    return panel


def load_matrix(stats_path: str, space: str, order: list[str]) -> np.ndarray:
    """Symmetric matrix of mean cosines in ``order`` (diagonal = 1)."""
    with open(stats_path) as f:
        stats = json.load(f)
    key = SPACE_KEYS[space]
    if key not in stats:
        raise KeyError(f"{stats_path} has no '{key}' (was it computed with --head-space?)")
    pairs = stats[key]
    matrix = np.eye(len(order))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            entry = pairs.get(f"{a}_vs_{b}") or pairs.get(f"{b}_vs_{a}")
            if entry is None:
                raise KeyError(f"{stats_path}: no '{a}' vs '{b}' entry in '{key}'")
            matrix[i, j] = entry["mean"]
    return matrix


def scale_label(scale: str) -> str:
    """Tick label: ``subjetN`` -> N=N, ``ALL`` -> Uncl."""
    if scale == "ALL":
        return "Uncl."
    match = re.fullmatch(r"subjet(\d+)", scale)
    return rf"$N\!=\!{match.group(1)}$" if match else scale


def role_colors(order: list[str], teacher: str | None) -> list[str] | None:
    """Tick colours by teacher/student role, or ``None`` without a teacher list."""
    if not teacher:
        return None
    teacher_scales = {f"subjet{n.strip()}" for n in teacher.split(",") if n.strip()}
    return [TEACHER_COLOR if s in teacher_scales else STUDENT_COLOR for s in order]


def draw_panel(ax, matrix: np.ndarray, order: list[str], panel: dict, cmap: str, fontsize: float):
    """One annotated heatmap; returns the image for the shared colour bar."""
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    n = len(order)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=fontsize,
                color="white" if val < 0.4 or val > 0.9 else "black",
                fontweight="bold" if i == j else "normal",
            )
    labels = [scale_label(s) for s in order]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    title = "\n".join(t for t in (panel.get("title"), panel.get("subtitle")) if t)
    if title:
        ax.set_title(title)
    colors = role_colors(order, panel.get("teacher"))
    if colors:
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, color in zip(ticks, colors):
                tick.set_color(color)
                tick.set_fontweight("bold")
    return im


def plot(
    panels: list[dict],
    order: list[str],
    output: Path,
    ncols: int,
    colorbar: str,
    panel_size: float,
    cmap: str,
    fontsize: float,
) -> None:
    """Lay out the panels on a grid with one colour bar and save the figure."""
    nrows = int(np.ceil(len(panels) / ncols))
    plt.rcParams.update({"font.family": "serif", "font.size": fontsize + 2, "figure.dpi": 300})
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False
    )
    im = None
    for ax, panel in zip(axes.flat, panels):
        matrix = load_matrix(panel["stats"], panel["space"], order)
        im = draw_panel(ax, matrix, order, panel, cmap, fontsize)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    if colorbar == "bottom":
        fig.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.28)
        cax = fig.add_axes([0.25, 0.055, 0.55, 0.018])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    else:
        fig.subplots_adjust(right=0.88, wspace=0.30, hspace=0.38)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean Cosine Similarity")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cross-scale cosine-similarity matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--panel",
        nargs="+",
        action="append",
        required=True,
        metavar="KEY=VALUE",
        help="One panel: stats=PATH [space=...] [title=...] [subtitle=...] [teacher=6,8,16].",
    )
    parser.add_argument("--order", nargs="+", default=DEFAULT_ORDER, help="Scale order on both axes.")
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--colorbar", default="bottom", choices=("bottom", "right"))
    parser.add_argument("--panel-size", type=float, default=5.25, help="Inches per panel.")
    parser.add_argument("--cmap", default="RdYlGn")
    parser.add_argument("--fontsize", type=float, default=10, help="Cell annotation size.")
    parser.add_argument("--output", "-o", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    panels = [parse_panel(tokens) for tokens in args.panel]
    plot(
        panels,
        args.order,
        Path(args.output),
        ncols=args.ncols,
        colorbar=args.colorbar,
        panel_size=args.panel_size,
        cmap=args.cmap,
        fontsize=args.fontsize,
    )


if __name__ == "__main__":
    main()

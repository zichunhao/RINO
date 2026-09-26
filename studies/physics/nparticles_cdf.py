"""Constituent-multiplicity CDF and dataset statistics (``fig:nparticles-cdf``).

Reads the per-jet counts written by ``nparticles_count.py``, prints the
statistics quoted in "Dataset Details" (median, mean +/- std, and the
fraction of jets with at most n constituents) and plots the empirical CDF
with the median marked and the kT view range shaded.

Example::

    python studies/physics/nparticles_cdf.py \\
        --counts PROJECT_ROOT/experiments/studies/physics/nparticles_qcd.h5 \\
        --output PROJECT_ROOT/experiments/studies/physics/nparticles_cdf.pdf
"""

from __future__ import annotations

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import resolve_path, write_json  # noqa: E402

LINE_COLOR = "#2171b5"
MEDIAN_COLOR = "#cb181d"
VIEW_RANGE_COLOR = "#ff7f0e"


def multiplicity_stats(counts: np.ndarray, thresholds: list[int]) -> dict:
    """Median, mean, std and P(n <= t) of the constituent multiplicity."""
    return {
        "n_jets": int(len(counts)),
        "median": float(np.median(counts)),
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "fraction_at_most": {str(t): float(np.mean(counts <= t)) for t in thresholds},
    }


def plot_cdf(
    counts: np.ndarray, view_range: tuple[int, int], x_max: int, output
) -> None:
    """Step CDF of the multiplicity, median marker and shaded kT view range."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "text.usetex": False,
            "figure.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Exact CDF from the unique values (no subsampling).
    values, freq = np.unique(counts, return_counts=True)
    ax.step(
        values,
        np.cumsum(freq) / len(counts),
        where="mid",
        color=LINE_COLOR,
        linewidth=1.8,
    )

    median = np.median(counts)
    ax.axvline(x=median, color=MEDIAN_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.annotate(
        f"Median = {median:.0f}",
        xy=(median, 0.5),
        xytext=(median + 15, 0.35),
        fontsize=10,
        color=MEDIAN_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=MEDIAN_COLOR, lw=1.0),
        ha="left",
    )
    ax.axvspan(*view_range, alpha=0.08, color=VIEW_RANGE_COLOR)

    ax.set_xlabel("Number of constituents per jet")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--counts",
        default="PROJECT_ROOT/experiments/studies/physics/nparticles_qcd.h5",
        help="HDF5 file from nparticles_count.py.",
    )
    parser.add_argument("--dataset", default="combined", help="Dataset (split) to use.")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 16, 32, 64],
        help="Report the fraction of jets with at most this many constituents.",
    )
    parser.add_argument(
        "--view-range",
        type=int,
        nargs=2,
        default=[2, 16],
        metavar=("N_MIN", "N_MAX"),
        help="kT view range (numbers of subjets) to shade.",
    )
    parser.add_argument("--x-max", type=int, default=100, help="x-axis limit.")
    parser.add_argument(
        "--output",
        default="PROJECT_ROOT/experiments/studies/physics/nparticles_cdf.pdf",
        help="Figure path.",
    )
    parser.add_argument(
        "--stats-json", default=None, help="Optional path for the statistics JSON."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(resolve_path(args.counts), "r") as handle:
        counts = handle[args.dataset][:]

    stats = multiplicity_stats(counts, args.thresholds)
    print(f"{stats['n_jets']:,} jets ({args.dataset})")
    print(
        f"Median {stats['median']:.0f}, mean {stats['mean']:.1f} +/- {stats['std']:.1f}"
    )
    for t, frac in stats["fraction_at_most"].items():
        print(f"  n <= {int(t):3d}: {frac:.5f} ({100 * frac:.2f}%)")

    plot_cdf(counts, tuple(args.view_range), args.x_max, resolve_path(args.output))
    if args.stats_json:
        write_json(stats, args.stats_json)


if __name__ == "__main__":
    main()

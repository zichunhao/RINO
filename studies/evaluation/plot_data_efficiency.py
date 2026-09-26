"""Plot OOD accuracy versus finetuning label fraction (data-efficiency curves).

Reads the per-seed CSV written by ``harvest_finetune.py`` for label-efficiency
runs, whose manifest tags each experiment with ``method`` and ``fraction`` (the
fraction of JetNet training labels, e.g. 0.001, 0.01, 0.1, 0.5, 1.0; the 100%
points are the Table 1 runs). Draws the mean over seeds with a ±1σ band per
method on a log x-axis, and optionally a dotted reference line at one method's
largest-fraction value and an annotation comparing two points.

Usage:
    # Main-text figure (six methods, reference line and annotation)
    python studies/evaluation/plot_data_efficiency.py \
        --csv experiments/studies/label_eff_per_seed.csv --where head=mlp \
        --methods RINO Supervised MPMv1 MPMv2 JetCLR JetCLR-scale \
        --reference Supervised --annotate RINO@0.01 Supervised@1.0 \
        --output experiments/studies/data_efficiency.pdf

    # Appendix figure (all methods)
    python studies/evaluation/plot_data_efficiency.py \
        --csv experiments/studies/label_eff_per_seed.csv --where head=mlp \
        --reference Supervised --figsize 6.0 4.5 \
        --output experiments/studies/data_efficiency_full.pdf
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import as_float, filter_rows, mean_std, parse_where, read_csvs, write_csv  # noqa: E402

# Line styles of the paper figure, keyed by lower-case method name.
METHOD_STYLES = {
    "rino": {"color": "#2171b5", "marker": "o", "ls": "-", "lw": 2.5, "ms": 7},
    "supervised": {"color": "#cb181d", "marker": "s", "ls": "--", "lw": 2.0, "ms": 6},
    "mpmv1": {"color": "#2ca02c", "marker": "^", "ls": "-.", "lw": 1.6, "ms": 6},
    "mpmv2": {"color": "#17becf", "marker": "v", "ls": "-.", "lw": 1.6, "ms": 6},
    "omnijet-alpha": {"color": "#8c564b", "marker": "*", "ls": ":", "lw": 1.4, "ms": 8},
    "jetclr": {"color": "#9467bd", "marker": "h", "ls": ":", "lw": 1.4, "ms": 7},
    "jetclr-scale": {"color": "#ff7f0e", "marker": "D", "ls": "--", "lw": 1.6, "ms": 5},
}
METHOD_STYLES["omnijet"] = METHOD_STYLES["omnijet-alpha"]
DISPLAY_NAMES = {"rino": "RINO (ours)", "omnijet-alpha": r"OmniJet-$\alpha$", "omnijet": r"OmniJet-$\alpha$"}
SYMBOLS_TEX = {"gtrsim": r"\gtrsim", "approx": r"\approx", "geq": r"\geq", "gt": ">"}

RC_PARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.dpi": 300,
}


def _key(method: str) -> str:
    return method.strip().lower().replace(" ", "-").replace("_", "-")


def curves(rows: list[dict], group: str, x: str, metric: str, ddof: int) -> dict[str, dict]:
    """``{method: {"x": array, "mean": array, "std": array, "n": array}}`` sorted by x."""
    by_point: dict[tuple[str, float], list[float]] = {}
    for r in rows:
        by_point.setdefault((r[group], as_float(r[x])), []).append(as_float(r[metric]))
    out: dict[str, dict] = {}
    for (method, xv), values in by_point.items():
        m, s, n = mean_std(values, ddof)
        c = out.setdefault(method, {"x": [], "mean": [], "std": [], "n": []})
        c["x"].append(xv)
        c["mean"].append(m)
        c["std"].append(0.0 if math.isnan(s) else s)
        c["n"].append(n)
    for c in out.values():
        order = np.argsort(c["x"])
        for k in c:
            c[k] = np.asarray(c[k])[order]
    return out


def point(data: dict[str, dict], spec: str) -> tuple[str, float, float]:
    """Resolve ``METHOD@FRACTION`` to ``(method, fraction, mean)``."""
    method, frac = spec.rsplit("@", 1)
    c = data[method]
    idx = np.flatnonzero(np.isclose(c["x"], float(frac)))
    if len(idx) == 0:
        raise SystemExit(f"{method} has no point at fraction {frac} (have {c['x'].tolist()}).")
    return method, float(frac), float(c["mean"][idx[0]])


def make_figure(data, methods, output, reference=None, annotate=None, symbol="gtrsim", figsize=(5.5, 4.0), ylim=None):
    """Draw the curves with ±1σ bands and save the figure."""
    plt.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=figsize)
    all_x = sorted({float(v) for m in methods for v in data[m]["x"]})

    zorder = 10
    for method in methods:
        c = data[method]
        style = METHOD_STYLES.get(_key(method), {"marker": "o", "ls": "-", "lw": 1.6, "ms": 6})
        x = 100.0 * c["x"]
        label = DISPLAY_NAMES.get(_key(method), method)
        (line,) = ax.plot(
            x, c["mean"], marker=style["marker"], linestyle=style["ls"], color=style.get("color"),
            linewidth=style["lw"], markersize=style["ms"], label=label, zorder=zorder,
        )
        ax.fill_between(x, c["mean"] - c["std"], c["mean"] + c["std"], alpha=0.15, color=line.get_color(), zorder=zorder - 1)
        zorder -= 1

    if reference:
        ref = data[reference]
        color = METHOD_STYLES.get(_key(reference), {}).get("color", "gray")
        ax.axhline(y=ref["mean"][-1], color=color, linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)

    if annotate:
        (m_a, x_a, y_a), (m_b, x_b, _) = point(data, annotate[0]), point(data, annotate[1])
        color = METHOD_STYLES.get(_key(m_a), {}).get("color", "black")
        text = rf"{100 * x_a:g}% {m_a} ${SYMBOLS_TEX[symbol]}$ {100 * x_b:g}% {m_b}"
        ax.annotate(
            text, xy=(100 * x_a, y_a), xytext=(2 * 100 * x_a, y_a + 0.08), fontsize=10, color=color, ha="center",
            fontweight="bold", arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Data fraction (%)")
    ax.set_ylabel("Top vs QCD accuracy (OOD)")
    ticks = [100.0 * v for v in all_x]
    ax.set_xlim(ticks[0] * 0.6, ticks[-1] * 1.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, nargs="+", required=True, help="Per-seed CSV(s) from harvest_finetune.py.")
    parser.add_argument("--where", type=str, action="append", default=None, help="Filter key=value[,value]; repeatable.")
    parser.add_argument("--subset", type=str, default="Tbqq_vs_QCD", help="Label subset to plot.")
    parser.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"])
    parser.add_argument("--group", type=str, default="method", help="Column identifying a curve.")
    parser.add_argument("--x", type=str, default="fraction", help="Column holding the label fraction (0-1].")
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Curves to draw, in legend order (default: all).")
    parser.add_argument("--reference", type=str, default=None, help="Method whose largest-fraction mean is drawn as a dotted line.")
    parser.add_argument("--annotate", type=str, nargs=2, default=None, metavar=("A@FRAC", "B@FRAC"), help="Annotate point A relative to point B.")
    parser.add_argument("--symbol", choices=sorted(SYMBOLS_TEX), default="gtrsim", help="Comparison symbol of the annotation.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(5.5, 4.0))
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.45, 0.92))
    parser.add_argument("--ddof", type=int, default=1, help="Delta degrees of freedom of the std over seeds.")
    parser.add_argument("--summary-out", type=str, default=None, help="Also write the plotted mean/std per point as CSV.")
    parser.add_argument("--output", type=str, required=True, help="Figure path (.pdf or .png).")
    args = parser.parse_args()

    where = parse_where(args.where)
    where["subset"] = [args.subset]
    rows = filter_rows(read_csvs(args.csv), where)
    if not rows:
        raise SystemExit("No rows left after filtering.")
    data = curves(rows, args.group, args.x, args.metric, args.ddof)
    methods = args.methods or list(data)
    unknown = [m for m in methods if m not in data]
    if unknown:
        raise SystemExit(f"Methods {unknown} not in the data (have {list(data)}).")

    for m in methods:
        pts = ", ".join(f"{100 * x:g}%: {mu:.3f}±{sd:.3f} (n={n})" for x, mu, sd, n in zip(*(data[m][k] for k in ("x", "mean", "std", "n"))))
        print(f"  {m:<14s} {pts}")
    if args.summary_out:
        write_csv(args.summary_out, [
            {args.group: m, args.x: x, "metric": args.metric, "mean": mu, "std": sd, "n": n}
            for m in methods for x, mu, sd, n in zip(*(data[m][k] for k in ("x", "mean", "std", "n")))
        ])
    make_figure(data, methods, args.output, args.reference, args.annotate, args.symbol, tuple(args.figsize), tuple(args.ylim))


if __name__ == "__main__":
    main()

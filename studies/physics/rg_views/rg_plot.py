#!/usr/bin/env python3
"""Figures for the RG study from rg_metrics.py JSON output.

  --classify   one or more `classify` results: AUC per view (x = N, top axis = median kT
               merge scale q_N of sample A). Infrared deformations should approach
               AUC = 0.5 for coarse views; relevant information (top vs QCD) should not.
  --paired     a `paired` result: relative merge-scale shift |dq|/q vs q_N (log-log,
               with a 1/q reference) and EMD/pT per view for each algorithm.
  --scaling    a `scaling` result: <N_excl>(q) and its local exponent.

Example:
  python rg_plot.py --classify results/auc_*.json --paired results/paired_qcd_nominal.json \\
      --scaling results/scaling_reco.json --outdir plots
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ALGO_STYLE = {"kt": ("C0", "-", r"$k_\mathrm{T}$"), "ca": ("C1", "--", "C/A"), "antikt": ("C2", ":", r"anti-$k_\mathrm{T}$")}


def _load(path):
    with open(path) as fp:
        return json.load(fp)


def _view_axis(ax, nums, q_table=None):
    xs = list(range(len(nums) + 1))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in nums] + ["uncl."])
    ax.set_xlabel("view (number of subjets $N$)")
    if q_table:
        top = ax.secondary_xaxis("top")
        top.set_xticks(xs[:-1])
        top.set_xticklabels(
            [f"{q_table[f'N{n}']['median']:.2g}" if q_table[f"N{n}"]["median"] else "-" for n in nums]
        )
        top.set_xlabel(r"median $k_\mathrm{T}$ merge scale $q_N$ [GeV]")
    return xs


def _save(fig, outdir, stem, formats):
    for ext in formats:
        out = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(out, dpi=150)
        print("wrote", out)
    plt.close(fig)


def plot_classify(paths, outdir, tag="", formats=("pdf",), q_axis=True, labels=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    data = [_load(p) for p in paths]
    nums = sorted({int(v[1:]) for d in data for v in d["auc"] if v.startswith("N")})
    q_table = next((d["q_kt"] for d in data if all(f"N{n}" in d.get("q_kt", {}) for n in nums)), None)
    if not q_axis:
        q_table = None
    _view_axis(ax, nums, q_table)
    # colour = comparison, line style = clustering algorithm (a trailing "(ca)" etc. in the
    # name is dropped so that the same comparison keeps its colour across algorithms)
    algos = {d.get("algorithm", "kt") for d in data}
    bases = []
    for i, d in enumerate(data):
        algo = d.get("algorithm", "kt")
        base = re.sub(r"\s*\((kt|ca|antikt)\)$", "", d["name"])
        if labels and i < len(labels):
            base = labels[i]  # an explicit legend label, used as given
        if base not in bases:
            bases.append(base)
        c = f"C{bases.index(base)}"
        ls = ALGO_STYLE.get(algo, ("", "-", algo))[1]
        label = base if (len(algos) == 1 or labels) else f"{base}, {ALGO_STYLE.get(algo, ('', '', algo))[2]}"
        views = list(d["auc"].keys())
        x = [nums.index(int(v[1:])) if v.startswith("N") else len(nums) for v in views]
        y = np.array([d["auc"][v]["mean"] for v in views])
        e = np.array([d["auc"][v]["std"] for v in views])
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=2, label=label, color=c, ls=ls)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("test ROC AUC (A vs B, one view)")
    ax.set_ylim(0.45, 1.01)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, outdir, f"rg_classify{'_' + tag if tag else ''}", formats)


def plot_paired(path, outdir, formats=("pdf",)):
    d = _load(path)
    nums = d["nums"]
    for key, e in d["pairs"].items():
        la, lb = key.split("->")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        # left: relative merge-scale shift vs q_N (kT only)
        q = np.array([e["dq_over_q"][f"N{n}"].get("q_median") or np.nan for n in nums], dtype=float)
        r = np.array([e["dq_over_q"][f"N{n}"].get("median", np.nan) for n in nums], dtype=float)
        lo = np.array([e["dq_over_q"][f"N{n}"].get("p16", np.nan) for n in nums], dtype=float)
        hi = np.array([e["dq_over_q"][f"N{n}"].get("p84", np.nan) for n in nums], dtype=float)
        a1.errorbar(q, r, yerr=[r - lo, hi - r], marker="o", capsize=2, color="C0", label="median, 16-84%")
        for n, qq, rr in zip(nums, q, r):
            a1.annotate(f"N={n}", (qq, rr), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ok = np.isfinite(q) & np.isfinite(r)
        if ok.any():
            q0, r0 = q[ok][-1], r[ok][-1]
            qq = np.geomspace(np.nanmin(q), np.nanmax(q), 20)
            a1.plot(qq, r0 * q0 / qq, color="gray", ls="--", lw=0.8, label=r"$\propto 1/q$")
        a1.set_xscale("log")
        a1.set_yscale("log")
        a1.set_xlabel(rf"median $q_N$ at {lb} level [GeV]")
        a1.set_ylabel(rf"$|q_N^\mathrm{{{la}}} - q_N^\mathrm{{{lb}}}| / q_N^\mathrm{{{lb}}}$")
        a1.legend(fontsize=8, frameon=False)
        a1.set_title(f"{la} -> {lb}: merge-scale shift (kT)", fontsize=9)
        # right: EMD / pT per view and algorithm
        xs = _view_axis(a2, nums, e["q_kt"].get(lb))
        for algo, (c, ls, lab) in ALGO_STYLE.items():
            if algo not in e["emd_over_pt"]:
                continue
            y = [e["emd_over_pt"][algo][f"N{n}"]["median"] for n in nums] + [e["emd_over_pt"]["uncl"]["median"]]
            a2.plot(xs, y, marker="o", color=c, ls=ls, label=lab)
        a2.set_yscale("log")
        a2.set_ylabel(r"median EMD / $p_\mathrm{T}^\mathrm{jet}$")
        a2.legend(fontsize=8, frameon=False)
        a2.set_title(f"{la} -> {lb}: EMD between the two views", fontsize=9)
        fig.tight_layout()
        _save(fig, outdir, f"rg_paired_{la}_{lb}", formats)


def plot_scaling_ratio(paths, outdir, labels=None, tag="", formats=("pdf",), styles=None):
    """Only <N_excl>(q) of each result divided by the first one, with the statistical error band."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ref = _load(paths[0])
    n0, s0 = np.array(ref["mean_nexcl"]), np.array(ref["sem_nexcl"])
    q = np.array(ref["qgrid"])
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    for i, path in enumerate(paths[1:], start=1):
        d = _load(path)
        n, s = np.array(d["mean_nexcl"]), np.array(d["sem_nexcl"])
        r = n / n0
        e = r * np.hypot(s / n, s0 / n0)
        fmt = styles[i] if styles and i < len(styles) else f"C{i}-"
        lab = labels[i] if labels and i < len(labels) else os.path.basename(path)
        line, = ax.plot(q, r, fmt, lw=1.8, label=lab)
        ax.fill_between(q, r - e, r + e, color=line.get_color(), alpha=0.25, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"resolution $q = \sqrt{d_\mathrm{cut}}$ [GeV]")
    ax.set_ylabel(rf"$\langle N_\mathrm{{excl}}\rangle$ / {labels[0] if labels else 'reference'}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, outdir, f"rg_scaling_ratio{'_' + tag if tag else ''}", formats)


def plot_reach(ref_path, coupling_path, cutoff_path, hadron_paths, outdir, tag="", formats=("pdf",),
               cutoff_scale=2.0, hadron_scale=1.0, matching_scale=45.0, qmin=0.4, qmax=150.0):
    """How far up in scale each kind of change reaches: <N_excl>(q) relative to the reference.

    The q axis is inverted (coarse on the left, fine on the right, like the classify figures).
    The scale of each small-scale change is marked, the region above the ME/PS matching scale
    is shaded, and the top axis shows the median scale of each fixed-N view of the reference.
    The hadronization variants are drawn as one band (their envelope)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ref = _load(ref_path)
    q = np.array(ref["qgrid"])
    n0, s0 = np.array(ref["mean_nexcl"]), np.array(ref["sem_nexcl"])

    def ratio(path):
        d = _load(path)
        n, s = np.array(d["mean_nexcl"]), np.array(d["sem_nexcl"])
        r = n / n0
        return r, r * np.hypot(s / n, s0 / n0)

    ax.axvspan(matching_scale, qmax, color="0.90", lw=0)
    ax.text(np.sqrt(matching_scale * qmax), 0.906, "matrix element\n(not varied)", ha="center", va="bottom",
            fontsize=7.5, color="0.35")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    had = np.array([ratio(p)[0] for p in hadron_paths])
    for path, color, label in ((coupling_path, "C1", r"weaker coupling ($\mu_R \times 2$)"),
                               (cutoff_path, "C0", rf"shower cutoff 0.5 $\to$ {cutoff_scale:g} GeV")):
        r, e = ratio(path)
        ax.plot(q, r, color=color, lw=2.0, label=label)
        ax.fill_between(q, r - e, r + e, color=color, alpha=0.25, lw=0)
    ax.fill_between(q, had.min(0), had.max(0), color="0.55", alpha=0.45, lw=0,
                    label=f"hadronization ({len(hadron_paths)} settings)")
    # where each small-scale change lives
    ax.axvline(cutoff_scale, color="C0", lw=1.0, ls=":")
    ax.text(cutoff_scale * 1.08, 1.057, f"new cutoff\n{cutoff_scale:g} GeV", ha="right", va="top", fontsize=7.5, color="C0")
    ax.axvline(hadron_scale, color="0.35", lw=1.0, ls=":")
    ax.text(hadron_scale * 0.93, 1.057, f"hadronization\n~{hadron_scale:g} GeV", ha="left", va="top", fontsize=7.5,
            color="0.35")
    ax.set_xscale("log")
    ax.set_xlim(qmax, qmin)  # inverted: coarse -> fine
    ax.set_ylim(0.90, 1.06)
    ax.set_xlabel(r"resolution $q$ [GeV]   (coarse $\rightarrow$ fine)")
    ax.set_ylabel("mean number of subjets / nominal")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    top = ax.secondary_xaxis("top")
    ticks = [(v["median"], k[1:]) for k, v in ref.get("q_kt", {}).items() if v.get("median") and qmin < v["median"] < qmax]
    top.set_xticks([t[0] for t in ticks])
    top.set_xticklabels([t[1] for t in ticks])
    top.minorticks_off()
    top.set_xlabel("view with $N$ subjets (median scale)")
    fig.tight_layout()
    _save(fig, outdir, f"rg_reach{'_' + tag if tag else ''}", formats)


def plot_scaling(paths, outdir, labels=None, tag="", formats=("pdf",), ratio=False, styles=None):
    """<N_excl>(q) and its local exponent; several results are overlaid (e.g. two samples).

    With ratio=True a middle panel shows <N_excl>(q) of each result divided by the first one
    (all results must share the q grid). styles: optional matplotlib format strings per
    result, e.g. ["k-", "C1-", "C1--"], so that up/down variations can share a colour."""
    if ratio:
        fig, (a1, ar, a2) = plt.subplots(3, 1, figsize=(6.0, 7.6), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1.2, 1]})
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    n_ref = None
    for i, path in enumerate(paths):
        d = _load(path)
        lab = labels[i] if labels and i < len(labels) else f"{os.path.basename(d['views_file'])}:{d['level']}"
        q = np.array(d["qgrid"])
        n = np.array(d["mean_nexcl"])
        g = np.array([np.nan if x is None else x for x in d["local_exponent"]])
        fmt = styles[i] if styles and i < len(styles) else f"C{i}-"
        a1.plot(q, n, fmt, marker=".", ms=3, label=f"{lab} ({d['n_jets']} jets)")
        a2.plot(q, g, fmt, marker=".", ms=3)
        if ratio:
            if n_ref is None:
                n_ref = n
            ar.plot(q, n / n_ref, fmt, marker=".", ms=3)
        if i == 0:
            for k, v in d["q_kt"].items():
                if not v["median"]:
                    continue
                a1.axvline(v["median"], color="gray", lw=0.6, ls=":")
                a1.text(v["median"], np.nanmax(n), k, fontsize=7, rotation=90, va="top")
            min_const = d["min_constituents"]
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylabel(r"$\langle N_\mathrm{excl}\rangle$ (exclusive $k_\mathrm{T}$)")
    a1.set_title(f"jets with >= {min_const} constituents; dotted: median $q_N$ of the first sample", fontsize=9)
    a1.legend(fontsize=8, frameon=False, loc="lower left")
    if ratio:
        ar.axhline(1.0, color="gray", lw=0.8, ls="--")
        ar.set_ylabel(f"ratio to {labels[0] if labels else 'first'}", fontsize=8)
    a2.set_xlabel(r"resolution $q = \sqrt{d_\mathrm{cut}}$ [GeV]")
    a2.set_ylabel(r"$-d\ln\langle N\rangle / d\ln q$")
    fig.tight_layout()
    stem = f"rg_scaling_{tag}" if tag else f"rg_scaling_{_load(paths[0])['level']}"
    _save(fig, outdir, stem, formats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classify", nargs="*", default=[])
    ap.add_argument("--paired", default=None)
    ap.add_argument("--scaling", nargs="*", default=[], help="one or more scaling results, overlaid")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels, one per --scaling result, or one per --classify result if no --scaling")
    ap.add_argument("--ratio", action="store_true", help="add a panel with <N_excl> divided by the first --scaling result")
    ap.add_argument("--styles", nargs="*", default=None, help="matplotlib format per --scaling result, e.g. k- C1- C1--")
    ap.add_argument("--ratio-only", action="store_true",
                    help="draw only the ratio of each --scaling result to the first one (rg_scaling_ratio_<tag>)")
    ap.add_argument("--reach", nargs="+", default=None, metavar="SCALING_JSON",
                    help="reach figure: REF COUPLING CUTOFF HADRON [HADRON ...] scaling results; the ratio of "
                         "<N_excl>(q) to REF on an inverted q axis, with the scales of the changes marked")
    ap.add_argument("--no-q-axis", action="store_true",
                    help="omit the q_N top axis of a classify figure (e.g. when the first file is a parton-level sample)")
    ap.add_argument("--tag", default="", help="suffix for the classify/scaling figure names")
    ap.add_argument("--formats", nargs="+", default=["pdf"], help="e.g. pdf png")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    if args.classify:
        plot_classify(args.classify, args.outdir, args.tag, args.formats, q_axis=not args.no_q_axis,
                      labels=None if args.scaling else args.labels)
    if args.paired:
        plot_paired(args.paired, args.outdir, args.formats)
    if args.reach:
        if len(args.reach) < 4:
            ap.error("--reach needs REF COUPLING CUTOFF and at least one HADRON scaling result")
        plot_reach(args.reach[0], args.reach[1], args.reach[2], args.reach[3:], args.outdir, args.tag, args.formats)
    if args.scaling:
        if args.ratio_only:
            plot_scaling_ratio(args.scaling, args.outdir, args.labels, args.tag, args.formats, args.styles)
        else:
            plot_scaling(args.scaling, args.outdir, args.labels, args.tag, args.formats, args.ratio, args.styles)


if __name__ == "__main__":
    main()

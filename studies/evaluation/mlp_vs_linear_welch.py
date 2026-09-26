"""Welch's t-test between two finetuning heads (MLP vs. linear) per method.

For each method, compares the per-seed OOD accuracies of two conditions (by
default ``head=mlp`` against ``head=linear``) with Welch's unequal-variance
t-test:

    t  = (mu_A - mu_B) / sqrt(s_A^2 / n_A + s_B^2 / n_B)
    df = (s_A^2/n_A + s_B^2/n_B)^2 / [(s_A^2/n_A)^2/(n_A-1) + (s_B^2/n_B)^2/(n_B-1)]

with sample standard deviations (ddof=1) and a two-sided p-value. Inputs are
the per-seed CSV of ``harvest_finetune.py`` or, when only published summaries
are available, a summary CSV with ``mean``, ``std`` and ``n`` columns (for
instance written by ``make_table.py --summary-out``).

Usage:
    python studies/evaluation/mlp_vs_linear_welch.py \
        --csv experiments/studies/finetune_per_seed.csv --subset Tbqq_vs_QCD

    # From summary statistics
    python studies/evaluation/mlp_vs_linear_welch.py --csv <path/to/summary.csv> --format latex
"""

import argparse
import math

import numpy as np
from scipy import stats

from common import as_float, filter_rows, fmt_pm, parse_where, read_csvs, render_table


def welch_from_stats(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> dict:
    """Welch t, Welch-Satterthwaite degrees of freedom and two-sided p from summary statistics."""
    v1, v2 = s1**2 / n1, s2**2 / n2
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    p = 2.0 * stats.t.sf(abs(t), df)
    return {"t": t, "df": df, "p": p}


def collect_conditions(rows: list[dict], group: str, contrast: str, cond_a: str, cond_b: str, metric: str) -> dict:
    """``{method: {cond: {"mean", "std", "n"}}}`` from per-seed rows (ddof=1) or summary rows."""
    out: dict[str, dict] = {}
    rows = [r for r in rows if r.get(contrast) in (cond_a, cond_b)]
    if rows and "run" in rows[0]:
        values: dict[tuple, list[float]] = {}
        for r in rows:
            values.setdefault((r[group], r[contrast]), []).append(as_float(r[metric]))
        for (method, cond), v in values.items():
            arr = np.asarray(v)
            out.setdefault(method, {})[cond] = {"mean": arr.mean(), "std": arr.std(ddof=1), "n": len(arr), "values": arr}
    else:
        for r in rows:
            if r.get("metric", metric) != metric:
                continue
            n = as_float(r.get("n"))
            if math.isnan(n):
                raise SystemExit(f"Summary row for {r[group]} ({contrast}={r[contrast]}) has no sample size 'n'.")
            out.setdefault(r[group], {})[r[contrast]] = {"mean": as_float(r["mean"]), "std": as_float(r["std"]), "n": int(n)}
    return {m: c for m, c in out.items() if cond_a in c and cond_b in c}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, nargs="+", required=True, help="Per-seed or summary CSV(s).")
    parser.add_argument("--where", type=str, action="append", default=None, help="Filter key=value[,value]; repeatable.")
    parser.add_argument("--subset", type=str, default="Tbqq_vs_QCD", help="Label subset to compare.")
    parser.add_argument("--metric", type=str, default="acc", choices=["acc", "auc"])
    parser.add_argument("--group", type=str, default="method", help="Column identifying a method.")
    parser.add_argument("--contrast", type=str, default="head", help="Column holding the two conditions.")
    parser.add_argument("--a", type=str, default="mlp", help="First condition (numerator sign).")
    parser.add_argument("--b", type=str, default="linear", help="Second condition.")
    parser.add_argument("--order", type=str, nargs="+", default=None, help="Row order of methods.")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown")
    args = parser.parse_args()

    where = parse_where(args.where)
    rows = read_csvs(args.csv)
    if rows and "subset" in rows[0]:
        where["subset"] = [args.subset]
    rows = filter_rows(rows, where)
    if not rows:
        raise SystemExit("No rows left after filtering.")
    conditions = collect_conditions(rows, args.group, args.contrast, args.a, args.b, args.metric)
    if not conditions:
        raise SystemExit(f"No {args.group} has both {args.contrast}={args.a} and {args.contrast}={args.b}.")

    methods = list(conditions)
    if args.order:
        methods = [m for m in args.order if m in conditions] + [m for m in methods if m not in args.order]

    header = [args.group, args.a, args.b, "n_a", "n_b", "t", "df", "p (two-sided)"]
    body = []
    for m in methods:
        a, b = conditions[m][args.a], conditions[m][args.b]
        w = welch_from_stats(a["mean"], a["std"], a["n"], b["mean"], b["std"], b["n"])
        t, p, df = w["t"], w["p"], w["df"]
        if "values" in a and "values" in b:
            # Per-seed inputs: take t and p from scipy's Welch test (same values).
            res = stats.ttest_ind(a["values"], b["values"], equal_var=False)
            t, p = float(res.statistic), float(res.pvalue)
        body.append([
            m,
            fmt_pm(a["mean"], a["std"], args.digits, args.format),
            fmt_pm(b["mean"], b["std"], args.digits, args.format),
            str(a["n"]),
            str(b["n"]),
            f"{t:+.1f}" if args.format != "csv" else f"{t:.4f}",
            f"{df:.1f}",
            f"{p:.2g}",
        ])
    print(render_table(header, body, args.format))


if __name__ == "__main__":
    main()

"""Build mean ± std tables from harvested CSV files.

Reads per-seed CSVs from ``harvest_finetune.py`` (aggregated here over runs)
and/or summary CSVs from ``harvest_probes.py`` (used as stored), filters them,
and pivots them into a table whose rows and columns are chosen from the tag
columns. With ``--highlight``, the best mean of each column is bold and entries
within one standard deviation of the best (that of the best entry) are
underlined, as in the paper's tables.

Usage:
    # Table 1: k-NN, linear probe, linear head and MLP head on t->bqq vs QCD
    python studies/evaluation/make_table.py \
        --csv experiments/studies/finetune_per_seed.csv experiments/studies/probes.csv \
        --rows method --cols head --col-order knn linear_probe linear mlp \
        --subset Tbqq_vs_QCD --metric acc --highlight --format latex

    # Cross-topology (linear head): ACC and AUC per unseen topology
    python studies/evaluation/make_table.py --csv experiments/studies/finetune_per_seed.csv \
        --where head=linear --rows method --cols subset metric \
        --subset H4q_vs_QCD Hbb_vs_QCD --metric acc auc --highlight

    # Ablation block with the relative change of the OOD accuracy
    python studies/evaluation/make_table.py --csv experiments/studies/ablation_per_seed.csv \
        --where head=mlp --rows variant --cols metric --metric acc --gain-ref previous
"""

import argparse
import math
import sys

from common import (
    as_float,
    filter_rows,
    fmt_pm,
    parse_where,
    read_csvs,
    render_table,
    to_summary,
    write_csv,
)


def _ordered(values: list, order: list | None) -> list:
    """Unique values in first-seen order, or following ``order`` (values not listed go last)."""
    unique = list(dict.fromkeys(values))
    if not order:
        return unique
    ranked = [v for v in order if v in unique]
    return ranked + [v for v in unique if v not in ranked]


def pivot(summary: list[dict], rows: list[str], cols: list[str], row_order=None, col_order=None):
    """Map ``(row_key, col_key) -> summary row``; each cell must be unique."""
    for c in rows + cols:
        missing = [r for r in summary if c not in r]
        if missing:
            raise SystemExit(f"Column '{c}' is missing from {len(missing)} rows; available: {sorted(summary[0])}")
    row_keys = _ordered([" / ".join(str(r[c]) for c in rows) for r in summary], row_order)
    col_keys = _ordered([" / ".join(str(r[c]) for c in cols) for r in summary], col_order)
    cells: dict[tuple, dict] = {}
    for r in summary:
        key = (" / ".join(str(r[c]) for c in rows), " / ".join(str(r[c]) for c in cols))
        if key in cells:
            raise SystemExit(
                f"Several entries for row '{key[0]}', column '{key[1]}'. Add --where filters or more --rows/--cols."
            )
        cells[key] = r
    return row_keys, col_keys, cells


def highlight_flags(row_keys, col_keys, cells, lower_is_better: bool = False) -> dict[tuple, tuple[bool, bool]]:
    """``(bold, underline)`` per cell: bold = best mean in its column; underline = within the best's 1 sigma."""
    flags: dict[tuple, tuple[bool, bool]] = {}
    for ck in col_keys:
        entries = [(rk, cells[(rk, ck)]) for rk in row_keys if (rk, ck) in cells]
        entries = [(rk, c) for rk, c in entries if not math.isnan(as_float(c["mean"]))]
        if not entries:
            continue
        pick = min if lower_is_better else max
        best_rk, best = pick(entries, key=lambda e: as_float(e[1]["mean"]))
        best_mean, best_std = as_float(best["mean"]), as_float(best["std"], 0.0)
        for rk, c in entries:
            is_best = rk == best_rk
            within = (not is_best) and abs(as_float(c["mean"]) - best_mean) <= best_std
            flags[(rk, ck)] = (is_best, within)
    return flags


def gain_column(row_keys, col_key, cells, reference: str) -> list[str]:
    """Relative change (%) of ``col_key`` means w.r.t. the previous or first row."""
    out = []
    means = [as_float(cells[(rk, col_key)]["mean"]) if (rk, col_key) in cells else float("nan") for rk in row_keys]
    for i, m in enumerate(means):
        ref = means[0] if reference == "first" else (means[i - 1] if i > 0 else float("nan"))
        out.append("" if i == 0 or math.isnan(ref) or math.isnan(m) else f"{100.0 * (m - ref) / ref:+.1f}%")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, nargs="+", required=True, help="Per-seed and/or summary CSV files.")
    parser.add_argument("--rows", type=str, nargs="+", default=["method"], help="Columns forming the table rows.")
    parser.add_argument("--cols", type=str, nargs="+", default=["head"], help="Columns forming the table columns.")
    parser.add_argument("--row-order", type=str, nargs="+", default=None, help="Row order (row keys joined by ' / ').")
    parser.add_argument("--col-order", type=str, nargs="+", default=None, help="Column order (keys joined by ' / ').")
    parser.add_argument("--metric", type=str, nargs="+", default=["acc"], help="Metrics to keep (acc, auc).")
    parser.add_argument("--subset", type=str, nargs="+", default=None, help="Subsets to keep (e.g. Tbqq_vs_QCD).")
    parser.add_argument("--where", type=str, action="append", default=None, help="Filter key=value[,value]; repeatable.")
    parser.add_argument("--ddof", type=int, default=1, help="Delta degrees of freedom of the std over seeds.")
    parser.add_argument("--digits", type=int, default=3, help="Decimal digits.")
    parser.add_argument("--format", choices=["markdown", "latex", "csv"], default="markdown")
    parser.add_argument("--highlight", action="store_true", help="Bold the best entry per column, underline those within its 1 sigma.")
    parser.add_argument("--lower-is-better", action="store_true", help="Treat the smallest mean as best when highlighting.")
    parser.add_argument("--gain-ref", choices=["previous", "first"], default=None, help="Append the relative change of --gain-col.")
    parser.add_argument("--gain-col", type=str, default=None, help="Column key for --gain-ref (default: the last column).")
    parser.add_argument("--summary-out", type=str, default=None, help="Also write the aggregated long-format CSV here.")
    args = parser.parse_args()

    rows = read_csvs(args.csv)
    summary = to_summary(rows, metrics=tuple(args.metric), ddof=args.ddof)
    where = parse_where(args.where)
    where["metric"] = args.metric
    if args.subset:
        where["subset"] = args.subset
    summary = filter_rows(summary, where)
    if not summary:
        raise SystemExit("No rows left after filtering.")
    if args.summary_out:
        write_csv(args.summary_out, summary)

    row_keys, col_keys, cells = pivot(summary, args.rows, args.cols, args.row_order, args.col_order)
    flags = highlight_flags(row_keys, col_keys, cells, args.lower_is_better) if args.highlight else {}

    header = [" / ".join(args.rows)] + col_keys
    body = []
    for rk in row_keys:
        line = [rk]
        for ck in col_keys:
            c = cells.get((rk, ck))
            if c is None:
                line.append(fmt_pm(float("nan"), float("nan"), args.digits, args.format))
                continue
            bold, under = flags.get((rk, ck), (False, False))
            line.append(fmt_pm(as_float(c["mean"]), as_float(c["std"]), args.digits, args.format, bold, under))
        body.append(line)

    if args.gain_ref:
        gain_col = args.gain_col or col_keys[-1]
        if gain_col not in col_keys:
            raise SystemExit(f"--gain-col '{gain_col}' not among columns {col_keys}")
        header.append(f"change vs {args.gain_ref}")
        for line, g in zip(body, gain_column(row_keys, gain_col, cells, args.gain_ref)):
            line.append(g)

    # Seeds per cell, to make missing runs visible.
    counts = sorted({str(c.get("n", "")) or "not stored" for c in cells.values()})
    print(render_table(header, body, args.format))
    print(f"\n(n per cell: {', '.join(counts)}; std ddof={args.ddof} for per-seed inputs)", file=sys.stderr)


if __name__ == "__main__":
    main()

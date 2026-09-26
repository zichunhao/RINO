#!/usr/bin/env python3
"""Tabulate domain-shift results written by ``compute_domain_shift.py``.

* ``--finetuned`` only: W1/MMD of the finetuned representations
  (Table "Domain shift in finetuned representations").
* ``--pretrained`` and ``--finetuned``: W1/MMD of the pretrained
  representations with the signed change after finetuning,
  finetuned - pretrained (App. table "Domain shift in pretrained
  representations"). Methods are matched by their ``method`` name.

Values are means over runs; ``± std`` is shown when a method has more than one
run.

Usage:
    python studies/representation/domain_shift/summarize_domain_shift.py \
        --finetuned results/domain_shift/finetuned/*.json \
        --pretrained results/domain_shift/pretrained/*.json \
        --format latex --output results/domain_shift/pretrained_table.tex
"""

import argparse
import csv
import io
import json
from pathlib import Path

METRICS = ("w1_qcd", "w1_top", "mmd_qcd", "mmd_top")
HEADERS = {
    "w1_qcd": ("W1 (QCD)", r"$W_1$ (QCD) $\downarrow$"),
    "w1_top": ("W1 (top)", r"$W_1$ (top) $\downarrow$"),
    "mmd_qcd": ("MMD (QCD)", r"MMD (QCD) $\downarrow$"),
    "mmd_top": ("MMD (top)", r"MMD (top) $\downarrow$"),
}


def load_results(paths: list[str]) -> dict[str, dict]:
    """``{method: summary}`` from compute_domain_shift.py JSON files."""
    results = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        method = data["method"]
        if method in results:
            raise ValueError(f"Method '{method}' appears in more than one file")
        results[method] = data["summary"]
    return results


def _number(x: float, digits: int, signed: bool = False) -> str:
    """Format a value; signed values drop the leading zero (e.g. -.218)."""
    if not signed:
        return f"{x:.{digits}f}"
    s = f"{x:+.{digits}f}"
    return s.replace("+0.", "+.").replace("-0.", "-.")


def format_cell(
    stat: dict, delta: float | None, digits: int, fmt: str
) -> str:
    """One table cell: mean (± std over runs) and optionally the signed change."""
    value = _number(stat["mean"], digits)
    if stat.get("n_runs", 1) > 1:
        pm = r" \pm " if fmt == "latex" else " ± "
        value += pm + _number(stat["std"], digits)
    if fmt == "latex":
        cell = f"${value}$"
        if delta is not None:
            color = "ForestGreen" if delta < 0 else "red"
            cell += rf" {{\scriptsize\color{{{color}}}${_number(delta, digits, True)}$}}"
        return cell
    if delta is not None:
        return f"{value} ({_number(delta, digits, True)})"
    return value


def build_rows(
    main: dict[str, dict],
    finetuned: dict[str, dict] | None,
    order: list[str] | None,
    digits: int,
    fmt: str,
) -> list[list[str]]:
    """Rows ``[method, cell...]`` of the table, in ``order`` if given."""
    methods = order or sorted(main)
    unknown = [m for m in methods if m not in main]
    if unknown:
        raise KeyError(f"No results for {unknown}; available: {sorted(main)}")
    rows = []
    for method in methods:
        cells = [method]
        for metric in METRICS:
            delta = None
            if finetuned is not None:
                if method not in finetuned:
                    raise KeyError(f"No finetuned results for '{method}'")
                delta = finetuned[method][metric]["mean"] - main[method][metric]["mean"]
            cells.append(format_cell(main[method][metric], delta, digits, fmt))
        rows.append(cells)
    return rows


def render(rows: list[list[str]], fmt: str) -> str:
    """Render rows as a markdown table, a LaTeX tabular body or CSV."""
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["method", *METRICS])
        writer.writerows(rows)
        return buf.getvalue()
    if fmt == "latex":
        header = [r"\textbf{Method}", *(HEADERS[m][1] for m in METRICS)]
        lines = [" & ".join(header) + r" \\", r"\midrule"]
        lines += [" & ".join(r) + r" \\" for r in rows]
        return "\n".join(lines) + "\n"
    header = ["Method", *(HEADERS[m][0] for m in METRICS)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabulate W1/MMD domain-shift results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--finetuned", nargs="+", required=True, help="Finetuned-representation JSONs."
    )
    parser.add_argument(
        "--pretrained",
        nargs="+",
        default=None,
        help="Pretrained-representation JSONs; if given, the table shows these "
        "values with the signed change after finetuning.",
    )
    parser.add_argument("--order", nargs="+", default=None, help="Row order (method names).")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--format", default="markdown", choices=("markdown", "latex", "csv"))
    parser.add_argument("--output", "-o", default=None, help="Write here instead of stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    finetuned = load_results(args.finetuned)
    if args.pretrained:
        pretrained = load_results(args.pretrained)
        rows = build_rows(pretrained, finetuned, args.order, args.digits, args.format)
    else:
        rows = build_rows(finetuned, None, args.order, args.digits, args.format)
    text = render(rows, args.format)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        print(f"Saved {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

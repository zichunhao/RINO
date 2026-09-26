"""Scale-factor proxy, TopTagging -> JetClass (appendix table ``tab:sf-proxy``).

For each finetuning run of a method, the signal score s = sigmoid(logit) is
read from the inference outputs of the source (TopTagging test set, signal
label 1) and the target (JetClass test set, signal label 8 = Tbqq). The source
signal-score CDF is cut at the quantiles ``--rank-edges`` (w-quantiles q_w,
with q_0 = -inf and q_1 = +inf), and every interval [a, b) selects jets with
q_a <= s < q_b in both domains (ties at a boundary go to the higher bin). The
per-interval scale factor is

    SF_[a,b) = eps_tgt / eps_src,  eps_d = P_d(q_a <= s < q_b | signal).

Each run is checked for closure: the source and target fractions each sum to
one, and the source-weighted SFs sum to one. Per interval, the table reports
the mean +/- std (ddof=1) over runs, and the last column is the mean over
intervals of |mean SF - 1|.

Two subcommands:

``compute``
    Reads ``<EXP_DIR>/run-*/inference/output_test_{toptagging,jetclass}_best-0.pt``
    per method, writes ``sf_proxy_<NAME>.json`` per method and prints the table::

        python studies/physics/scale_factor_proxy.py compute \\
            --method Supervised=PROJECT_ROOT/experiments/<tt-finetune-supervised> \\
            --method RINO=PROJECT_ROOT/experiments/<tt-finetune-rino> \\
            --output-dir PROJECT_ROOT/experiments/studies/physics/sf_proxy

``table``
    Re-prints the table from the JSON files, e.g. as LaTeX rows::

        python studies/physics/scale_factor_proxy.py table \\
            --inputs "PROJECT_ROOT/experiments/studies/physics/sf_proxy/*.json" \\
            --format latex
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from _common import resolve_path, write_json

DEFAULT_RANK_EDGES = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]


def load_signal_scores(
    path: Path, signal_labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Signal-jet scores and all labels from one inference output file.

    The score is sigmoid(logit) evaluated in the stored precision of the
    logits, so saturated scores tie exactly as in the paper tables.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    logits = data["logits"].numpy()
    if logits.ndim == 2:
        if logits.shape[1] != 1:
            raise ValueError(
                f"{path}: expected binary logits of shape (N, 1), got {logits.shape}"
            )
        logits = logits[:, 0]
    with np.errstate(over="ignore"):
        scores = 1.0 / (1.0 + np.exp(-logits))
    labels = data["label"].numpy().astype(int).reshape(-1)
    return scores[np.isin(labels, signal_labels)], labels


def bin_scale_factors(
    source: np.ndarray, target: np.ndarray, rank_edges: list[float]
) -> list[dict]:
    """Per-interval source/target signal fractions and SF for one run."""
    internal = np.quantile(source, rank_edges[1:-1], method="linear")
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    rows = []
    for (a, b), low, high in zip(
        zip(rank_edges[:-1], rank_edges[1:]), edges[:-1], edges[1:]
    ):
        frac_source = float(np.mean((source >= low) & (source < high)))
        frac_target = float(np.mean((target >= low) & (target < high)))
        if frac_source <= 0.0:
            raise RuntimeError(
                f"Empty source interval [{a}, {b}) with score bounds [{low}, {high})"
            )
        rows.append(
            {
                "threshold_low": float(low),
                "threshold_high": float(high),
                "fraction_source": frac_source,
                "fraction_target": frac_target,
                "sf": frac_target / frac_source,
            }
        )

    closures = {
        "source fractions": sum(r["fraction_source"] for r in rows),
        "target fractions": sum(r["fraction_target"] for r in rows),
        "source-weighted SF": sum(r["fraction_source"] * r["sf"] for r in rows),
    }
    for what, total in closures.items():
        if not np.isclose(total, 1.0, atol=1e-12):
            raise RuntimeError(f"Closure failed: {what} sum to {total}")
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def compute_method(
    name: str,
    exp_dir: Path,
    args: argparse.Namespace,
    expected_runs: int | None,
) -> dict:
    """Scale factors of every complete run of one method, summarized per bin."""
    run_ids, skipped, per_run = [], [], []
    reference_labels = None
    for run_dir in sorted(exp_dir.glob(args.run_glob)):
        source_path = run_dir / args.source_file
        target_path = run_dir / args.target_file
        if not (source_path.exists() and target_path.exists()):
            skipped.append({"run_id": run_dir.name, "reason": "missing inference file"})
            print(f"[{name}] skip {run_dir.name}: missing inference file")
            continue
        source, source_labels = load_signal_scores(source_path, args.source_signal)
        target, target_labels = load_signal_scores(target_path, args.target_signal)

        # All runs must score the same test jets in the same order.
        if reference_labels is None:
            reference_labels = (source_labels, target_labels)
        elif not (
            np.array_equal(reference_labels[0], source_labels)
            and np.array_equal(reference_labels[1], target_labels)
        ):
            raise RuntimeError(f"[{name}] label ordering differs in {run_dir.name}")

        per_run.append(bin_scale_factors(source, target, args.rank_edges))
        run_ids.append(run_dir.name)

    if not per_run:
        raise RuntimeError(f"[{name}] no complete runs under {exp_dir}")
    if expected_runs is not None and len(per_run) != expected_runs:
        raise RuntimeError(
            f"[{name}] expected {expected_runs} complete runs, found "
            f"{len(per_run)}: {run_ids}"
        )
    print(
        f"[{name}] {len(per_run)} runs, {len(source):,} source and "
        f"{len(target):,} target signal jets"
    )

    bins = []
    for index, (a, b) in enumerate(zip(args.rank_edges[:-1], args.rank_edges[1:])):
        rows = [run[index] for run in per_run]
        sf_mean, sf_std = _mean_std([r["sf"] for r in rows])
        entry = {
            "rank_low": a,
            "rank_high": b,
            "sf_mean": sf_mean,
            "sf_std": sf_std,
            "sf_per_run": [r["sf"] for r in rows],
            "fraction_source_mean": _mean_std([r["fraction_source"] for r in rows])[0],
            "fraction_target_mean": _mean_std([r["fraction_target"] for r in rows])[0],
        }
        # Score thresholds; the outermost bounds are infinite and omitted.
        for side in ("low", "high"):
            values = [r[f"threshold_{side}"] for r in rows]
            if np.all(np.isfinite(values)):
                entry[f"threshold_{side}_mean"], entry[f"threshold_{side}_std"] = (
                    _mean_std(values)
                )
        bins.append(entry)

    return {
        "name": name,
        "exp_dir": str(exp_dir),
        "source_file": args.source_file,
        "target_file": args.target_file,
        "source_signal_labels": args.source_signal,
        "target_signal_labels": args.target_signal,
        "score": "sigmoid(logit)",
        "quantile_method": "numpy linear",
        "interval_rule": "threshold_low <= score < threshold_high",
        "rank_edges": args.rank_edges,
        "n_runs": len(per_run),
        "run_ids": run_ids,
        "skipped_runs": skipped,
        "bins": bins,
        "mean_abs_sf_minus_one": float(np.mean([abs(b["sf_mean"] - 1) for b in bins])),
    }


def _interval_label(a: float, b: float, last: bool) -> str:
    return f"[{a:g}, {b:g}{']' if last else ')'}"


def format_table(results: list[dict], fmt: str = "markdown") -> str:
    """Methods x intervals table of mean +/- std SF and mean |SF - 1|.

    In LaTeX mode the entry closest to unity in each column is bold.
    """
    edges = results[0]["rank_edges"]
    for result in results:
        if result["rank_edges"] != edges:
            raise ValueError("All methods must use the same rank edges")
    n_bins = len(edges) - 1
    headers = ["Method"] + [
        _interval_label(edges[i], edges[i + 1], i == n_bins - 1) for i in range(n_bins)
    ]
    headers.append("mean abs(SF-1)")

    # Distance from unity per column, used to bold the best entry.
    distance = np.array(
        [
            [abs(b["sf_mean"] - 1) for b in r["bins"]] + [r["mean_abs_sf_minus_one"]]
            for r in results
        ]
    )
    best = distance.argmin(axis=0)

    lines = []
    if fmt == "latex":
        headers[-1] = r"$\langle|\mathrm{SF}-1|\rangle$"
        lines.append(" & ".join(headers) + r" \\")
    else:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
    for row, result in enumerate(results):
        cells = [f"{b['sf_mean']:.3f} ± {b['sf_std']:.3f}" for b in result["bins"]]
        cells.append(f"{result['mean_abs_sf_minus_one']:.3f}")
        if fmt == "latex":
            cells = [
                (rf"$\mathbf{{{c}}}$" if best[col] == row else f"${c}$").replace(
                    "±", r"\pm"
                )
                for col, c in enumerate(cells)
            ]
            lines.append(" & ".join([result["name"]] + cells) + r" \\")
        else:
            lines.append("| " + " | ".join([result["name"]] + cells) + " |")
    return "\n".join(lines)


def _parse_pair(spec: str) -> tuple[str, str]:
    name, sep, value = spec.partition("=")
    if not sep or not name or not value:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got {spec!r}")
    return name, value


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def run_compute(args: argparse.Namespace) -> None:
    if args.rank_edges[0] != 0.0 or args.rank_edges[-1] != 1.0:
        raise ValueError("--rank-edges must start at 0 and end at 1")
    if list(args.rank_edges) != sorted(set(args.rank_edges)):
        raise ValueError("--rank-edges must be strictly increasing")
    expected = {name: int(n) for name, n in args.expected_runs or []}
    unknown = set(expected) - {name for name, _ in args.method}
    if unknown:
        raise ValueError(f"--expected-runs names without --method: {sorted(unknown)}")

    out_dir = resolve_path(args.output_dir)
    results = []
    for name, exp_dir in args.method:
        result = compute_method(name, resolve_path(exp_dir), args, expected.get(name))
        write_json(result, out_dir / f"sf_proxy_{_safe_name(name)}.json")
        results.append(result)
    print()
    print(format_table(results, args.format))


def run_table(args: argparse.Namespace) -> None:
    paths = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(str(resolve_path(pattern))))
        if not matches:
            raise FileNotFoundError(f"No files match {pattern!r}")
        paths.extend(matches)
    results = []
    for path in paths:
        with open(path) as handle:
            results.append(json.load(handle))
    print(format_table(results, args.format))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compute = sub.add_parser("compute", help="Compute SFs from inference outputs.")
    compute.add_argument(
        "--method",
        action="append",
        type=_parse_pair,
        required=True,
        metavar="NAME=EXP_DIR",
        help="Method name and its finetuning experiment directory (holding "
        "run-* subdirectories). Repeat per method; table rows keep this order.",
    )
    compute.add_argument(
        "--expected-runs",
        action="append",
        type=_parse_pair,
        metavar="NAME=N",
        help="Fail unless exactly N complete runs are found for NAME.",
    )
    compute.add_argument("--run-glob", default="run-*", help="Run directory pattern.")
    compute.add_argument(
        "--source-file",
        default="inference/output_test_toptagging_best-0.pt",
        help="Source (TopTagging) inference output, relative to each run directory.",
    )
    compute.add_argument(
        "--target-file",
        default="inference/output_test_jetclass_best-0.pt",
        help="Target (JetClass) inference output, relative to each run directory.",
    )
    compute.add_argument(
        "--source-signal",
        type=int,
        nargs="+",
        default=[1],
        help="Signal label(s) in the source file (TopTagging top = 1).",
    )
    compute.add_argument(
        "--target-signal",
        type=int,
        nargs="+",
        default=[8],
        help="Signal label(s) in the target file (JetClass Tbqq = 8).",
    )
    compute.add_argument(
        "--rank-edges",
        type=float,
        nargs="+",
        default=DEFAULT_RANK_EDGES,
        help="Source signal-score CDF edges, from 0 to 1.",
    )
    compute.add_argument(
        "--output-dir",
        default="PROJECT_ROOT/experiments/studies/physics/sf_proxy",
        help="Directory for the per-method JSON files.",
    )
    compute.add_argument("--format", choices=["markdown", "latex"], default="markdown")

    table = sub.add_parser("table", help="Print the table from computed JSON files.")
    table.add_argument(
        "--inputs", nargs="+", required=True, help="JSON files or glob patterns."
    )
    table.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compute":
        run_compute(args)
    else:
        run_table(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Domain shift between source (JetNet) and target (JetClass) representations.

For one method, reads the inference outputs of one or more runs (e.g. the
finetuning seeds) and computes, separately for QCD and top jets,

    W1   per-dimension 1-Wasserstein distance, averaged over dimensions
    MMD  maximum mean discrepancy with the RBF kernel exp(-gamma ||x - y||^2),
         gamma = 1/d by default (d = representation dimension)

on a seeded random subsample of ``--n-per-domain`` jets per domain and class
(5,000 by default, the protocol of App. "Domain Shift in Representation
Space"). The metric functions are those of ``dino/domain_shift_metrics.py``.
With several runs the JSON also holds the mean and standard deviation over
runs (the error-bar variant of the table).

Inputs are the ``.pt`` files written by ``dino/dino_inference.py`` (finetuned
models) or by ``extract_pretrained_reps.py`` (pretrained models); each must
contain ``rep`` (N, d) and ``label`` (N,).

Usage:
    # Finetuned representations, one directory per finetuning seed
    python studies/representation/domain_shift/compute_domain_shift.py \
        --method RINO \
        --inference-dirs experiments/<finetune-run>/run-*/inference \
        --output results/domain_shift/finetuned/RINO.json

    # Pretrained representations from extract_pretrained_reps.py
    python studies/representation/domain_shift/compute_domain_shift.py \
        --method RINO --inference-dirs experiments/<run>/pretrained-reps \
        --source-file output_test_jetnet_pretrained-0.pt \
        --target-file output_test_jetclass_pretrained-0.pt \
        --output results/domain_shift/pretrained/RINO.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "dino"))

from domain_shift_metrics import compute_w1_per_dim, mmd_rbf  # noqa: E402

CLASSES = ("qcd", "top")
METRICS = tuple(f"{m}_{c}" for m in ("w1", "mmd") for c in CLASSES)


def load_representations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(rep, label)`` arrays from an inference output file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("rep", "label") if k not in data]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    rep = data["rep"].float().numpy()
    label = data["label"].numpy().astype(int).reshape(-1)
    if len(rep) != len(label):
        raise ValueError(f"{path}: {len(rep)} representations but {len(label)} labels")
    return rep, label


def split_classes(
    rep: np.ndarray, label: np.ndarray, qcd_label: int, top_label: int, name: str
) -> dict[str, np.ndarray]:
    """Representations of the QCD and top jets of one domain."""
    out = {"qcd": rep[label == qcd_label], "top": rep[label == top_label]}
    for cls, x in out.items():
        if len(x) == 0:
            raise ValueError(
                f"No {cls} jets in {name} (labels present: {sorted(set(label.tolist()))})"
            )
    return out


def subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random subset of ``n`` rows without replacement (all rows if fewer)."""
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), size=n, replace=False)]


def domain_shift(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    n_per_domain: int,
    gamma: float | None,
    seed: int,
) -> dict[str, float | int]:
    """W1 and MMD between source and target, per class.

    The same ``n_per_domain``-jet subsets are used for both metrics. Subsets
    are drawn in the order source QCD, source top, target QCD, target top, as
    in ``dino/domain_shift_metrics.py``, so both give the same values for the
    same inputs and seed.
    """
    rng = np.random.default_rng(seed)
    source = {cls: subsample(source[cls], n_per_domain, rng) for cls in CLASSES}
    target = {cls: subsample(target[cls], n_per_domain, rng) for cls in CLASSES}
    out: dict[str, float | int] = {}
    for cls in CLASSES:
        xs, xt = source[cls], target[cls]
        out[f"w1_{cls}"] = float(compute_w1_per_dim(xs, xt).mean())
        out[f"mmd_{cls}"] = float(mmd_rbf(xs, xt, gamma=gamma))
        out[f"n_source_{cls}"] = int(len(xs))
        out[f"n_target_{cls}"] = int(len(xt))
    return out


def summarize(runs: list[dict]) -> dict[str, dict[str, float | int]]:
    """Mean and standard deviation (numpy default, ddof=0) of each metric over runs."""
    summary = {}
    for metric in METRICS:
        vals = np.array([r[metric] for r in runs], dtype=float)
        summary[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n_runs": int(len(vals)),
        }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W1/MMD domain shift of jet representations (QCD and top).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True, help="Method name stored in the output.")
    parser.add_argument(
        "--inference-dirs",
        nargs="+",
        required=True,
        help="One directory per run, each holding the source and target files.",
    )
    parser.add_argument("--source-file", default="output_test_jetnet_best-0.pt")
    parser.add_argument("--target-file", default="output_test_jetclass_best-0.pt")
    parser.add_argument(
        "--source-labels",
        nargs=2,
        type=int,
        default=[0, 1],
        metavar=("QCD", "TOP"),
        help="QCD and top labels in the source files (JetNet: 0 = g/q, 1 = t).",
    )
    parser.add_argument(
        "--target-labels",
        nargs=2,
        type=int,
        default=[0, 8],
        metavar=("QCD", "TOP"),
        help="QCD and top labels in the target files (JetClass: 0 = QCD, 8 = t->bqq).",
    )
    parser.add_argument(
        "--n-per-domain", type=int, default=5000, help="Jets per domain and class."
    )
    parser.add_argument(
        "--gamma", type=float, default=None, help="RBF kernel gamma (default: 1/d)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsampling seed (same for every run).")
    parser.add_argument("--output", "-o", required=True, help="Output JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.n_per_domain <= 0:
        raise ValueError("--n-per-domain must be positive")

    runs = []
    for run_dir in map(Path, args.inference_dirs):
        source_rep, source_label = load_representations(run_dir / args.source_file)
        target_rep, target_label = load_representations(run_dir / args.target_file)
        source = split_classes(source_rep, source_label, *args.source_labels, name=str(run_dir))
        target = split_classes(target_rep, target_label, *args.target_labels, name=str(run_dir))
        metrics = domain_shift(source, target, args.n_per_domain, args.gamma, args.seed)
        runs.append({"inference_dir": str(run_dir), "dim": int(source_rep.shape[1]), **metrics})
        print(
            f"{args.method} {run_dir}: "
            + ", ".join(f"{m}={metrics[m]:.4f}" for m in METRICS),
            flush=True,
        )

    summary = summarize(runs)
    result = {
        "method": args.method,
        "protocol": {
            "source_file": args.source_file,
            "target_file": args.target_file,
            "source_labels": dict(zip(CLASSES, args.source_labels)),
            "target_labels": dict(zip(CLASSES, args.target_labels)),
            "n_per_domain": args.n_per_domain,
            "gamma": args.gamma if args.gamma is not None else "1/d",
            "seed": args.seed,
        },
        "runs": runs,
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"{args.method} ({len(runs)} run(s)): "
        + ", ".join(f"{m}={s['mean']:.4f}±{s['std']:.4f}" for m, s in summary.items())
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

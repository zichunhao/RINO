#!/usr/bin/env python3
"""Aggregate inference results across multiple runs.

Loads inference outputs (logits + labels) from each run-{N}/ subdirectory,
computes per-run accuracy, and reports mean +/- std across runs.
Works for both finetuned (RINO + head) and supervised baseline experiments.

Usage:
    # Single experiment
    python dino/aggregate_inferences.py \
        --experiment-dir experiments/finetune/finetune-rino-linear-frozen

    # All experiments under a base dir
    python dino/aggregate_inferences.py --base-dir experiments/finetune

    # Supervised baselines
    python dino/aggregate_inferences.py --base-dir experiments/toptagging-scratch

    # Save results to a JSON file
    python dino/aggregate_inferences.py --base-dir experiments/finetune -o results.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, split: str) -> float:
    """Compute accuracy from logits and labels, handling JetClass label remapping.

    For JetClass with a binary (top-vs-QCD) head, the OOD metric is restricted to
    QCD (label 0) vs Tbqq (label 8), with score > 0 meaning Tbqq. Jets of the other
    signal classes are excluded, since the head was not trained to separate them
    from QCD and counting them would inflate the accuracy.
    """
    if "jetclass" in split:
        if logits.shape[-1] == 1:
            keep = (labels == 0) | (labels == 8)
            logits = logits[keep]
            labels = (labels[keep] == 8).long()
        else:
            # Multi-class head: drop leptonic top (label 9)
            mask = labels != 9
            logits = logits[mask]
            labels = labels[mask]

    if logits.shape[-1] == 1:
        preds = (logits[:, 0] > 0).long()
    else:
        preds = logits.argmax(dim=-1)

    return (preds == labels).float().mean().item()


_SCALAR_METRICS = ("acc", "precision", "recall", "f1", "auc")


def load_split_metrics(inference_dir: Path, split: str) -> dict | None:
    """Load per-split scalar metrics from inference dir.

    Supports two formats:
    - New: metrics_{epoch}.json → {split: {acc, precision, recall, f1, auc, ...}}
    - Old: metrics_{split}_{epoch}.json → {"classification": {"acc": ...}, "knn": {...}, ...}

    Returns a unified dict with at minimum {"acc": float} and optionally
    {"precision", "recall", "f1", "auc"} plus "knn"/"linear_probe" keys.
    """
    # 1. New format: metrics_{epoch}.json with {split: {metrics}} structure
    for json_path in sorted(inference_dir.glob("metrics_*.json")):
        # Skip old-format filenames that embed the split name
        stem = json_path.stem  # e.g. "metrics_best" or "metrics_val_jetnet_best"
        parts = stem.split("_")
        # New format has exactly 2 parts: "metrics" + epoch
        if len(parts) == 2:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if split in data and isinstance(data[split], dict):
                    entry = {k: v for k, v in data[split].items()
                             if k in _SCALAR_METRICS}  # drop roc/cm arrays
                    if entry:
                        return entry
            except (json.JSONDecodeError, KeyError):
                pass

    # 2. Old format: metrics_{split}_{epoch}.json
    old_candidates = sorted(inference_dir.glob(f"metrics_{split}_*.json"))
    if old_candidates:
        with open(old_candidates[0]) as f:
            data = json.load(f)
        entry = {}
        if "classification" in data:
            entry.update({k: v for k, v in data["classification"].items()
                          if k in _SCALAR_METRICS})
        # Preserve knn/linear_probe sections for probe aggregation
        for key in ("knn", "linear_probe"):
            if key in data:
                entry[key] = data[key]
        return entry or None

    return None


def _save_metrics_json(inference_dir: Path, split: str, acc: float, epoch: str = "best"):
    """Save/update the combined metrics_{epoch}.json file."""
    json_path = inference_dir / f"metrics_{epoch}.json"
    data = {}
    if json_path.exists():
        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    data.setdefault(split, {})["acc"] = acc
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _recompute_acc_from_pt(inference_dir: Path, split: str) -> float | None:
    """Recompute acc from the saved logits and labels via compute_accuracy.

    Any `acc` cached in the JSON or .pt files is ignored, so the JetClass value is
    always the restricted Tbqq-vs-QCD metric of compute_accuracy.
    """
    candidates = sorted(inference_dir.glob(f"output_{split}_*.pt"))
    if not candidates:
        return None

    all_logits, all_labels = [], []
    for path in candidates:
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if "logits" in data and "label" in data:
            all_logits.append(data["logits"])
            all_labels.append(data["label"])

    if not all_logits:
        return None

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return compute_accuracy(logits, labels, split)


def load_split_scalar_metrics(inference_dir: Path, split: str) -> dict | None:
    """Load all scalar metrics for a split. Returns dict with acc/precision/recall/f1/auc
    or None if not available. Falls back to recomputing acc from .pt files.

    For JetClass splits, acc is always recomputed from the .pt outputs (restricted
    Tbqq-vs-QCD metric); auc/precision/recall/f1 are taken from the JSON metrics.
    The .pt inference outputs are only read, never modified.
    """
    # 1. Try JSON metrics file (new or old format)
    metrics = load_split_metrics(inference_dir, split)
    if metrics is not None and "acc" in metrics:
        out = {k: float(metrics[k]) for k in _SCALAR_METRICS if k in metrics}
        if "jetclass" in split:
            recomputed = _recompute_acc_from_pt(inference_dir, split)
            if recomputed is not None:
                out["acc"] = recomputed
        return out

    # 2. Fallback: recompute acc from logits + labels in the .pt files
    acc = _recompute_acc_from_pt(inference_dir, split)
    if acc is None:
        return None
    candidates = sorted(inference_dir.glob(f"output_{split}_*.pt"))
    epoch = candidates[0].stem.split("_")[-1]
    _save_metrics_json(inference_dir, split, acc, epoch)
    return {"acc": acc}


# Backward-compatible alias
def load_split_acc(inference_dir: Path, split: str) -> float | None:
    m = load_split_scalar_metrics(inference_dir, split)
    return m["acc"] if m and "acc" in m else None


def _discover_splits(run_dirs: list[Path]) -> set[str]:
    """Discover available splits from output files and metrics JSONs."""
    splits = set()
    for run_dir in run_dirs:
        # From .pt files
        for f in run_dir.glob("output_*.pt"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                split_parts = parts[1:-1]
                splits.add("_".join(split_parts))
        # From metrics JSONs
        for f in run_dir.glob("metrics_*.json"):
            # metrics_{split}_{epoch}.json
            stem = f.stem  # e.g. metrics_val_jetclass_best
            parts = stem.split("_")
            if len(parts) >= 3:
                split_parts = parts[1:-1]
                splits.add("_".join(split_parts))
    return splits


def _aggregate_probe_metrics(run_metrics: list[dict], probe_key: str) -> dict:
    """Aggregate a probe metric (knn or linear_probe) across runs.

    Each entry in run_metrics is the per-run JSON dict for that probe.
    Returns dict with per-metric mean/std across runs.
    """
    if not run_metrics:
        return {}

    if probe_key == "knn":
        # run_metrics[i] = {"1": {"acc_mean": ..., ...}, "5": {...}, ...}
        all_k = set()
        for rm in run_metrics:
            all_k.update(rm.keys())
        aggregated = {}
        for k in sorted(all_k, key=lambda x: int(x)):
            k_metrics = {}
            for metric_name in ("acc", "f1", "precision", "recall"):
                key = f"{metric_name}_mean"
                vals = [rm[k][key] for rm in run_metrics if k in rm and key in rm[k]]
                if vals:
                    k_metrics[f"{metric_name}_mean"] = float(np.mean(vals))
                    k_metrics[f"{metric_name}_std"] = float(np.std(vals))
                    k_metrics[f"{metric_name}_values"] = vals
            aggregated[k] = k_metrics
        return aggregated
    else:
        # linear_probe: run_metrics[i] = {"test_mean": ..., "f1_mean": ..., ...}
        aggregated = {}
        for metric_name in ("train", "test", "f1", "precision", "recall"):
            key = f"{metric_name}_mean"
            vals = [rm[key] for rm in run_metrics if key in rm]
            if vals:
                aggregated[f"{metric_name}_mean"] = float(np.mean(vals))
                aggregated[f"{metric_name}_std"] = float(np.std(vals))
                aggregated[f"{metric_name}_values"] = vals
        return aggregated


def aggregate_experiment(experiment_dir: Path) -> dict:
    """Aggregate results for one experiment across all run-{N}/ subdirectories."""
    run_dirs = sorted(experiment_dir.glob("run-*/inference"))
    if not run_dirs:
        # Also check if inference is directly in the experiment dir (no run subdirs)
        if (experiment_dir / "inference").is_dir():
            run_dirs = [experiment_dir / "inference"]

    if not run_dirs:
        return {"error": f"No run directories found in {experiment_dir}"}

    splits = _discover_splits(run_dirs)

    results = {}
    for split in sorted(splits):
        run_accs = []
        run_knn_metrics = []
        run_lp_metrics = []

        run_metric_vals: dict[str, list[float]] = {m: [] for m in _SCALAR_METRICS}

        for run_dir in tqdm(
            run_dirs, desc=f"  {split}", unit="run", leave=False
        ):
            scalar = load_split_scalar_metrics(run_dir, split)
            if scalar:
                for m in _SCALAR_METRICS:
                    if m in scalar:
                        run_metric_vals[m].append(float(scalar[m]))

            metrics = load_split_metrics(run_dir, split)
            if metrics is not None:
                if "knn" in metrics:
                    run_knn_metrics.append(metrics["knn"])
                if "linear_probe" in metrics:
                    run_lp_metrics.append(metrics["linear_probe"])

        split_results = {}
        run_accs = run_metric_vals.get("acc", [])
        if run_accs:
            split_results.update({
                "n_runs": len(run_accs),
            })
            for m in _SCALAR_METRICS:
                vals = run_metric_vals[m]
                if vals:
                    split_results[m] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": vals,
                    }
        # Keep backward-compatible top-level mean/std/accs for acc
        if run_accs:
            split_results["mean"] = split_results["acc"]["mean"]
            split_results["std"] = split_results["acc"]["std"]
            split_results["accs"] = run_accs

        if run_knn_metrics:
            split_results["knn"] = _aggregate_probe_metrics(run_knn_metrics, "knn")

        if run_lp_metrics:
            split_results["linear_probe"] = _aggregate_probe_metrics(
                run_lp_metrics, "linear_probe"
            )

        if split_results:
            results[split] = split_results

    return results


def print_results(all_results: dict[str, dict]):
    """Pretty-print one table per metric, rows=experiments, cols=splits."""
    # Collect splits and experiment names (skip errors)
    all_splits = sorted({
        split
        for res in all_results.values()
        if "error" not in res
        for split in res
    })
    exp_names = [n for n, r in all_results.items() if "error" not in r]
    error_names = [n for n, r in all_results.items() if "error" in r]

    if not all_splits:
        print("No results found.")
        return

    exp_col = max(len(n) for n in exp_names) + 2
    cell_w = 16  # "0.9108±0.0010"

    def _header():
        h = f"{'Model':<{exp_col}}"
        for s in all_splits:
            h += f"  {s:^{cell_w}}"
        return h

    def _divider(header):
        return "-" * len(header)

    # One table per scalar metric
    for metric in _SCALAR_METRICS:
        hdr = _header()
        print(f"\n── {metric.upper()} ──")
        print(hdr)
        print(_divider(hdr))
        for exp_name in exp_names:
            res = all_results[exp_name]
            row = f"{exp_name:<{exp_col}}"
            for split in all_splits:
                s = res.get(split, {})
                m = s.get(metric)
                if m and "mean" in m:
                    cell = f"{m['mean']:.4f}±{m['std']:.4f}"
                else:
                    cell = "—"
                row += f"  {cell:^{cell_w}}"
            print(row)

    # Probe metrics (knn / linear_probe) if any
    has_probes = any(
        "knn" in s or "linear_probe" in s
        for res in all_results.values()
        if "error" not in res
        for s in res.values()
    )
    if has_probes:
        print("\n── PROBE METRICS ──")
        for exp_name in exp_names:
            res = all_results[exp_name]
            for split in all_splits:
                s = res.get(split, {})
                if "knn" in s:
                    for k, km in s["knn"].items():
                        if "acc_mean" in km:
                            print(f"  {exp_name}/{split}  kNN k={k}: "
                                  f"acc={km['acc_mean']:.4f}±{km['acc_std']:.4f}")
                if "linear_probe" in s:
                    lp = s["linear_probe"]
                    if "test_mean" in lp:
                        print(f"  {exp_name}/{split}  linear_probe: "
                              f"acc={lp['test_mean']:.4f}±{lp['test_std']:.4f}")

    if error_names:
        print("\n── ERRORS ──")
        for n in error_names:
            print(f"  {n}: {all_results[n]['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate finetune results across runs"
    )
    parser.add_argument(
        "--experiment-dir",
        nargs="*",
        type=Path,
        default=None,
        help="Path(s) to experiment directories (each containing run-*/inference/)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Auto-discover all experiment directories under this path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    experiment_dirs = []
    if args.base_dir:
        # Auto-discover: any dir containing run-* subdirs
        for d in sorted(args.base_dir.iterdir()):
            if d.is_dir() and d.name != ".skip" and any(d.glob("run-*")):
                experiment_dirs.append(d)
        # Also include dirs with direct inference/ subdir (single-run)
        for d in sorted(args.base_dir.iterdir()):
            if d.is_dir() and d.name != ".skip" and (d / "inference").is_dir() and d not in experiment_dirs:
                experiment_dirs.append(d)
    if args.experiment_dir:
        experiment_dirs.extend(args.experiment_dir)

    if not experiment_dirs:
        print("No experiment directories specified. Use --experiment-dir or --base-dir.")
        sys.exit(1)

    all_results = {}
    for exp_dir in tqdm(experiment_dirs, desc="Experiments", unit="exp"):
        exp_name = exp_dir.name
        all_results[exp_name] = aggregate_experiment(exp_dir)

    print()
    print_results(all_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

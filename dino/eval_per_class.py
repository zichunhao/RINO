"""Aggregate per-subset metrics and ensemble across runs.

Reads metrics JSONs saved by dino_inference.py (which now include a "subsets"
key with per-subset metrics from eval_subsets config), computes mean±std
across runs, and produces ensemble predictions.

Usage:
    python dino/eval_per_class.py \
        --exp-dir experiments/finetune-jetnet-final/finetune-jn-dino-g6-l2-pbin \
        --num-runs 10

    # Or batch multiple models:
    python dino/eval_per_class.py \
        --exp-dirs experiments/finetune-jetnet-final/finetune-jn-dino-g6-l2-pbin \
                    experiments/finetune-jetnet-final/finetune-jn-rino
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

METRIC_KEYS = ["acc", "precision", "recall", "f1", "auc"]
SUBSET_NAMES = ["Tbqq_vs_QCD", "Tbl_vs_QCD", "top_vs_QCD"]


def agg(values):
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def compute_ensemble_subsets(exp_dir, num_runs, split, epoch):
    """Load all run outputs, average logits, compute per-subset metrics."""
    logits_list = []
    labels = None
    for run_idx in range(1, num_runs + 1):
        path = exp_dir / f"run-{run_idx}" / "inference" / f"output_{split}_{epoch}-0.pt"
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        logits_list.append(data["logits"])
        labels = data["label"]
        del data  # Free the large rep tensor immediately

    if len(logits_list) < 2 or labels is None:
        return None

    ens_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    probs = torch.sigmoid(ens_logits[:, 0]).numpy()
    labels_np = labels.numpy()

    results = {}
    subsets = {
        "Tbqq_vs_QCD": {"positive": [8], "negative": [0]},
        "Tbl_vs_QCD": {"positive": [9], "negative": [0]},
        "top_vs_QCD": {"positive": [8, 9], "negative": [0]},
    }
    for name, sdef in subsets.items():
        pos = np.isin(labels_np, sdef["positive"])
        neg = np.isin(labels_np, sdef["negative"])
        mask = pos | neg
        if mask.sum() == 0:
            continue
        sub_probs = probs[mask]
        sub_labels = pos[mask].astype(int)
        preds = (sub_probs > 0.5).astype(int)
        results[name] = {
            "acc": float(accuracy_score(sub_labels, preds)),
            "precision": float(precision_score(sub_labels, preds, zero_division=0)),
            "recall": float(recall_score(sub_labels, preds, zero_division=0)),
            "f1": float(f1_score(sub_labels, preds, zero_division=0)),
            "auc": float(roc_auc_score(sub_labels, sub_probs)),
        }
    return results


def process_model(exp_dir, num_runs, split="test_jetclass", epoch="best"):
    """Aggregate one model's per-subset metrics across runs + ensemble."""
    exp_dir = Path(exp_dir)

    # Collect per-run subset metrics from JSONs
    per_run = []  # list of {subset_name: {metric: value}}
    for run_idx in range(1, num_runs + 1):
        json_path = exp_dir / f"run-{run_idx}" / "inference" / f"metrics_{split}_{epoch}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            data = json.load(f)

        if "subsets" in data:
            per_run.append(data["subsets"])
        elif "classification" in data:
            # Fallback: old format without subsets — use overall as top_vs_QCD
            per_run.append({"top_vs_QCD": data["classification"]})

    if not per_run:
        print(f"  {exp_dir.name}: no metrics found")
        return None

    # Aggregate mean±std per subset per metric
    aggregated = {}
    for subset_name in SUBSET_NAMES:
        runs_with_subset = [r[subset_name] for r in per_run if subset_name in r]
        if not runs_with_subset:
            continue
        aggregated[subset_name] = {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in runs_with_subset if k in r]
            if vals:
                mean, std = agg(vals)
                aggregated[subset_name][k] = {"mean": mean, "std": std}

    # Ensemble
    ensemble = compute_ensemble_subsets(exp_dir, num_runs, split, epoch)

    return {
        "model": exp_dir.name,
        "num_runs": len(per_run),
        "subsets": aggregated,
        "ensemble": ensemble,
    }


def print_table(results_list):
    """Print a comparison table across models."""
    print(f"\n{'=' * 100}")
    print(f"{'Model':<45s} {'Subset':<16s} {'ACC':>12s} {'AUC':>12s} {'Recall':>12s} {'Prec':>12s}")
    print(f"{'-' * 100}")

    for res in results_list:
        model = res["model"]
        first = True
        for subset_name in SUBSET_NAMES:
            if subset_name not in res["subsets"]:
                continue
            s = res["subsets"][subset_name]
            label = model if first else ""
            first = False
            row = f"{label:<45s} {subset_name:<16s}"
            for k in ["acc", "auc", "recall", "precision"]:
                if k in s:
                    row += f" {s[k]['mean']:.3f}±{s[k]['std']:.3f}"
                else:
                    row += f" {'—':>11s}"
            print(row)

        if res.get("ensemble"):
            for subset_name in SUBSET_NAMES:
                if subset_name not in res["ensemble"]:
                    continue
                e = res["ensemble"][subset_name]
                row = f"{'  (ensemble)':<45s} {subset_name:<16s}"
                for k in ["acc", "auc", "recall", "precision"]:
                    row += f" {e.get(k, 0):>11.4f} "
                print(row)
        print()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-dir", type=str, help="Single experiment directory")
    group.add_argument("--exp-dirs", type=str, nargs="+", help="Multiple experiment directories")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--split", type=str, default="test_jetclass")
    parser.add_argument("--epoch", type=str, default="best")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    dirs = [args.exp_dir] if args.exp_dir else args.exp_dirs

    all_results = []
    for d in dirs:
        print(f"\nProcessing: {Path(d).name}")
        res = process_model(d, args.num_runs, args.split, args.epoch)
        if res:
            all_results.append(res)

    if all_results:
        print_table(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Embedding diagnostics for transfer robustness.

Computes metrics on saved inference outputs to estimate how well a backbone
will transfer across domains (e.g., JetNet → JetClass) without running the
full finetune pipeline.

Metrics:
  - uniformity:  log(mean(exp(-2||z_i - z_j||^2)))  — lower = more uniform
  - alignment:   mean ||z_i - z_j||^2 for same-class pairs — lower = tighter
  - knn_acc:     k-nearest-neighbor accuracy on the embeddings
  - cka:         centered kernel alignment between two sets of embeddings

Usage:
    # Single experiment, single split
    python dino/embedding_diagnostics.py \
        --experiment-dir experiments/dino-g6-l2-pbin \
        --splits test_jetclass test_jetnet \
        --epoch 199

    # Compare multiple experiments
    python dino/embedding_diagnostics.py \
        --experiment-dir experiments/dino-g6-l2-pbin \
                        experiments/dino-ibot-g6l2-pbin \
        --splits test_jetclass test_jetnet \
        --epoch 199

    # All experiments matching a pattern
    python dino/embedding_diagnostics.py \
        --base-dir experiments \
        --pattern "dino-*" \
        --splits test_jetclass test_jetnet \
        --epoch 199

    # CKA between two splits (cross-domain similarity)
    python dino/embedding_diagnostics.py \
        --experiment-dir experiments/dino-g6-l2-pbin \
        --cka-splits test_jetnet test_jetclass \
        --epoch 199

    # Save to JSON
    python dino/embedding_diagnostics.py \
        --experiment-dir experiments/dino-g6-l2-pbin \
        --splits test_jetclass test_jetnet \
        --epoch 199 -o diagnostics.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------------ #
# Core metrics                                                        #
# ------------------------------------------------------------------ #


def uniformity(z: torch.Tensor, t: float = 2.0, max_pairs: int = 50_000) -> float:
    """Compute uniformity: log(mean(exp(-t * ||z_i - z_j||^2))).

    Lower (more negative) = more uniform distribution on the hypersphere.

    Args:
        z: Embeddings, shape (N, D). L2-normalized internally.
        t: Temperature (default 2.0, standard choice).
        max_pairs: Subsample to this many points if N is large (O(N^2) cost).
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    if z.shape[0] > max_pairs:
        idx = torch.randperm(z.shape[0])[:max_pairs]
        z = z[idx]
    # Pairwise squared distances
    sq_dists = torch.cdist(z, z, p=2).pow(2)
    # Exclude diagonal
    n = z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    sq_dists = sq_dists[mask]
    return torch.log(torch.exp(-t * sq_dists).mean()).item()


def alignment(
    z: torch.Tensor,
    labels: torch.Tensor,
    max_pairs_per_class: int = 10_000,
) -> float:
    """Compute alignment: mean ||z_i - z_j||^2 for same-class pairs.

    Lower = tighter same-class clusters (better alignment).

    Args:
        z: Embeddings, shape (N, D). L2-normalized internally.
        labels: Integer labels, shape (N,).
        max_pairs_per_class: Subsample within each class to bound cost.
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    total_dist = 0.0
    total_pairs = 0
    for label in labels.unique():
        z_c = z[labels == label]
        if z_c.shape[0] < 2:
            continue
        if z_c.shape[0] > max_pairs_per_class:
            idx = torch.randperm(z_c.shape[0])[:max_pairs_per_class]
            z_c = z_c[idx]
        sq_dists = torch.cdist(z_c, z_c, p=2).pow(2)
        n = z_c.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=z_c.device)
        total_dist += sq_dists[mask].sum().item()
        total_pairs += mask.sum().item()
    if total_pairs == 0:
        return float("nan")
    return total_dist / total_pairs


def knn_accuracy(
    z: torch.Tensor,
    labels: torch.Tensor,
    k: int = 20,
    max_samples: int = 50_000,
) -> float:
    """Compute k-nearest-neighbor accuracy.

    Args:
        z: Embeddings, shape (N, D).
        labels: Integer labels, shape (N,).
        k: Number of neighbors.
        max_samples: Subsample if N is large.
    """
    z = torch.nn.functional.normalize(z, dim=-1)
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0])[:max_samples]
        z = z[idx]
        labels = labels[idx]

    # Cosine similarity via dot product on L2-normed vectors
    sims = z @ z.T
    # Zero out self-similarity
    sims.fill_diagonal_(-float("inf"))
    # Top-k neighbors
    _, topk_idx = sims.topk(k, dim=-1)
    topk_labels = labels[topk_idx]
    # Majority vote
    preds = topk_labels.mode(dim=-1).values
    return (preds == labels).float().mean().item()


def linear_cka(
    z1: torch.Tensor,
    z2: torch.Tensor,
    max_samples: int = 50_000,
) -> float:
    """Compute linear CKA (Centered Kernel Alignment) between two sets of embeddings.

    Measures representational similarity: higher = more similar representations.
    z1 and z2 must correspond to the same samples (same ordering).

    Args:
        z1: Embeddings from domain/model 1, shape (N, D1).
        z2: Embeddings from domain/model 2, shape (N, D2).
        max_samples: Subsample if N is large.
    """
    assert z1.shape[0] == z2.shape[0], (
        f"CKA requires same number of samples, got {z1.shape[0]} and {z2.shape[0]}"
    )
    if z1.shape[0] > max_samples:
        idx = torch.randperm(z1.shape[0])[:max_samples]
        z1 = z1[idx]
        z2 = z2[idx]

    # Center
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    # HSIC with linear kernel: HSIC(K, L) = ||X^T Y||_F^2 / (n-1)^2
    # CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    cross = torch.linalg.norm(z1.T @ z2, "fro").pow(2)
    self1 = torch.linalg.norm(z1.T @ z1, "fro").pow(2)
    self2 = torch.linalg.norm(z2.T @ z2, "fro").pow(2)

    denom = torch.sqrt(self1 * self2)
    if denom < 1e-12:
        return float("nan")
    return (cross / denom).item()


# ------------------------------------------------------------------ #
# I/O helpers                                                         #
# ------------------------------------------------------------------ #


def load_inference_output(
    inference_dir: Path, split: str, epoch: str | int
) -> dict[str, torch.Tensor] | None:
    """Load inference output .pt file(s) for a given split and epoch.

    Handles both single-file (output_SPLIT_EPOCH.pt) and multi-file
    (output_SPLIT_EPOCH-0.pt, ...) formats.
    """
    # Single file
    single = inference_dir / f"output_{split}_{epoch}.pt"
    if single.exists():
        return torch.load(single, map_location="cpu", weights_only=False)

    # Multi-file (sharded)
    shards = sorted(inference_dir.glob(f"output_{split}_{epoch}-*.pt"))
    if not shards:
        return None

    combined = {}
    for shard_path in shards:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for key, val in shard.items():
            if isinstance(val, torch.Tensor):
                combined.setdefault(key, []).append(val)
            else:
                combined[key] = val  # scalars — keep last

    for key, val in combined.items():
        if isinstance(val, list):
            combined[key] = torch.cat(val, dim=0)
    return combined


def find_experiments(
    base_dir: Path | None,
    experiment_dirs: list[Path] | None,
    pattern: str = "*",
) -> list[Path]:
    """Resolve experiment directories from CLI args."""
    dirs = []
    if experiment_dirs:
        dirs.extend(experiment_dirs)
    if base_dir:
        dirs.extend(sorted(base_dir.glob(pattern)))
    # Filter to dirs that have an inference/ subdirectory
    return [d for d in dirs if d.is_dir() and (d / "inference").is_dir()]


def remap_labels_binary(labels: torch.Tensor, split: str) -> torch.Tensor:
    """Remap labels to binary (0=QCD, 1=signal) matching finetune convention."""
    if "jetclass" in split:
        # Remove leptonic top (label 9), remap: QCD=0→0, else→1
        mask = labels != 9
        labels = labels[mask]
        labels = (labels != 0).long()
        return labels
    return labels


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #


def compute_diagnostics(
    inference_dir: Path,
    split: str,
    epoch: str | int,
    k: int = 20,
) -> dict | None:
    """Compute all per-split diagnostics for one experiment."""
    data = load_inference_output(inference_dir, split, epoch)
    if data is None:
        return None

    rep = data.get("rep")
    if rep is None:
        return None

    labels_raw = data.get("label")
    result = {"n_samples": rep.shape[0], "rep_dim": rep.shape[1]}

    # Uniformity (always computable)
    result["uniformity"] = uniformity(rep)

    # Alignment and kNN (need labels)
    if labels_raw is not None:
        labels = remap_labels_binary(labels_raw, split)
        # If label remapping filtered samples, filter rep too
        if "jetclass" in split:
            mask = labels_raw != 9
            rep_filtered = rep[mask]
        else:
            rep_filtered = rep
            labels = labels_raw

        if rep_filtered.shape[0] == labels.shape[0]:
            result["alignment"] = alignment(rep_filtered, labels)
            result["knn_acc"] = knn_accuracy(rep_filtered, labels, k=k)

    return result


def compute_cka_cross_domain(
    inference_dir: Path,
    split1: str,
    split2: str,
    epoch: str | int,
) -> float | None:
    """Compute CKA between embeddings from two splits.

    Since splits have different samples, we subsample to the same size
    and compute CKA. This measures how similar the embedding *geometry*
    is across domains, not sample correspondence.

    For a fair comparison, we sample equal numbers per class from each split.
    """
    data1 = load_inference_output(inference_dir, split1, epoch)
    data2 = load_inference_output(inference_dir, split2, epoch)
    if data1 is None or data2 is None:
        return None

    rep1, rep2 = data1.get("rep"), data2.get("rep")
    if rep1 is None or rep2 is None:
        return None

    # Subsample to same size
    n = min(rep1.shape[0], rep2.shape[0], 50_000)
    idx1 = torch.randperm(rep1.shape[0])[:n]
    idx2 = torch.randperm(rep2.shape[0])[:n]
    return linear_cka(rep1[idx1], rep2[idx2])


def main():
    parser = argparse.ArgumentParser(
        description="Embedding diagnostics for transfer robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        nargs="*",
        type=Path,
        help="One or more experiment directories (each should have inference/ subdir)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory to search for experiments (used with --pattern)",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for experiment names under --base-dir (default: '*')",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_jetnet", "test_jetclass"],
        help="Splits to compute diagnostics on (default: test_jetnet test_jetclass)",
    )
    parser.add_argument(
        "--epoch",
        default="best",
        help="Epoch to load (default: 'best'). Use 'best', an integer, or 'LAST'.",
    )
    parser.add_argument(
        "--cka-splits",
        nargs=2,
        metavar=("SPLIT1", "SPLIT2"),
        help="Compute cross-domain CKA between two splits (e.g., test_jetnet test_jetclass)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=20,
        help="Number of neighbors for kNN (default: 20)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    experiments = find_experiments(args.base_dir, args.experiment_dir, args.pattern)
    if not experiments:
        print("No experiments found. Check --experiment-dir or --base-dir + --pattern.")
        sys.exit(1)

    all_results = {}

    for exp_dir in experiments:
        exp_name = exp_dir.name
        inference_dir = exp_dir / "inference"
        exp_results = {}

        # Per-split diagnostics
        for split in args.splits:
            diag = compute_diagnostics(inference_dir, split, args.epoch, k=args.k)
            if diag is not None:
                exp_results[split] = diag

        # Cross-domain CKA
        if args.cka_splits:
            s1, s2 = args.cka_splits
            cka_val = compute_cka_cross_domain(inference_dir, s1, s2, args.epoch)
            if cka_val is not None:
                exp_results[f"cka_{s1}_vs_{s2}"] = cka_val

        if exp_results:
            all_results[exp_name] = exp_results

    if not all_results:
        print("No inference outputs found for the specified experiments/splits/epoch.")
        sys.exit(1)

    # Print summary table
    _print_summary(all_results, args.splits, args.cka_splits)

    # Save JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def _print_summary(
    all_results: dict,
    splits: list[str],
    cka_splits: list[str] | None,
):
    """Print a compact comparison table."""
    # Collect all metric keys
    metric_keys = []
    for split in splits:
        for suffix in ["uniformity", "alignment", "knn_acc"]:
            metric_keys.append((split, suffix))

    # Header
    col_headers = []
    for split, metric in metric_keys:
        short_split = split.replace("test_", "").replace("train_", "tr_")
        col_headers.append(f"{short_split}/{metric}")
    if cka_splits:
        s1_short = cka_splits[0].replace("test_", "")
        s2_short = cka_splits[1].replace("test_", "")
        col_headers.append(f"cka({s1_short},{s2_short})")

    # Determine column widths
    name_width = max(len(name) for name in all_results) + 2
    col_width = max(max(len(h) for h in col_headers) + 2, 14)

    # Print header
    header = f"{'Experiment':<{name_width}}"
    for h in col_headers:
        header += f"{h:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Print rows
    for exp_name, exp_results in sorted(all_results.items()):
        row = f"{exp_name:<{name_width}}"
        for split, metric in metric_keys:
            val = exp_results.get(split, {}).get(metric)
            if val is None:
                row += f"{'—':>{col_width}}"
            elif metric == "knn_acc":
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{val:>{col_width}.4f}"
        if cka_splits:
            cka_key = f"cka_{cka_splits[0]}_vs_{cka_splits[1]}"
            cka_val = exp_results.get(cka_key)
            if cka_val is None:
                row += f"{'—':>{col_width}}"
            else:
                row += f"{cka_val:>{col_width}.4f}"
        print(row)

    # Print cross-domain gap if both splits have kNN
    if len(splits) >= 2:
        print()
        s1, s2 = splits[0], splits[1]
        s1_short = s1.replace("test_", "").replace("train_", "tr_")
        s2_short = s2.replace("test_", "").replace("train_", "tr_")
        print(f"kNN gap ({s1_short} - {s2_short}):")
        for exp_name, exp_results in sorted(all_results.items()):
            acc1 = exp_results.get(s1, {}).get("knn_acc")
            acc2 = exp_results.get(s2, {}).get("knn_acc")
            if acc1 is not None and acc2 is not None:
                gap = acc1 - acc2
                print(f"  {exp_name}: {gap:+.4f}  ({s1_short}={acc1:.4f}, {s2_short}={acc2:.4f})")


if __name__ == "__main__":
    main()

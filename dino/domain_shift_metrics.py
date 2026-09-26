"""
Domain Shift Metrics: Wasserstein-1, MMD, and Proxy A-distance.

Measures the domain gap between JetNet and JetClass representations
in the finetuned embedding space. Computes:
- Per-dimension Wasserstein-1 distance (W1), averaged over embedding dimensions
- Maximum Mean Discrepancy with RBF kernel exp(-gamma ||x - y||^2), gamma = 1/d (MMD)
- Proxy A-distance (linear domain classifier accuracy)

Each metric is computed separately for QCD (background) and signal (top) jets.

Sampling protocol: for each domain (JetNet, JetClass) and each class, a seeded
random subset of `--n-per-domain` jets (default 5,000) is drawn once, and all
metrics are computed on these same subsets. Classes with fewer jets use all of
them.

Usage:
  python dino/domain_shift_metrics.py \
    --jn-reps <path_to_jetnet_reps.pt> \
    --jc-reps <path_to_jetclass_reps.pt> \
    --output results/domain_shift.json
"""

import torch
import numpy as np
import argparse
import json
import os
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def subsample(X, n, rng):
    """Return a random subset of at most `n` rows of X (all rows if len(X) <= n)."""
    if len(X) <= n:
        return X
    return X[rng.choice(len(X), n, replace=False)]


def compute_w1_per_dim(X, Y):
    """Per-dimension Wasserstein-1 distance between two distributions."""
    d = X.shape[1]
    return np.array([wasserstein_distance(X[:, i], Y[:, i]) for i in range(d)])


def mmd_rbf(X, Y, gamma=None):
    """Maximum Mean Discrepancy with RBF (Gaussian) kernel; gamma defaults to 1/d."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    XX = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    YY = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    XY = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def proxy_a_distance(X_source, X_target, rng):
    """Proxy A-distance via linear domain classifier (Ben-David et al., 2010).

    Both domains are balanced to the size of the smaller one by random sampling.
    """
    n = min(len(X_source), len(X_target))
    X = np.vstack([subsample(X_source, n, rng), subsample(X_target, n, rng)])
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    lr = LogisticRegression(max_iter=500, C=1.0)
    cv_acc = cross_val_score(lr, X, y, cv=5, scoring="accuracy").mean()
    return 2 * (cv_acc - 0.5), cv_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jn-reps", required=True, help="Path to JetNet inference output .pt")
    parser.add_argument("--jc-reps", required=True, help="Path to JetClass inference output .pt")
    parser.add_argument("--model-name", default="unknown", help="Model name for output")
    parser.add_argument(
        "--n-per-domain",
        type=int,
        default=5000,
        help="Jets drawn per domain and per class for all metrics (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--output", "-o", default="domain_shift_results.json")
    # Label conventions
    parser.add_argument("--jn-signal-label", type=int, default=1, help="JetNet signal label (default: 1=top)")
    parser.add_argument("--jn-background-label", type=int, default=0, help="JetNet background label (default: 0=g+q)")
    parser.add_argument("--jc-signal-label", type=int, default=8, help="JetClass signal label (default: 8=Tbqq)")
    parser.add_argument("--jc-background-label", type=int, default=0, help="JetClass background label (default: 0=QCD)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load representations
    print(f"Loading JetNet reps: {args.jn_reps}")
    jn = torch.load(args.jn_reps, map_location="cpu", weights_only=False)
    jn_rep = jn["rep"].float().numpy()
    jn_label = jn["label"].numpy().astype(int)

    print(f"Loading JetClass reps: {args.jc_reps}")
    jc = torch.load(args.jc_reps, map_location="cpu", weights_only=False)
    jc_rep = jc["rep"].float().numpy()
    jc_label = jc["label"].numpy().astype(int)

    # Split by class, then draw the per-domain, per-class subsets
    n = args.n_per_domain
    jn_bkg = subsample(jn_rep[jn_label == args.jn_background_label], n, rng)
    jn_sig = subsample(jn_rep[jn_label == args.jn_signal_label], n, rng)
    jc_bkg = subsample(jc_rep[jc_label == args.jc_background_label], n, rng)
    jc_sig = subsample(jc_rep[jc_label == args.jc_signal_label], n, rng)

    print(f"JN: {len(jn_bkg)} bkg, {len(jn_sig)} signal")
    print(f"JC: {len(jc_bkg)} bkg, {len(jc_sig)} signal")

    # === Wasserstein-1 (per-dimension) ===
    print("\nComputing Wasserstein-1...")
    w1_bkg = compute_w1_per_dim(jn_bkg, jc_bkg)
    w1_sig = compute_w1_per_dim(jn_sig, jc_sig)
    w1_class_jn = compute_w1_per_dim(jn_bkg, jn_sig)
    w1_class_jc = compute_w1_per_dim(jc_bkg, jc_sig)

    print(f"  W1 background (domain shift): mean={w1_bkg.mean():.4f}, max={w1_bkg.max():.4f}")
    print(f"  W1 signal (domain shift):     mean={w1_sig.mean():.4f}, max={w1_sig.max():.4f}")
    print(f"  W1 class separation (JN):     mean={w1_class_jn.mean():.4f}")
    print(f"  W1 class separation (JC):     mean={w1_class_jc.mean():.4f}")
    print(f"  Domain/class ratio (bkg):     {w1_bkg.mean() / w1_class_jn.mean():.4f}")

    # === MMD ===
    print("\nComputing MMD...")
    mmd_bkg = mmd_rbf(jn_bkg, jc_bkg)
    mmd_sig = mmd_rbf(jn_sig, jc_sig)
    mmd_class_jn = mmd_rbf(jn_bkg, jn_sig)
    mmd_class_jc = mmd_rbf(jc_bkg, jc_sig)

    print(f"  MMD background (domain shift): {mmd_bkg:.6f}")
    print(f"  MMD signal (domain shift):     {mmd_sig:.6f}")
    print(f"  MMD class separation (JN):     {mmd_class_jn:.6f}")
    print(f"  MMD class separation (JC):     {mmd_class_jc:.6f}")

    # === Proxy A-distance ===
    print("\nComputing Proxy A-distance...")
    pad_bkg, acc_bkg = proxy_a_distance(jn_bkg, jc_bkg, rng)
    pad_sig, acc_sig = proxy_a_distance(jn_sig, jc_sig, rng)

    print(f"  Proxy A-dist (bkg): {pad_bkg:.4f} (classifier ACC={acc_bkg:.4f})")
    print(f"  Proxy A-dist (sig): {pad_sig:.4f} (classifier ACC={acc_sig:.4f})")

    # Save results
    results = {
        "model": args.model_name,
        "n_per_domain": n,
        "seed": args.seed,
        "n_jn_bkg": len(jn_bkg),
        "n_jn_sig": len(jn_sig),
        "n_jc_bkg": len(jc_bkg),
        "n_jc_sig": len(jc_sig),
        "w1_bkg_mean": float(w1_bkg.mean()),
        "w1_bkg_max": float(w1_bkg.max()),
        "w1_sig_mean": float(w1_sig.mean()),
        "w1_class_jn_mean": float(w1_class_jn.mean()),
        "w1_class_jc_mean": float(w1_class_jc.mean()),
        "domain_class_ratio_bkg": float(w1_bkg.mean() / w1_class_jn.mean()),
        "mmd_bkg": mmd_bkg,
        "mmd_sig": mmd_sig,
        "mmd_class_jn": mmd_class_jn,
        "mmd_class_jc": mmd_class_jc,
        "proxy_a_bkg": pad_bkg,
        "proxy_a_sig": pad_sig,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

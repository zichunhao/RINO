"""kNN + frozen linear probe on saved backbone embeddings for Tbqq vs QCD.

Usage:
    python dino/probe_tbqq.py \
        --train-output experiments/finetune-jetnet-final/<model>/run-1/inference/output_test_jetnet_best-0.pt \
        --test-output experiments/finetune-jetnet-final/<model>/run-1/inference/output_test_jetclass_best-0.pt \
        --k 20

Train on JetNet test embeddings (in-distribution), evaluate on JetClass test (OOD).
Filters to Tbqq (label 8) and QCD (label 0) only for evaluation.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


def load_and_filter(path, labels_positive=None, labels_negative=None):
    """Load output file and optionally filter to specific labels."""
    d = torch.load(path, map_location="cpu")
    rep = d["rep"].numpy()
    label = d["label"].numpy().astype(int)

    if labels_positive is not None and labels_negative is not None:
        pos_mask = np.isin(label, labels_positive)
        neg_mask = np.isin(label, labels_negative)
        keep = pos_mask | neg_mask
        rep = rep[keep]
        binary_label = pos_mask[keep].astype(int)
        return rep, binary_label
    return rep, label


def run_knn(X_train, y_train, X_test, y_test, k=20):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    y_prob = knn.predict_proba(X_test_s)[:, 1]

    return {
        "acc": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def run_linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr.fit(X_train_s, y_train)
    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)[:, 1]

    return {
        "acc": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True,
                        help="Path to JetNet test output (used as train for probe)")
    parser.add_argument("--test-output", type=Path, required=True,
                        help="Path to JetClass test output (OOD evaluation)")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results JSON here")
    parser.add_argument("--positive-labels", type=int, nargs="+", default=[8],
                        help="Positive class labels for test set (default: [8] = Tbqq)")
    parser.add_argument("--negative-labels", type=int, nargs="+", default=[0],
                        help="Negative class labels for test set (default: [0] = QCD)")
    parser.add_argument("--task-name", type=str, default=None,
                        help="Name for this probe task (for display)")
    args = parser.parse_args()

    task_name = args.task_name or f"{args.positive_labels}_vs_{args.negative_labels}"

    # JetNet test = in-distribution "training" data for probe
    # Binary: top (label 1 in JetNet) vs QCD (label 0 in JetNet)
    print(f"Loading train: {args.train_output}")
    X_train, y_train = load_and_filter(args.train_output)
    # JetNet labels: 0=QCD, 1=top — already binary
    print(f"  Train: {X_train.shape}, labels: {np.unique(y_train, return_counts=True)}")

    # JetClass test: filter to specified labels
    print(f"Loading test: {args.test_output}")
    X_test, y_test = load_and_filter(
        args.test_output, labels_positive=args.positive_labels, labels_negative=args.negative_labels
    )
    print(f"  Test ({task_name}): {X_test.shape}, labels: {np.unique(y_test, return_counts=True)}")

    # kNN
    print(f"\nRunning kNN (k={args.k})...")
    knn_results = run_knn(X_train, y_train, X_test, y_test, k=args.k)
    print(f"  kNN ACC: {knn_results['acc']:.4f}, AUC: {knn_results['auc']:.4f}, "
          f"Recall: {knn_results['recall']:.4f}, Prec: {knn_results['precision']:.4f}")

    # Linear probe
    print(f"\nRunning linear probe...")
    lp_results = run_linear_probe(X_train, y_train, X_test, y_test)
    print(f"  LP  ACC: {lp_results['acc']:.4f}, AUC: {lp_results['auc']:.4f}, "
          f"Recall: {lp_results['recall']:.4f}, Prec: {lp_results['precision']:.4f}")

    results = {"knn": knn_results, "linear_probe": lp_results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()

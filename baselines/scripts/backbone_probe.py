"""Backbone-only kNN + linear probe evaluation.

Loads a pretrained backbone (backbone.pt or DINO checkpoint), runs forward pass
on JetClass test data, and evaluates with kNN (k=20) and linear probe on
Tbqq vs QCD (labels [0, 8]).

Usage:
    python dino/backbone_probe.py \
        --backbone-path experiments/mpmv1/.../backbone.pt \
        --dataloader-config configs/dataloaders/jetclass-raw/kinematics.yaml \
        --pooling mean \
        --output experiments/mpmv1/.../probe_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root and dino/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # baselines/scripts -> baselines -> root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dino"))

from models.jet_transformer_encoder import JetTransformerEncoder
from utils.producers.dataloader import get_dataloader_and_config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("BackboneProbe")

BASELINE_JTE_CONFIG = dict(
    part_dim=7, d_model=256, nhead=16, num_layers=8,
    pooling="mean", activation="SwiGLU", norm="RMSNorm",
    layer_scale_init=0.01, num_registers=4,
    apply_final_norm=True, apply_embedding_norm=True,
)


@torch.no_grad()
def extract_embeddings(model, dataloader, device, label_filter=None):
    """Run backbone forward pass and collect pooled embeddings + labels."""
    model.eval()
    reps, labels = [], []
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(device)
        mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)
        label = batch["aux"]["label"]

        out = model(particles=particles, mask=mask)
        rep = out[0] if isinstance(out, tuple) else out
        reps.append(rep.cpu())
        labels.append(torch.tensor(label))

    reps = torch.cat(reps, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy().astype(int)

    if label_filter is not None:
        keep = np.isin(labels, label_filter)
        reps = reps[keep]
        labels = labels[keep]
        # Binarize: first label group = 0, second = 1
        binary = np.zeros_like(labels)
        for lbl in label_filter[1:]:
            binary[labels == lbl] = 1
        # Actually for [0, 8]: 0=QCD(negative), 8=Tbqq(positive)
        binary = (labels == label_filter[-1]).astype(int) if len(label_filter) == 2 else binary
        labels = binary

    return reps, labels


def run_knn(X, y, k=20, n_experiments=5, sample_size=200000):
    """kNN evaluation with subsampling."""
    results = []
    for i in range(n_experiments):
        rng = np.random.default_rng(42 + i)
        n = min(sample_size, len(X))
        idx = rng.choice(len(X), n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # 80/20 split
        split = int(0.8 * n)
        perm = rng.permutation(n)
        train_idx, test_idx = perm[:split], perm[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[train_idx])
        X_test = scaler.transform(X_sub[test_idx])

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(X_train, y_sub[train_idx])
        y_pred = knn.predict(X_test)
        results.append(accuracy_score(y_sub[test_idx], y_pred))

    return {"acc_mean": np.mean(results), "acc_std": np.std(results), "acc_values": results}


def run_linear_probe(X, y, n_experiments=5, sample_size=200000):
    """Linear probe evaluation with subsampling."""
    train_accs, test_accs, test_aucs = [], [], []
    for i in range(n_experiments):
        rng = np.random.default_rng(42 + i)
        n = min(sample_size, len(X))
        idx = rng.choice(len(X), n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        split = int(0.8 * n)
        perm = rng.permutation(n)
        train_idx, test_idx = perm[:split], perm[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[train_idx])
        X_test = scaler.transform(X_sub[test_idx])

        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
        lr.fit(X_train, y_sub[train_idx])

        train_accs.append(lr.score(X_train, y_sub[train_idx]))
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]
        test_accs.append(accuracy_score(y_sub[test_idx], y_pred))
        test_aucs.append(roc_auc_score(y_sub[test_idx], y_prob))

    return {
        "train_mean": np.mean(train_accs), "train_std": np.std(train_accs),
        "test_mean": np.mean(test_accs), "test_std": np.std(test_accs),
        "test_auc_mean": np.mean(test_aucs), "test_auc_std": np.std(test_aucs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--dataloader-config", type=Path, required=True)
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls_token", "last_token"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--n-experiments", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device: {device}")

    # Build backbone
    jte_config = dict(BASELINE_JTE_CONFIG)
    jte_config["pooling"] = args.pooling
    model = JetTransformerEncoder(**jte_config).to(device)

    # Load weights
    LOGGER.info(f"Loading backbone from {args.backbone_path}")
    ckpt = torch.load(args.backbone_path, map_location=device, weights_only=False)
    if "backbone" in ckpt:
        sd = ckpt["backbone"]
    elif "teacher" in ckpt:
        sd = ckpt["teacher"]
    else:
        sd = ckpt

    # Filter to matching keys, allow missing (e.g. particle_embedding for OmniJet)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        LOGGER.warning(f"Missing {len(missing)} keys (random init): {sorted(missing)[:5]}")
    model.load_state_dict(filtered, strict=False)
    LOGGER.info(f"Loaded {len(filtered)}/{len(model_keys)} keys")

    # Build dataloader
    dummy_config = {
        "training": {
            "dataloader": {
                "test_jetclass": {
                    "config": str(args.dataloader_config),
                    "num_workers": 4,
                    "cached": True,
                    "preprocessed": False,
                    "kwargs": {"batch_size": 500},
                }
            }
        }
    }
    dataloader, _ = get_dataloader_and_config(
        config=dummy_config, split="test_jetclass", mode="training"
    )

    # Extract embeddings (filter to Tbqq=8, QCD=0)
    LOGGER.info("Extracting embeddings (Tbqq vs QCD)...")
    reps, labels = extract_embeddings(model, dataloader, device, label_filter=[0, 8])
    LOGGER.info(f"Embeddings: {reps.shape}, labels: {np.unique(labels, return_counts=True)}")

    # kNN
    LOGGER.info(f"Running kNN (k={args.k})...")
    knn_results = run_knn(reps, labels, k=args.k, n_experiments=args.n_experiments, sample_size=args.sample_size)
    LOGGER.info(f"kNN (k={args.k}) | acc: {knn_results['acc_mean']:.4f} ± {knn_results['acc_std']:.4f}")

    # Linear probe
    LOGGER.info("Running linear probe...")
    lp_results = run_linear_probe(reps, labels, n_experiments=args.n_experiments, sample_size=args.sample_size)
    LOGGER.info(f"Linear probe | test acc: {lp_results['test_mean']:.4f} ± {lp_results['test_std']:.4f}")

    results = {"knn": knn_results, "linear_probe": lp_results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

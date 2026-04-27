"""Linear regression probe for jet mass from backbone embeddings.

Tests whether the representation encodes jet mass information.
Probes both ungroomed mass (from 4-momentum) and soft-drop mass.

Usage:
    python dino/probe_mass.py \
        --output-file experiments/.../inference_val_jetclass/output_val_jetclass_best-0.pt \
        --root-dir /dev/shm/JetClass/raw/val_5M
"""

import argparse
import numpy as np
import torch
import uproot
import glob
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


def load_jet_features(root_dir, patterns):
    """Load jet mass (ungroomed + SD) from ROOT files."""
    sd_masses, masses = [], []
    for pattern in patterns:
        files = sorted(glob.glob(str(Path(root_dir) / pattern)))
        for f in files:
            tree = uproot.open(f)["tree"]
            sd_masses.append(tree["jet_sdmass"].array(library="np"))
            E = tree["jet_energy"].array(library="np")
            pt = tree["jet_pt"].array(library="np")
            eta = tree["jet_eta"].array(library="np")
            phi = tree["jet_phi"].array(library="np")
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            m2 = np.maximum(E**2 - px**2 - py**2 - pz**2, 0)
            masses.append(np.sqrt(m2))
    return np.concatenate(masses), np.concatenate(sd_masses)


def run_regression(X_train, y_train, X_test, y_test, name=""):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_train)
    y_pred = reg.predict(X_te)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(np.exp(y_test) - 1, np.exp(y_pred) - 1)
    print(f"  {name}: R²={r2:.4f}, MAE={mae:.1f} GeV")
    return r2, mae, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--root-patterns", nargs="+", default=[
        "HToBB_*.root", "HToCC_*.root", "HToGG_*.root", "HToWW4Q_*.root",
        "TTBar_*.root", "ZJetsToNuNu_*.root", "WToQQ_*.root", "ZToQQ_*.root",
    ])
    parser.add_argument("--max-jets", type=int, default=200000)
    args = parser.parse_args()

    print(f"Loading embeddings: {args.output_file}")
    data = torch.load(args.output_file, map_location="cpu", weights_only=False)
    rep = data["rep"].numpy()
    labels = data["label"].numpy().astype(int)
    del data

    print(f"Loading masses from: {args.root_dir}")
    masses, sd_masses = load_jet_features(args.root_dir, args.root_patterns)
    assert len(masses) == len(rep), f"Mismatch: {len(masses)} vs {len(rep)}"

    log_mass = np.log(masses + 1)
    log_sd_mass = np.log(sd_masses + 1)

    # Subsample
    n = min(args.max_jets, len(rep))
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(rep))[:n]
    X, y_mass, y_sd, y_labels = rep[idx], log_mass[idx], log_sd_mass[idx], labels[idx]

    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]

    print(f"\n=== Overall (n_train={n_train}, n_test={n-n_train}) ===")
    r2_m, mae_m, pred_mass = run_regression(X_train, y_mass[:n_train], X_test, y_mass[n_train:], "Ungroomed mass")
    r2_sd, mae_sd, pred_sd = run_regression(X_train, y_sd[:n_train], X_test, y_sd[n_train:], "Soft-drop mass")

    # Per-class breakdown
    label_names = {0: "QCD", 1: "Hbb", 2: "Hcc", 3: "Hgg", 4: "H4q", 6: "Zqq", 7: "Wqq", 8: "Tbqq"}
    test_labels = y_labels[n_train:]

    print(f"\n=== Per-class R² ===")
    print(f"  {'Class':5s} {'n':>6s}  {'Mass R²':>8s}  {'SD R²':>8s}")
    for lbl, name in sorted(label_names.items()):
        mask = test_labels == lbl
        if mask.sum() > 100:
            r2_m_cls = r2_score(y_mass[n_train:][mask], pred_mass[mask])
            r2_sd_cls = r2_score(y_sd[n_train:][mask], pred_sd[mask])
            print(f"  {name:5s} {mask.sum():6d}  {r2_m_cls:8.4f}  {r2_sd_cls:8.4f}")


if __name__ == "__main__":
    main()

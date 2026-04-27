#!/usr/bin/env python3
"""Train the shared VQ-VAE for particle tokenization.

Usage:
    python baselines/vqvae/train.py -c baselines/nrp/configs/vqvae/train.yaml

The VQ-VAE tokenizes per-particle RINO kinematics (7 features) into discrete
codebook indices. The trained checkpoint is used downstream by MPMv1 (masked
prediction) and OmniJet-alpha (autoregressive prediction).
"""

import argparse
import os
import sys
from itertools import cycle, islice
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import yaml

# Allow imports from baselines/ root
_BASELINES_DIR = Path(__file__).resolve().parents[1]
if str(_BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINES_DIR))

_PROJECT_ROOT = _BASELINES_DIR.parent

from vqvae.vqvae_lightning import SharedVQVAELightning  # noqa: E402

# ── RINO feature computation (same as mpmv1/src/datamodules/rino_iterator.py) ──
import awkward as ak  # noqa: E402
import uproot  # noqa: E402

_NORM = {
    "log_pt": {"mean": 1.7, "std": 1.8},
    "log_energy": {"mean": 2.0, "std": 1.8},
    "log_rel_pt": {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "deta": {"mean": 0.0, "std": 0.14},
    "dphi": {"mean": 0.0, "std": 0.14},
    "delta_R": {"mean": 0.14, "std": 0.25},
}
_EPS = np.finfo(np.float32).eps
_LABEL_BRANCHES = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
    "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl",
]


def _norm(x, key):
    return (x - _NORM[key]["mean"]) / _NORM[key]["std"]


def _load_rino_root(filepath, max_n_csts=128):
    """Load a JetClass ROOT file and return RINO features."""
    branches = ["part_energy", "part_px", "part_py", "part_pz", "part_deta", "part_dphi"]
    jet_branches = ["jet_pt", "jet_energy"]

    with uproot.open(filepath) as f:
        treename = None
        for k, v in f.items():
            if getattr(v, "classname", "") == "TTree":
                treename = k.split(";")[0]
                break
        tree = f[treename]
        outputs = tree.arrays(filter_name=branches, library="ak")
        jet_out = tree.arrays(filter_name=jet_branches, library="np")

    awk_arr = ak.pad_none(outputs, max_n_csts, clip=True)
    part_energy = ak.to_numpy(awk_arr["part_energy"]).astype("float32").data
    part_px = ak.to_numpy(awk_arr["part_px"]).astype("float32").data
    part_py = ak.to_numpy(awk_arr["part_py"]).astype("float32").data
    part_deta = ak.to_numpy(awk_arr["part_deta"]).astype("float32").data
    part_dphi = ak.to_numpy(awk_arr["part_dphi"]).astype("float32").data

    jet_pt = jet_out["jet_pt"].astype("float32")
    jet_energy = jet_out["jet_energy"].astype("float32")

    mask = ~np.isnan(part_energy)
    for arr in [part_energy, part_px, part_py, part_deta, part_dphi]:
        np.nan_to_num(arr, copy=False, nan=0.0)

    part_pt = np.sqrt(part_px**2 + part_py**2)
    features = np.stack([
        _norm(np.log(part_pt + _EPS), "log_pt"),
        _norm(np.log(part_energy + _EPS), "log_energy"),
        _norm(np.log(part_pt / (jet_pt[:, None] + _EPS) + _EPS), "log_rel_pt"),
        _norm(np.log(part_energy / (jet_energy[:, None] + _EPS) + _EPS), "log_rel_energy"),
        _norm(np.sqrt(part_deta**2 + part_dphi**2), "delta_R"),
        _norm(part_deta, "deta"),
        _norm(part_dphi, "dphi"),
    ], axis=-1)
    features[~mask] = 0.0
    return features, mask


class RINOH5Dataset(torch.utils.data.Dataset):
    """HDF5 dataset for RINO-preprocessed JetClass data.

    Reads directly from HDF5 via h5py slice indexing — fast, no caching needed.
    Expects HDF5 files produced by baselines/mpmv2/scripts/make_jetclass.py
    (main_rino_kinematics) with datasets: csts (N, 128, 7), mask (N, 128).
    """

    def __init__(self, path: str, csts_dim: int = 7):
        super().__init__()
        self.path = path
        self.csts_dim = csts_dim
        self._file = None
        with h5py.File(path, mode="r") as f:
            self.n_jets = len(f["csts"])
        print(f"Opened {path}: {self.n_jets} jets")

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, mode="r", swmr=True)

    def __len__(self):
        return self.n_jets

    def __getitem__(self, idx):
        self._ensure_open()
        csts = self._file["csts"][idx].astype(np.float32)
        mask = self._file["mask"][idx].astype(bool)
        # Select only the first csts_dim features
        if csts.ndim == 2:
            csts = csts[:, :self.csts_dim]
        return {
            "sequence": torch.from_numpy(csts),
            "mask": torch.from_numpy(mask),
        }


class RINOVQVAEDataModule(pl.LightningDataModule):
    """DataModule for VQ-VAE training on RINO HDF5 files.

    Uses preprocessed HDF5 files from make_jetclass.py (main_rino_kinematics).
    Fast random-access via h5py — no ROOT parsing overhead.
    """

    def __init__(self, train_path, val_path, batch_size=2000, num_workers=4,
                 csts_dim=7):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csts_dim = csts_dim

    def train_dataloader(self):
        ds = RINOH5Dataset(self.train_path, self.csts_dim)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        ds = RINOH5Dataset(self.val_path, self.csts_dim)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser(description="Train shared VQ-VAE")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve PROJECT_ROOT in paths
    for key in ("experiment_dir",):
        if key in config and isinstance(config[key], str):
            config[key] = config[key].replace("PROJECT_ROOT", str(_PROJECT_ROOT))
            config[key] = config[key].replace("JOBNAME", config.get("name", "vqvae"))

    # Data
    dl_config = config["training"]["dataloader"]
    datamodule = RINOVQVAEDataModule(
        train_path=dl_config["train_path"],
        val_path=dl_config["val_path"],
        batch_size=dl_config.get("batch_size", 2000),
        num_workers=dl_config.get("num_workers", 4),
        csts_dim=dl_config.get("csts_dim", 7),
    )

    # Model
    model = SharedVQVAELightning(
        model_kwargs=config.get("model_params", {}),
        lr=config["training"].get("lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # Trainer
    experiment_dir = Path(config.get("experiment_dir", "experiments/vqvae"))
    experiment_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=experiment_dir / "checkpoints",
            filename="vqvae-{epoch:03d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    logger = pl.loggers.WandbLogger(
        project="parcel-baselines",
        name=config.get("name", "vqvae"),
        save_dir=str(experiment_dir),
    ) if config.get("use_wandb", True) else True

    # Auto-select precision: bf16 if supported, else fp32
    precision = config.get("precision", "auto")
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "32-true"
    print(f"Using precision: {precision}")

    trainer = pl.Trainer(
        max_epochs=config["training"].get("num_epochs", 100),
        accelerator="auto",
        devices="auto",
        precision=precision,
        gradient_clip_val=config["training"].get("grad_clip", 5.0),
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(experiment_dir),
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

"""JetNet binary (top vs QCD) datamodule for OmniJet-alpha finetuning.

Reads JetNet30_gqt HDF5 files, computes the same 7 RINO kinematics used for
PARCEL pretraining, and returns per-sample dicts:
    sequence: (n_csts, 7) float32 — RINO-normalized particle features
    mask:     (n_csts,)   bool     — True = valid particle
    labels:   ()          long     — 0 = QCD (g/q), 1 = top

The VQ-VAE tokenization happens inside the classification Lightning module.
Lazy-open h5py pattern so DataLoader workers each re-open after fork.
"""

import logging

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

_NORM = {
    "log_pt":         {"mean":  1.7, "std": 1.8},
    "log_energy":     {"mean":  2.0, "std": 1.8},
    "log_rel_pt":     {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "delta_R":        {"mean":  0.14, "std": 0.25},
    "deta":           {"mean":  0.0, "std": 0.14},
    "dphi":           {"mean":  0.0, "std": 0.14},
}
_EPS = np.finfo(np.float32).eps

# JetNet30_gqt label convention: 0=gluon, 1=quark, 2=top.
_TOP_LABEL = 2


def _norm(x: np.ndarray, key: str) -> np.ndarray:
    s = _NORM[key]
    return (x - s["mean"]) / s["std"]


class JetNetBinaryDataset(Dataset):
    """Per-sample dataset reading one JetNet30_gqt HDF5 file."""

    def __init__(
        self,
        path: str,
        n_csts: int = 128,
        csts_dim: int = 7,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.n_csts = n_csts
        self.csts_dim = csts_dim
        self.file = None
        with h5py.File(self.path, mode="r") as f:
            self.n_jets = int(f["jet_label"].shape[0])
        log.info(f"JetNetBinaryDataset({self.path}): {self.n_jets} jets")

    def _ensure_open(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.path, mode="r")

    def __len__(self) -> int:
        return self.n_jets

    def _compute_features(
        self,
        part_px: np.ndarray,
        part_py: np.ndarray,
        part_energy: np.ndarray,
        part_deta: np.ndarray,
        part_dphi: np.ndarray,
        part_is_real: np.ndarray,
        jet_pt: float,
        jet_energy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = part_is_real.astype(bool)
        n_src = part_px.shape[0]

        part_pt = np.sqrt(part_px**2 + part_py**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_pt         = _norm(np.log(part_pt + _EPS),                           "log_pt")
            log_energy     = _norm(np.log(part_energy + _EPS),                       "log_energy")
            log_rel_pt     = _norm(np.log(part_pt / (jet_pt + _EPS) + _EPS),         "log_rel_pt")
            log_rel_energy = _norm(np.log(part_energy / (jet_energy + _EPS) + _EPS), "log_rel_energy")
            delta_R        = _norm(np.sqrt(part_deta**2 + part_dphi**2),             "delta_R")
            deta           = _norm(part_deta,                                        "deta")
            dphi           = _norm(part_dphi,                                        "dphi")

        feats = np.stack(
            [log_pt, log_energy, log_rel_pt, log_rel_energy, delta_R, deta, dphi],
            axis=-1,
        ).astype(np.float32)
        feats[~mask] = 0.0

        if n_src >= self.n_csts:
            feats_out = feats[: self.n_csts]
            mask_out = mask[: self.n_csts]
        else:
            feats_out = np.zeros((self.n_csts, 7), dtype=np.float32)
            mask_out = np.zeros(self.n_csts, dtype=bool)
            feats_out[:n_src] = feats
            mask_out[:n_src] = mask

        feats_out = feats_out[:, : self.csts_dim]
        return feats_out, mask_out

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        f = self.file

        part_px      = f["part_px"][idx].astype(np.float32)
        part_py      = f["part_py"][idx].astype(np.float32)
        part_energy  = f["part_energy"][idx].astype(np.float32)
        part_deta    = f["part_deta"][idx].astype(np.float32)
        part_dphi    = f["part_dphi"][idx].astype(np.float32)
        part_is_real = f["part_is_real"][idx].astype(np.float32)

        jet_pt     = float(f["jet_pt"][idx])
        jet_energy = float(f["jet_energy"][idx])
        jet_label  = int(f["jet_label"][idx])

        sequence, mask = self._compute_features(
            part_px, part_py, part_energy, part_deta, part_dphi, part_is_real,
            jet_pt, jet_energy,
        )

        label = 1 if jet_label == _TOP_LABEL else 0

        return {
            "sequence": torch.from_numpy(sequence),
            "mask": torch.from_numpy(mask),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class JetNetBinaryDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping JetNet binary top-vs-QCD train/val/test splits."""

    def __init__(
        self,
        train_path: str = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_train.h5",
        val_path: str   = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_val.h5",
        test_path: str  = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_test.h5",
        batch_size: int = 500,
        num_workers: int = 6,
        n_csts: int = 128,
        csts_dim: int = 7,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _make_loader(self, path: str, shuffle: bool, drop_last: bool) -> DataLoader:
        ds = JetNetBinaryDataset(
            path=path,
            n_csts=self.hparams.n_csts,
            csts_dim=self.hparams.csts_dim,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.hparams.train_path, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.hparams.val_path, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.hparams.test_path, shuffle=False, drop_last=False)

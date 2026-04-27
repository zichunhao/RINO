"""JetNet binary (top vs QCD) datamodule for MPMv2 finetuning.

Reads JetNet30_gqt HDF5 files and computes the same 7 RINO kinematic features
used for MPMv2 pretraining. Returns per-sample dicts with the batch schema
expected by src.models.classifier.Classifier:
    csts:    (n_csts, csts_dim) float32
    csts_id: (n_csts,) long    — filled with zeros (MPMv2 RINO backbone ignores IDs)
    mask:    (n_csts,) bool
    labels:  () long           — 0=QCD, 1=top
    jets:    (jet_dim,) float32

Lazy-open h5py pattern so DataLoader workers each re-open the file after fork.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# Normalisation constants (same as dino/dataloader/jetclass/processors.py).
_NORM = {
    "log_pt":         {"mean":  1.7, "std": 1.8},
    "log_energy":     {"mean":  2.0, "std": 1.8},
    "log_rel_pt":     {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "delta_R":        {"mean":  0.14, "std": 0.25},
    "deta":           {"mean":  0.0, "std": 0.14},
    "dphi":           {"mean":  0.0, "std": 0.14},
}

_JET_NORM = {
    "jet_log_pt":     {"mean": 6.4, "std": 0.25},
    "jet_log_energy": {"mean": 6.7, "std": 0.5},
    "jet_eta":        {"mean": 0.0, "std": 1.3},
    "jet_phi":        {"mean": 0.0, "std": np.pi},
}

_EPS = np.finfo(np.float32).eps

# JetNet30_gqt label convention: gluon=0, light-quark=1, top=2 (gqt ordering).
# Binary task folds {g, q} -> 0 and {top} -> 1.
_TOP_LABEL = 2


def _norm(x: np.ndarray, key: str) -> np.ndarray:
    s = _NORM[key]
    return (x - s["mean"]) / s["std"]


def _norm_jet(x: np.ndarray, key: str) -> np.ndarray:
    s = _JET_NORM[key]
    return (x - s["mean"]) / s["std"]


class JetNetBinaryDataset(Dataset):
    """Per-sample Dataset over a JetNet30_gqt HDF5 file.

    Pads/truncates particles to ``n_csts`` and returns RINO features.
    """

    def __init__(
        self,
        path: str,
        n_csts: int = 128,
        csts_dim: int = 7,
        n_jets: int | None = None,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.n_csts = n_csts
        self.csts_dim = csts_dim
        self._file = None
        with h5py.File(self.path, mode="r") as f:
            self.total_jets = int(f["jet_label"].shape[0])
        self.n_jets = self.total_jets if n_jets is None else min(n_jets, self.total_jets)
        # Classifier expects n_classes as an attribute on the datamodule; we also
        # expose it here so get_data_sample can be driven from a single dataset.
        self.n_classes = 2
        log.info(f"JetNetBinaryDataset({self.path}): {self.n_jets}/{self.total_jets} jets")

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.path, mode="r")

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
        """Compute 7 RINO features for one jet, padded to ``n_csts``."""
        mask = part_is_real.astype(bool)
        n_src = part_px.shape[0]

        part_pt = np.sqrt(part_px**2 + part_py**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_pt         = _norm(np.log(part_pt + _EPS),                        "log_pt")
            log_energy     = _norm(np.log(part_energy + _EPS),                    "log_energy")
            log_rel_pt     = _norm(np.log(part_pt / (jet_pt + _EPS) + _EPS),      "log_rel_pt")
            log_rel_energy = _norm(np.log(part_energy / (jet_energy + _EPS) + _EPS), "log_rel_energy")
            delta_R        = _norm(np.sqrt(part_deta**2 + part_dphi**2),          "delta_R")
            deta           = _norm(part_deta,                                     "deta")
            dphi           = _norm(part_dphi,                                     "dphi")

        feats = np.stack(
            [log_pt, log_energy, log_rel_pt, log_rel_energy, delta_R, deta, dphi],
            axis=-1,
        ).astype(np.float32)  # (n_src, 7)

        # Zero out padded positions
        feats[~mask] = 0.0

        # Pad or truncate to n_csts
        if n_src >= self.n_csts:
            feats_out = feats[: self.n_csts]
            mask_out = mask[: self.n_csts]
        else:
            feats_out = np.zeros((self.n_csts, 7), dtype=np.float32)
            mask_out = np.zeros(self.n_csts, dtype=bool)
            feats_out[:n_src] = feats
            mask_out[:n_src] = mask

        # Slice to requested csts_dim (in case a subset is wanted)
        feats_out = feats_out[:, : self.csts_dim]
        return feats_out, mask_out

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        f = self._file

        part_px     = f["part_px"][idx].astype(np.float32)
        part_py     = f["part_py"][idx].astype(np.float32)
        part_energy = f["part_energy"][idx].astype(np.float32)
        part_deta   = f["part_deta"][idx].astype(np.float32)
        part_dphi   = f["part_dphi"][idx].astype(np.float32)
        part_is_real = f["part_is_real"][idx].astype(np.float32)

        jet_pt     = float(f["jet_pt"][idx])
        jet_energy = float(f["jet_energy"][idx])
        jet_eta    = float(f["jet_eta"][idx])
        jet_phi    = float(f["jet_phi"][idx])
        jet_label  = int(f["jet_label"][idx])

        csts, mask = self._compute_features(
            part_px, part_py, part_energy, part_deta, part_dphi, part_is_real,
            jet_pt, jet_energy,
        )

        jets = np.array(
            [
                _norm_jet(np.log(jet_pt + _EPS), "jet_log_pt"),
                _norm_jet(np.log(jet_energy + _EPS), "jet_log_energy"),
                _norm_jet(jet_eta, "jet_eta"),
                _norm_jet(jet_phi, "jet_phi"),
            ],
            dtype=np.float32,
        )

        label = 1 if jet_label == _TOP_LABEL else 0

        return {
            "csts": torch.from_numpy(csts),
            "csts_id": torch.zeros(self.n_csts, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "labels": torch.tensor(label, dtype=torch.long),
            "jets": torch.from_numpy(jets),
        }


class JetNetBinaryDataModule(LightningDataModule):
    """Lightning DataModule wrapping train/val/test JetNetBinaryDatasets.

    Exposes ``get_data_sample()`` and ``n_classes`` so it is compatible with
    ``src.models.classifier.Classifier``.
    """

    def __init__(
        self,
        train_path: str = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_train.h5",
        val_path: str   = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_val.h5",
        test_path: str  = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/gqt_test.h5",
        n_csts: int = 128,
        csts_dim: int = 7,
        n_jets: int | None = None,
        batch_size: int = 500,
        num_workers: int = 6,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.n_csts = n_csts
        self.csts_dim = csts_dim
        self.n_jets = n_jets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Pre-instantiate val so n_classes and get_data_sample are available at init.
        self.val_set = JetNetBinaryDataset(
            path=self.val_path,
            n_csts=self.n_csts,
            csts_dim=self.csts_dim,
            n_jets=self.n_jets,
        )
        self.train_set: Dataset | None = None
        self.test_set: Dataset | None = None
        self.n_classes = self.val_set.n_classes

    def setup(self, stage: str) -> None:
        if stage in {"fit", "train"} and self.train_set is None:
            self.train_set = JetNetBinaryDataset(
                path=self.train_path,
                n_csts=self.n_csts,
                csts_dim=self.csts_dim,
                n_jets=self.n_jets,
            )
        if stage in {"predict", "test"} and self.test_set is None:
            self.test_set = JetNetBinaryDataset(
                path=self.test_path,
                n_csts=self.n_csts,
                csts_dim=self.csts_dim,
                n_jets=None,
            )

    def get_data_sample(self) -> dict:
        return self.val_set[0]

    def _loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

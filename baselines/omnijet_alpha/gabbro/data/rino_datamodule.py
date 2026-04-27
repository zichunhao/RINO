"""RINO kinematics DataModule for OmniJet-alpha.

Reads from preprocessed HDF5 files (produced by make_jetclass.py main_rino_kinematics).
Returns continuous features for online tokenization by a frozen VQ-VAE in the model step.
"""

import logging
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# Normalization statistics from dino/dataloader/jetclass/processors.py
_NORM = {
    "log_pt": {"mean": 1.7, "std": 1.8},
    "log_energy": {"mean": 2.0, "std": 1.8},
    "log_rel_pt": {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "deta": {"mean": 0.0, "std": 0.14},
    "dphi": {"mean": 0.0, "std": 0.14},
    "delta_R": {"mean": 0.14, "std": 0.25},
    "jet_log_energy": {"mean": 6.7, "std": 0.5},
    "jet_log_pt": {"mean": 6.4, "std": 0.25},
    "jet_eta": {"mean": 0.0, "std": 1.3},
    "jet_phi": {"mean": 0.0, "std": 3.14159265},
}

_EPS = np.finfo(np.float32).eps
_LABEL_BRANCHES = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
    "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl",
]


def _normalize(x, mean, std):
    return (x - mean) / std


def load_rino_root_file(filepath, max_n_csts=128):
    """Load a JetClass ROOT file and return RINO-normalized features."""
    branches = [
        "part_energy", "part_px", "part_py", "part_pz",
        "part_deta", "part_dphi",
    ]
    jet_branches = ["jet_pt", "jet_energy", "jet_eta", "jet_phi"]

    with uproot.open(filepath) as f:
        treename = None
        for k, v in f.items():
            if getattr(v, "classname", "") == "TTree":
                treename = k.split(";")[0]
                break
        tree = f[treename]
        outputs = tree.arrays(filter_name=branches, library="ak")
        jet_out = tree.arrays(filter_name=jet_branches, library="np")
        labels = tree.arrays(filter_name=_LABEL_BRANCHES, library="pd")

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

    # 7 RINO particle features
    features = np.stack([
        _normalize(np.log(part_pt + _EPS), _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]),
        _normalize(np.log(part_energy + _EPS), _NORM["log_energy"]["mean"], _NORM["log_energy"]["std"]),
        _normalize(np.log(part_pt / (jet_pt[:, None] + _EPS) + _EPS), _NORM["log_rel_pt"]["mean"], _NORM["log_rel_pt"]["std"]),
        _normalize(np.log(part_energy / (jet_energy[:, None] + _EPS) + _EPS), _NORM["log_rel_energy"]["mean"], _NORM["log_rel_energy"]["std"]),
        _normalize(np.sqrt(part_deta**2 + part_dphi**2), _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]),
        _normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"]),
        _normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"]),
    ], axis=-1)  # (n_jets, n_csts, 7)

    features[~mask] = 0.0
    label = labels[_LABEL_BRANCHES].to_numpy().astype(np.float32).argmax(axis=1)

    return features, mask, label


class RINOH5Dataset(torch.utils.data.Dataset):
    """HDF5 dataset for RINO-preprocessed JetClass data.

    Reads from combined HDF5 files produced by make_jetclass.py (main_rino_kinematics).
    Fast random access via h5py slice indexing.
    """

    def __init__(self, path: str, csts_dim: int = 7):
        super().__init__()
        self.path = path
        self.csts_dim = csts_dim
        self.file = None
        with h5py.File(path, mode="r") as f:
            self.n_jets = len(f["csts"])
        logger.info(f"Opened {path}: {self.n_jets} jets")

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.path, mode="r", swmr=True)

    def __len__(self):
        return self.n_jets

    def __getitem__(self, idx):
        self._ensure_open()
        csts = self.file["csts"][idx].astype(np.float32)[:, :self.csts_dim]
        mask = self.file["mask"][idx].astype(bool)
        return {
            "sequence": torch.from_numpy(csts),
            "mask": torch.from_numpy(mask),
        }


class RINODataModule(L.LightningDataModule):
    """Lightning DataModule for RINO HDF5 files.

    Uses preprocessed HDF5 files from make_jetclass.py (main_rino_kinematics).
    Fast random-access via h5py — no ROOT parsing overhead.
    """

    def __init__(
        self,
        train_path: str = "PROJECT_ROOT/data/JetClass/mpm-rino/train_100M_combined_QCD.h5",
        val_path: str = "PROJECT_ROOT/data/JetClass/mpm-rino/val_5M_combined_QCD.h5",
        batch_size: int = 500,
        num_workers: int = 4,
        csts_dim: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        ds = RINOH5Dataset(self.hparams.train_path, self.hparams.csts_dim)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        ds = RINOH5Dataset(self.hparams.val_path, self.hparams.csts_dim)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

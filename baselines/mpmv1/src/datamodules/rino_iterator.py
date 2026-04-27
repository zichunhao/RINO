"""RINO kinematics iterator for JetClass ROOT files.

Produces 7 normalized RINO features per particle and 4 jet-level features,
matching the RINO/DINO data pipeline for apples-to-apples comparison.

RINO particle features (7):
    0. norm_log_part_pt          — log(pT_i), normalized
    1. norm_log_part_energy      — log(E_i), normalized
    2. norm_log_rel_part_pt      — log(pT_i / pT_jet), normalized
    3. norm_log_rel_part_energy  — log(E_i / E_jet), normalized
    4. norm_part_delta_R         — sqrt(deta^2 + dphi^2), normalized
    5. norm_part_deta            — delta_eta, normalized
    6. norm_part_dphi            — delta_phi, normalized

RINO jet features (4):
    0. norm_log_jet_energy
    1. norm_log_jet_pt
    2. norm_jet_eta
    3. norm_jet_phi
"""

import sys
from itertools import cycle, islice
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

# Normalization statistics from dino/dataloader/jetclass/processors.py
# Measured on JetClass individual particles (all classes, training split).
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


def _load_rino_jetclass(filepath, n_csts=128):
    """Load a JetClass ROOT file and compute RINO features."""
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

    # Pad/clip particles to n_csts
    awk_arr = ak.pad_none(outputs, n_csts, clip=True)

    part_energy = ak.to_numpy(awk_arr["part_energy"]).astype("float32").data
    part_px = ak.to_numpy(awk_arr["part_px"]).astype("float32").data
    part_py = ak.to_numpy(awk_arr["part_py"]).astype("float32").data
    part_deta = ak.to_numpy(awk_arr["part_deta"]).astype("float32").data
    part_dphi = ak.to_numpy(awk_arr["part_dphi"]).astype("float32").data

    jet_pt = jet_out["jet_pt"].astype("float32")
    jet_energy = jet_out["jet_energy"].astype("float32")
    jet_eta = jet_out["jet_eta"].astype("float32")
    jet_phi = jet_out["jet_phi"].astype("float32")

    # Mask: True = valid particle (NaN from padding)
    mask = ~np.isnan(part_energy)
    part_energy = np.nan_to_num(part_energy, nan=0.0)
    part_px = np.nan_to_num(part_px, nan=0.0)
    part_py = np.nan_to_num(part_py, nan=0.0)
    part_deta = np.nan_to_num(part_deta, nan=0.0)
    part_dphi = np.nan_to_num(part_dphi, nan=0.0)

    part_pt = np.sqrt(part_px**2 + part_py**2)

    # 7 RINO particle features
    f0 = _normalize(np.log(part_pt + _EPS), _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"])
    f1 = _normalize(np.log(part_energy + _EPS), _NORM["log_energy"]["mean"], _NORM["log_energy"]["std"])
    f2 = _normalize(
        np.log(part_pt / (jet_pt[:, None] + _EPS) + _EPS),
        _NORM["log_rel_pt"]["mean"], _NORM["log_rel_pt"]["std"],
    )
    f3 = _normalize(
        np.log(part_energy / (jet_energy[:, None] + _EPS) + _EPS),
        _NORM["log_rel_energy"]["mean"], _NORM["log_rel_energy"]["std"],
    )
    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    f4 = _normalize(delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"])
    f5 = _normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    f6 = _normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])

    # Stack: (n_jets, n_csts, 7)
    nodes = np.stack([f0, f1, f2, f3, f4, f5, f6], axis=-1)
    # Zero out padded positions
    nodes[~mask] = 0.0

    # 4 jet features
    jf0 = _normalize(np.log(jet_energy + _EPS), _NORM["jet_log_energy"]["mean"], _NORM["jet_log_energy"]["std"])
    jf1 = _normalize(np.log(jet_pt + _EPS), _NORM["jet_log_pt"]["mean"], _NORM["jet_log_pt"]["std"])
    jf2 = _normalize(jet_eta, _NORM["jet_eta"]["mean"], _NORM["jet_eta"]["std"])
    jf3 = _normalize(jet_phi, _NORM["jet_phi"]["mean"], _NORM["jet_phi"]["std"])
    high = np.stack([jf0, jf1, jf2, jf3], axis=-1)

    # Labels: argmax over one-hot
    label = labels[_LABEL_BRANCHES].to_numpy().astype(np.float32).argmax(axis=1).reshape(-1, 1)

    return high, nodes, mask, label


def _batched(iterable, n):
    """Batch data into tuples of length n."""
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class RINOJetClassIterator:
    """Iterator that loads JetClass ROOT files and produces RINO features.

    Returns per-sample tuples of (high, nodes, mask, label) where:
        - high: (4,) jet-level features
        - nodes: (n_csts, 7) RINO particle features
        - mask: (n_csts,) bool mask
        - label: (1,) class index
    """

    def __init__(
        self,
        dset: str,
        n_load: int = 20,
        path: str = "PROJECT_ROOT/data/JetClass/raw/",
        n_nodes: int = 128,
        processes: list = None,
        max_files: int = None,
        features: list = None,  # unused, kept for interface compat
    ):
        self.n_load = n_load
        self.dset = dset
        self.features = None  # RINO iterator always produces 7 features
        self.n_nodes = n_nodes

        data_path = Path(path)
        if dset == "train":
            direct = data_path / "train_100M"
            self.n_samples = 100_000_000
        elif dset == "test":
            direct = data_path / "test_20M"
            self.n_samples = 10_000_000
        else:
            direct = data_path / "val_5M"
            self.n_samples = 1_000_000

        proc_dict = {
            "QCD": ["ZJets"],
            "WZ": ["ZTo", "WTo"],
            "ttbar": ["TTBar_", "TTBarLep"],
            "higgs": ["HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q"],
        }
        if isinstance(processes, str):
            processes = [processes]
        elif processes is None:
            processes = list(proc_dict.keys())
        self.processes = processes

        self.file_list = []
        for process in processes:
            for pr in proc_dict[process]:
                proc_files = np.array(list(direct.glob(f"{pr}*.root")))
                if dset == "test":
                    proc_files = proc_files[
                        np.argsort([int(f.stem.split("_")[-1]) for f in proc_files])
                    ]
                self.file_list.append(proc_files)

        stacked_files = np.array(self.file_list).transpose()
        if max_files is not None and dset == "train":
            stacked_files = stacked_files[:max_files]
        self.n_samples = int(1e5) * np.prod(stacked_files.shape)
        self.file_list = stacked_files.flatten().tolist()
        self.file_iterator = _batched(cycle(self.file_list), self.n_load)

        self.data_i = 0
        self.load_data()

    def load_data(self):
        self.data_i = 0
        files = next(self.file_iterator)
        data_list = [_load_rino_jetclass(f, self.n_nodes) for f in files]
        self.data = [
            np.concatenate([d[i] for d in data_list], axis=0)
            for i in range(4)
        ]
        # Shuffle
        if self.dset != "test":
            idx = np.random.permutation(len(self.data[0]))
            self.data = [d[idx] for d in self.data]

    def get_nclasses(self):
        return 10  # JetClass has 10 classes

    def __next__(self):
        try:
            sample = [d[self.data_i] for d in self.data]
            self.data_i += 1
        except IndexError:
            self.load_data()
            sample = [d[self.data_i] for d in self.data]
            self.data_i += 1
        return sample  # (high, nodes, mask, label)

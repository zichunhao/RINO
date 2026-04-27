"""JetNet binary (top vs QCD) iterator for MPMv1 finetuning.

Reads ``gqt_{train,val,test}.h5`` (JetNet30_gqt) with raw kinematics and
computes the 7 RINO features on-the-fly using the same normalisation
constants as MPMv1's RINO pretraining pipeline. Matches the 4-tuple
``(high, nodes, mask, label)`` interface of :class:`RINOH5Iterator`, so it
plugs straight into ``RINOIteratorWrapper`` + ``PointCloudDataModule``.

Lazy-open h5py: the file handle is opened on first ``__next__`` (per
DataLoader worker) so the object is fork-safe.
"""

import h5py
import numpy as np

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
# JetNet30_gqt label convention: 0 = gluon, 1 = light quark, 2 = top.
_TOP_LABEL = 2


def _norm(x: np.ndarray, key: str) -> np.ndarray:
    s = _NORM[key]
    return (x - s["mean"]) / s["std"]


class JetNetBinaryIterator:
    """Iterator that reads JetNet gqt HDF5 and yields per-sample tuples.

    Returns per-sample ``(high, nodes, mask, label)`` where:
        high: (0,)      — jet-level features (unused here)
        nodes: (n_nodes, 7) — 7 RINO particle features
        mask: (n_nodes,) bool
        label: (1,)     — binary: 1 for top, 0 for g/q
    """

    def __init__(
        self,
        dset: str,
        path: str = "PROJECT_ROOT/data/JetNet/JetNet30_gqt/",
        n_nodes: int = 128,
        n_load: int = 20,        # unused; interface compat
        processes: list = None,  # unused; interface compat
        max_files: int = None,   # unused; interface compat
        features: list = None,   # unused; interface compat
    ) -> None:
        self.dset = dset
        self.n_nodes = n_nodes
        self.features = None

        split = {"train": "gqt_train.h5", "val": "gqt_val.h5", "test": "gqt_test.h5"}
        h5_name = split.get(dset, "gqt_val.h5")
        self.h5_path = f"{path.rstrip('/')}/{h5_name}"

        # Lazy-open: only peek metadata here, open file handles per worker.
        self.file = None
        with h5py.File(self.h5_path, "r") as f:
            self.n_jets = int(f["jet_label"].shape[0])
        self.n_samples = self.n_jets

        # Iteration state — shuffle on train, pass-through otherwise.
        if dset == "train":
            self._indices = np.random.permutation(self.n_jets)
        else:
            self._indices = np.arange(self.n_jets)
        self.data_i = 0

        print(f"[JetNetBinaryIterator] Opened {self.h5_path}: {self.n_jets} jets")

    def _ensure_open(self) -> None:
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

    def get_nclasses(self) -> int:
        return 2

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
        valid = part_is_real.astype(bool)
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
        feats[~valid] = 0.0

        if n_src >= self.n_nodes:
            feats_out = feats[: self.n_nodes]
            mask_out = valid[: self.n_nodes]
        else:
            feats_out = np.zeros((self.n_nodes, 7), dtype=np.float32)
            mask_out = np.zeros(self.n_nodes, dtype=bool)
            feats_out[:n_src] = feats
            mask_out[:n_src] = valid
        return feats_out, mask_out

    def __next__(self):
        self._ensure_open()

        if self.data_i >= self.n_jets:
            if self.dset == "train":
                self._indices = np.random.permutation(self.n_jets)
            self.data_i = 0

        idx = int(self._indices[self.data_i])
        self.data_i += 1

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

        nodes, mask = self._compute_features(
            part_px, part_py, part_energy, part_deta, part_dphi, part_is_real,
            jet_pt, jet_energy,
        )

        label = np.array(
            [1 if jet_label == _TOP_LABEL else 0], dtype=np.float32
        )
        high = np.zeros(0, dtype=np.float32)
        return high, nodes, mask, label

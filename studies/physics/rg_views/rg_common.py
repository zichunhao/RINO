"""Shared helpers for the RG study: input reading, RINO-style views, EMD, classifier.

The views follow RINO's preprocessing (dino/preprocess/jetclass/cluster.py in the
RINO repository): each jet's constituents are padded with zero four-vectors to the
largest N, clustered with the exclusive version of the chosen algorithm at
R = 0.8 (FastJet, E-scheme), and the N subjets are sorted by pT. Detector-level
jets keep their 90 highest-pT constituents, as in RINO's shuffle step.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
import warnings

import awkward as ak
import numpy as np
import vector

vector.register_awkward()

R_JET = 0.8
DEFAULT_NUMS = (2, 3, 4, 6, 8, 16)
LEVELS = ("reco", "gen", "parton")
ALGORITHMS = ("kt", "ca", "antikt")

# Branch prefixes of the four-momenta for each level in the makeNtuplesRG.C output.
LEVEL_PREFIX = {"reco": "part_", "gen": "gen_part_", "parton": "parton_"}

# JetClass gen_match codes (makeNtuples.C FatJetLabel)
QCD_CODES = (50, 51, 52, 53, 54, 55)
TOP_HADRONIC_CODES = (11, 12)  # Top_bcq, Top_bqq
CLASS_CODES = {"qcd": 0, "top": 1}


def _import_fastjet():
    # FastJet prints its banner on import; keep stdout clean for the scripts.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        import fastjet  # noqa: F401
    return sys.modules["fastjet"]


fastjet = _import_fastjet()

FJ_ALGORITHMS = {
    "kt": fastjet.kt_algorithm,
    "ca": fastjet.cambridge_algorithm,
    "antikt": fastjet.antikt_algorithm,
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def expand_inputs(patterns):
    files = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        files.extend(matches if matches else [p])
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"input files not found: {missing[:5]}")
    return files


def jet_class(arrays) -> np.ndarray:
    """0 = QCD, 1 = hadronic top (bqq or bcq), -1 = anything else.

    Works for makeNtuples(RG) output (gen_match) and for the public JetClass
    format (label_* branches).
    """
    fields = arrays.fields
    if "gen_match" in fields:
        gm = ak.to_numpy(arrays["gen_match"]).astype(int)
        out = np.full(len(gm), -1)
        out[np.isin(gm, QCD_CODES)] = 0
        out[np.isin(gm, TOP_HADRONIC_CODES)] = 1
        return out
    if "label_QCD" in fields:
        out = np.full(len(arrays), -1)
        out[ak.to_numpy(arrays["label_QCD"]) > 0.5] = 0
        out[ak.to_numpy(arrays["label_Tbqq"]) > 0.5] = 1
        return out
    raise KeyError("input has neither gen_match nor label_* branches")


def p4(px, py, pz, energy):
    return ak.zip({"px": px, "py": py, "pz": pz, "E": energy}, with_name="Momentum4D")


def level_p4(arrays, level):
    pre = LEVEL_PREFIX[level]
    return p4(arrays[pre + "px"], arrays[pre + "py"], arrays[pre + "pz"], arrays[pre + "energy"])


def sort_by_pt(parts):
    pt = np.hypot(parts.px, parts.py)
    return parts[ak.argsort(pt, axis=1, ascending=False)]


def pad_zero(parts, n):
    """RINO-style padding: append zero four-vectors up to n particles."""
    fields = {f: ak.fill_none(ak.pad_none(parts[f], n, axis=1), 0.0) for f in ("px", "py", "pz", "E")}
    return p4(fields["px"], fields["py"], fields["pz"], fields["E"])


def to_numpy_p4(parts, maxlen):
    """Jagged Momentum4D -> (njets, maxlen, 4) float32 [px, py, pz, E], zero padded."""
    cols = [
        ak.to_numpy(ak.fill_none(ak.pad_none(parts[f], maxlen, axis=1, clip=True), 0.0))
        for f in ("px", "py", "pz", "E")
    ]
    return np.stack(cols, axis=-1).astype(np.float32)


def sum_p4(parts):
    return np.stack([ak.to_numpy(ak.sum(parts[f], axis=1)) for f in ("px", "py", "pz", "E")], axis=-1)


# ---------------------------------------------------------------------------
# Parton-level jets
# ---------------------------------------------------------------------------


def parton_jet_constituents(partons, axis_eta, axis_phi, max_dr=0.4):
    """Cluster each jet's parton list with anti-kT R = 0.8 and return the
    constituents of the parton jet closest to (axis_eta, axis_phi).

    Returns (constituents, valid): valid is False when no parton jet lies within
    max_dr of the axis.
    """
    padded = pad_zero(partons, 1)  # FastJet needs at least one input per jet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = fastjet.ClusterSequence(padded, fastjet.JetDefinition(fastjet.antikt_algorithm, R_JET))
        jets = cs.inclusive_jets()
        consts = cs.constituents()
    jets = ak.zip({"px": jets.px, "py": jets.py, "pz": jets.pz, "E": jets.E}, with_name="Momentum4D")
    pt = jets.pt
    deta = jets.eta - axis_eta
    dphi = (jets.phi - axis_phi + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)
    dr = ak.where(pt > 0, dr, 999.0)
    best = ak.argmin(dr, axis=1, keepdims=True)
    best_dr = ak.to_numpy(ak.fill_none(ak.flatten(dr[best]), 999.0))
    valid = best_dr < max_dr
    chosen = ak.firsts(consts[best])
    chosen = ak.fill_none(chosen, [], axis=0)
    chosen = p4(chosen.px, chosen.py, chosen.pz, chosen.E)
    # drop the zero padding particle, if it ended up in the chosen jet
    chosen = chosen[chosen.E > 0]
    return sort_by_pt(chosen), valid


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def make_views(parts, algorithm, nums, qgrid=None):
    """Exclusive-jet views of each jet for one algorithm.

    Returns dict with "N{n}" -> (njets, n, 4) subjet four-vectors sorted by pT,
    and for kT also "qN" -> (njets, len(nums)) merge scales sqrt(d_merge(N))
    [GeV] for the (N+1) -> N step, and "nexcl" -> (njets, len(qgrid)) number of
    exclusive jets at resolution d_cut = q^2.
    """
    padded = pad_zero(parts, max(nums))
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = fastjet.ClusterSequence(padded, fastjet.JetDefinition(FJ_ALGORITHMS[algorithm], R_JET))
        for n in nums:
            sj = cs.exclusive_jets(n)
            sj = p4(sj.px, sj.py, sj.pz, sj.E)
            out[f"N{n}"] = to_numpy_p4(sort_by_pt(sj), n)
        if algorithm == "kt":
            out["qN"] = np.stack(
                [np.sqrt(np.clip(ak.to_numpy(cs.exclusive_dmerge(n)), 0, None)) for n in nums], axis=-1
            ).astype(np.float32)
            if qgrid is not None:
                out["nexcl"] = np.stack(
                    [ak.to_numpy(cs.n_exclusive_jets(float(q) ** 2)) for q in qgrid], axis=-1
                ).astype(np.int16)
    return out


def view_names(nums):
    return [f"N{n}" for n in nums] + ["uncl"]


# ---------------------------------------------------------------------------
# Kinematics in the frame of a reference axis
# ---------------------------------------------------------------------------


def pt_y_phi(arr):
    """(..., 4) [px, py, pz, E] -> pt, rapidity, phi (rapidity 0 where undefined)."""
    px, py, pz, e = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    pt = np.hypot(px, py)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 0.5 * np.log((e + pz) / (e - pz))
    y = np.where((pt > 0) & np.isfinite(y), y, 0.0)
    phi = np.arctan2(py, px)
    return pt, y, phi


def relative_coords(arr, axis):
    """Constituents (njets, P, 4) relative to axis (njets, 4): pt, dy, dphi, mask."""
    pt, y, phi = pt_y_phi(arr)
    _, ya, phia = pt_y_phi(axis)
    dy = y - ya[:, None]
    dphi = (phi - phia[:, None] + np.pi) % (2 * np.pi) - np.pi
    mask = pt > 0
    return pt, np.where(mask, dy, 0.0), np.where(mask, dphi, 0.0), mask


# ---------------------------------------------------------------------------
# Energy mover's distance (Komiske, Metodiev, Thaler, PRL 123 (2019) 041801)
# ---------------------------------------------------------------------------


def _load_wasserstein():
    try:
        import wasserstein

        return wasserstein.EMD(R=R_JET, beta=1.0, norm=False)
    except Exception:
        pass
    # macOS wheels need the OpenMP runtime loaded globally first.
    for lib in (os.path.join(sys.prefix, "lib", "libomp.dylib"), "/opt/homebrew/opt/libomp/lib/libomp.dylib"):
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                for m in [m for m in sys.modules if m.startswith("wasserstein")]:
                    del sys.modules[m]
                import wasserstein

                return wasserstein.EMD(R=R_JET, beta=1.0, norm=False)
            except Exception:
                continue
    return None


_EMD = None
_EMD_LOADED = False


def _get_emd():
    # Loaded on first use only: on macOS a second OpenMP runtime (wasserstein's) in
    # the same process as torch's crashes, and `classify` never needs the EMD.
    global _EMD, _EMD_LOADED
    if not _EMD_LOADED:
        _EMD, _EMD_LOADED = _load_wasserstein(), True
    return _EMD


def _emd_lp(w0, c0, w1, c1, R=R_JET):
    """Exact EMD by linear programming (fallback when `wasserstein` is unavailable)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack

    n0, n1 = len(w0), len(w1)
    if n0 == 0 or n1 == 0:
        return float(abs(w0.sum() - w1.sum()))
    cost = (np.sqrt(((c0[:, None, :] - c1[None, :, :]) ** 2).sum(-1)) / R).ravel()
    nv = n0 * n1
    cols = np.arange(nv)
    a_row = coo_matrix((np.ones(nv), (np.repeat(np.arange(n0), n1), cols)), shape=(n0, nv))
    a_col = coo_matrix((np.ones(nv), (np.tile(np.arange(n1), n0), cols)), shape=(n1, nv))
    res = linprog(
        cost,
        A_ub=vstack([a_row, a_col]).tocsr(),
        b_ub=np.concatenate([w0, w1]),
        A_eq=np.ones((1, nv)),
        b_eq=[min(w0.sum(), w1.sum())],
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"EMD linear program failed: {res.message}")
    return float(res.fun + abs(w0.sum() - w1.sum()))


def emd(w0, c0, w1, c1):
    """EMD [GeV] between two weighted point sets in the (y, phi) plane, R = 0.8, beta = 1."""
    w0 = np.ascontiguousarray(w0, dtype=np.float64)
    w1 = np.ascontiguousarray(w1, dtype=np.float64)
    c0 = np.ascontiguousarray(c0, dtype=np.float64)
    c1 = np.ascontiguousarray(c1, dtype=np.float64)
    backend = _get_emd()
    if backend is not None and len(w0) and len(w1):
        return float(backend(w0, c0, w1, c1))
    return _emd_lp(w0, c0, w1, c1)


def emd_backend():
    return "wasserstein" if _get_emd() is not None else "scipy-linprog"


# ---------------------------------------------------------------------------
# ROC AUC (Mann-Whitney), no scikit-learn needed
# ---------------------------------------------------------------------------


def roc_auc(labels, scores):
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(bool)
    ranks = rankdata(scores)
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))

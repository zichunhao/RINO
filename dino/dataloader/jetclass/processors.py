from collections.abc import Callable

import awkward as ak
import numpy as np
import math
import sys

from .config import _DataloaderConfig

processors = {}
eps = sys.float_info.epsilon

# Normalization statistics for all kinematics processors.
# Particle stats measured on JetClass individual particles (all classes, training split).
_NORM = {
    # Particle log kinematics
    "log_pt": {"mean": 1.7, "std": 1.8},
    "log_energy": {"mean": 2.0, "std": 1.8},
    "log_rel_pt": {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    # Particle geometry
    "deta": {"mean": 0.0, "std": 0.14},
    "dphi": {"mean": 0.0, "std": 0.14},
    "delta_R": {"mean": 0.14, "std": 0.25},
    # Particle 4-momentum components
    "p4": {"mean": 0.0, "std": 25.0},
    # Jet kinematics
    "jet_log_energy": {"mean": 6.7, "std": 0.5},
    "jet_log_pt": {"mean": 6.4, "std": 0.25},
    "jet_eta": {"mean": 0.0, "std": 1.3},
    "jet_phi": {"mean": 0.0, "std": math.pi},
    "jet_nparticles": {"mean": 32, "std": 23},
}


def register_processor(processor: Callable[..., tuple[ak.Array | np.ndarray, ...]]):
    def wrapper(*args, _data: dict[str, ak.Array], **kwargs):
        args = [_data[x] for x in args]
        processed_kwargs = {}
        for k, v in kwargs.items():
            if v in _data:
                processed_kwargs[k] = _data[v]  # Pass as data reference
            else:
                # Pass as literal value
                processed_kwargs[k] = float(v)
        return processor(*args, **processed_kwargs)

    processors[processor.__name__] = wrapper
    return processor


def get_jetEPtEtaPhi(part_px, part_py, part_pz, part_energy):
    # sum up the 4-momentum components
    jet_energy = np.sum(part_energy, axis=-1)
    jet_px = np.sum(part_px, axis=-1)
    jet_py = np.sum(part_py, axis=-1)
    jet_pz = np.sum(part_pz, axis=-1)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_eta = np.arcsinh(jet_pz / (jet_pt + eps))
    jet_phi = np.arctan2(jet_py, jet_px)
    return jet_energy, jet_pt, jet_eta, jet_phi


@register_processor
def gaussian_smear(x: ak.Array, smear_factor):
    if smear_factor > 0:
        return x * (1 + np.random.normal(0, smear_factor, size=x.shape))
    return x


@register_processor
def cyl_to_cart(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz


@register_processor
def jetclass_labeler(
    label_QCD: ak.Array,  # 0
    label_Hbb: ak.Array,  # 1
    label_Hcc: ak.Array,  # 2
    label_Hgg: ak.Array,  # 3
    label_H4q: ak.Array,  # 4
    label_Hqql: ak.Array,  # 5
    label_Zqq: ak.Array,  # 6
    label_Wqq: ak.Array,  # 7
    label_Tbqq: ak.Array,  # 8
    label_Tbl: ak.Array,  # 9
):
    np_label_QCD = label_QCD.to_numpy()
    np_label_Hbb = label_Hbb.to_numpy()
    np_label_Hcc = label_Hcc.to_numpy()
    np_label_Hgg = label_Hgg.to_numpy()
    np_label_H4q = label_H4q.to_numpy()
    np_label_Hqql = label_Hqql.to_numpy()
    np_label_Zqq = label_Zqq.to_numpy()
    np_label_Wqq = label_Wqq.to_numpy()
    np_label_Tbqq = label_Tbqq.to_numpy()
    np_label_Tbl = label_Tbl.to_numpy()

    labels = np.stack(
        [
            np_label_QCD,
            np_label_Hbb,
            np_label_Hcc,
            np_label_Hgg,
            np_label_H4q,
            np_label_Hqql,
            np_label_Zqq,
            np_label_Wqq,
            np_label_Tbqq,
            np_label_Tbl,
        ],
        axis=1,
    )

    # assert np.all(np.sum(labels, axis=1) == 1)
    label = np.argmax(labels, axis=1)
    return (label,)


@register_processor
def jetclass_toptagging_labeler(
    label_QCD: ak.Array,
    label_Tbqq: ak.Array,
):
    np_label_QCD = label_QCD.to_numpy()
    np_label_Tbqq = label_Tbqq.to_numpy()

    labels = np.stack(
        [
            np_label_QCD,
            np_label_Tbqq,
        ],
        axis=1,
    )

    label = np.argmax(labels, axis=1)
    return (label,)


@register_processor
def jetnet_toptagging_labeler(
    label: np.ndarray,
):
    label = label.copy()
    # {g, q} -> QCD
    label[(label == 0) | (label == 1)] = 0
    # top
    label[label == 2] = 1
    return (label,)


@register_processor
def jetnet_wz_labeler(
    label: np.ndarray,
):
    """3-class labeler for W vs Z vs QCD.
    Input labels from gqwz data: g=0, q=1, w=2, z=3
    Output: QCD=0, W=1, Z=2
    """
    out = np.zeros_like(label)
    out[(label == 0) | (label == 1)] = 0  # g, q -> QCD
    out[label == 2] = 1  # w -> W
    out[label == 3] = 2  # z -> Z
    return (out,)


@register_processor
def jetnet_twz_labeler(
    label: np.ndarray,
):
    """4-class labeler for T vs W vs Z vs QCD.
    Input labels from gqtwz data: g=0, q=1, t=2, w=3, z=4
    Output: QCD=0, T=1, W=2, Z=3
    """
    out = np.zeros_like(label)
    out[(label == 0) | (label == 1)] = 0  # g, q -> QCD
    out[label == 2] = 1  # t -> T
    out[label == 3] = 2  # w -> W
    out[label == 4] = 3  # z -> Z
    return (out,)


@register_processor
def tanh(x):
    return (np.tanh(x),)


@register_processor
def log(x):
    return (np.log(x + eps),)


@register_processor
def one_hot(*args):
    return tuple(arg + 0.0 for arg in args)


@register_processor
def get_simplified_kinematics(
    part_energy: ak.Array,
    part_px: ak.Array,
    part_py: ak.Array,
    part_pz: ak.Array,
    part_deta: ak.Array,
    part_dphi: ak.Array,
):
    part_pt = np.sqrt(part_px**2 + part_py**2)

    # Use unnormalized values for kinematic calculations
    part_log_pt = np.log(part_pt + eps)

    part_log_energy = np.log(part_energy + eps)

    # Normalizations
    norm_part_log_pt = normalize(
        part_log_pt, _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_part_log_energy = normalize(
        part_log_energy, _NORM["log_energy"]["mean"], _NORM["log_energy"]["std"]
    )

    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])
    norm_part_delta_R = normalize(
        delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]
    )

    # Normalize 4-momentum for output
    norm_part_energy, norm_part_px, norm_part_py, norm_part_pz = normalize_part_p4(
        part_energy, part_px, part_py, part_pz
    )

    return (
        norm_part_energy,
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_log_pt,
        norm_part_log_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
    )


@register_processor
def get_rino_kinematics(
    part_energy: ak.Array,
    part_px: ak.Array,
    part_py: ak.Array,
    part_pz: ak.Array,
    part_deta: ak.Array,
    part_dphi: ak.Array,
    jet_energy: ak.Array,
    jet_pt: ak.Array,
):
    """Kinematics for RINO (kT-clustering SSL).

    Combines absolute log-scale features (jet energy scale) with jet-relative
    log-scale features (cross-view consistency). No Cartesian 4-momentum
    components. See dino/dataloader/processed/processors.py for rationale.

    Outputs (7 features):
        norm_log_pt          — log(pT_i),             normalized by particle stats
        norm_log_energy      — log(E_i),              normalized by particle stats
        norm_log_rel_pt      — log(pT_i / pT_jet),    normalized
        norm_log_rel_energy  — log(E_i / E_jet),      normalized
        norm_part_delta_R    — ΔR from jet axis,      normalized
        norm_part_deta       — Δη from jet axis,      normalized
        norm_part_dphi       — Δφ from jet axis,      normalized
    """
    part_pt = np.sqrt(part_px**2 + part_py**2)

    # ── absolute log scale (jet hardness) ─────────────────────────────────
    norm_log_pt = normalize(
        np.log(part_pt + eps), _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_log_energy = normalize(
        np.log(part_energy + eps),
        _NORM["log_energy"]["mean"],
        _NORM["log_energy"]["std"],
    )

    # ── jet-relative log scale (cross-view consistency) ───────────────────
    norm_log_rel_pt = normalize(
        np.log(part_pt / (jet_pt[:, None] + eps) + eps),
        _NORM["log_rel_pt"]["mean"],
        _NORM["log_rel_pt"]["std"],
    )
    norm_log_rel_energy = normalize(
        np.log(part_energy / (jet_energy[:, None] + eps) + eps),
        _NORM["log_rel_energy"]["mean"],
        _NORM["log_rel_energy"]["std"],
    )

    # ── geometry (already jet-relative) ───────────────────────────────────
    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])
    norm_part_delta_R = normalize(
        delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]
    )

    return (
        norm_log_pt,
        norm_log_energy,
        norm_log_rel_pt,
        norm_log_rel_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
    )


@register_processor
def get_ParT_kinematics(
    part_energy: ak.Array,
    part_px: ak.Array,
    part_py: ak.Array,
    part_pz: ak.Array,
    part_deta: ak.Array,
    part_dphi: ak.Array,
    jet_energy: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
    smear_factor: float = -1,  # Default to no smearing
):
    if smear_factor > 0:
        # Apply smearing to the particle 4-momentum
        part_energy = gaussian_smear(part_energy, smear_factor)
        part_px = gaussian_smear(part_px, smear_factor)
        part_py = gaussian_smear(part_py, smear_factor)
        part_pz = gaussian_smear(part_pz, smear_factor)
        part_p = np.sqrt(part_px**2 + part_py**2 + part_pz**2)
        # Ensure energy is at least the invariant mass
        part_energy = np.clip(part_energy, part_p + eps, None)

        # recompute jet_energy and jet_phi
        jet_energy, jet_pt, jet_eta, jet_phi = get_jetEPtEtaPhi(
            part_px, part_py, part_pz, part_energy
        )

        # recalculate part_pt, part_eta, part_phi
        part_pt = np.sqrt(part_px**2 + part_py**2)
        part_eta = np.arcsinh(part_pz / (part_pt + eps))
        part_phi = np.arctan2(part_py, part_px)
        part_deta = part_eta - jet_eta[:, None]
        part_dphi = part_phi - jet_phi[:, None]
        # Ensure dphi is in [-pi, pi]
        part_dphi = (part_dphi + math.pi) % (2 * math.pi) - math.pi
    else:
        part_pt = np.sqrt(part_px**2 + part_py**2)

    # Use unnormalized values for kinematic calculations
    part_log_pt = np.log(part_pt + eps)
    part_rel_pt = part_pt / (jet_pt + eps)
    part_log_rel_pt = np.log(part_rel_pt + eps)

    part_log_energy = np.log(part_energy + eps)
    part_rel_energy = part_energy / (jet_energy + eps)
    part_log_rel_energy = np.log(part_rel_energy + eps)

    # Normalizations
    norm_part_log_pt = normalize(
        part_log_pt, _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_part_log_energy = normalize(
        part_log_energy, _NORM["log_energy"]["mean"], _NORM["log_energy"]["std"]
    )

    norm_part_log_rel_pt = normalize(
        part_log_rel_pt, _NORM["log_rel_pt"]["mean"], _NORM["log_rel_pt"]["std"]
    )
    norm_part_log_rel_energy = normalize(
        part_log_rel_energy,
        _NORM["log_rel_energy"]["mean"],
        _NORM["log_rel_energy"]["std"],
    )

    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])
    norm_part_delta_R = normalize(
        delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]
    )

    # Normalize 4-momentum for output
    norm_part_energy, norm_part_px, norm_part_py, norm_part_pz = normalize_part_p4(
        part_energy, part_px, part_py, part_pz
    )

    return (
        norm_part_energy,
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_log_pt,
        norm_part_log_energy,
        norm_part_log_rel_pt,
        norm_part_log_rel_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
        jet_energy,
        jet_pt,
        jet_eta,
        jet_phi,
    )


@register_processor
def compute_jet_mass(
    jet_energy: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_sdmass: ak.Array,
):
    """Compute ungroomed jet mass from 4-momentum and pass through soft-drop mass."""
    jet_pz = jet_pt * np.sinh(jet_eta)
    p2 = jet_pt**2 + jet_pz**2
    m2 = np.maximum(jet_energy**2 - p2, 0)
    jet_mass = np.sqrt(m2)
    log_jet_mass = np.log(jet_mass + eps)
    log_jet_sdmass = np.log(jet_sdmass + eps)
    return jet_mass, jet_sdmass, log_jet_mass, log_jet_sdmass


@register_processor
def get_jet_kinematics(
    jet_nparticles: ak.Array,
    jet_energy: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
):
    (norm_jet_nparticles,) = normalize_nparticles(jet_nparticles)
    log_jet_energy = np.log(jet_energy + eps)
    log_jet_pt = np.log(jet_pt + eps)

    norm_log_jet_energy = normalize(
        log_jet_energy, _NORM["jet_log_energy"]["mean"], _NORM["jet_log_energy"]["std"]
    )
    norm_log_jet_pt = normalize(
        log_jet_pt, _NORM["jet_log_pt"]["mean"], _NORM["jet_log_pt"]["std"]
    )
    norm_jet_eta = normalize(jet_eta, _NORM["jet_eta"]["mean"], _NORM["jet_eta"]["std"])
    norm_jet_phi = normalize(jet_phi, _NORM["jet_phi"]["mean"], _NORM["jet_phi"]["std"])

    return (
        norm_jet_nparticles,
        norm_log_jet_energy,
        norm_log_jet_pt,
        norm_jet_eta,
        norm_jet_phi,
    )


@register_processor
def get_part_pt(part_px, part_py):
    part_pt = np.sqrt(part_px**2 + part_py**2)
    return (part_pt,)


@register_processor
def get_part_pt_rel(part_px, part_py, jet_pt):
    part_pt = np.sqrt(part_px**2 + part_py**2)
    part_rel_pt = part_pt / (jet_pt + eps)
    return (part_rel_pt,)


@register_processor
def get_part_E_rel(part_energy, jet_energy):
    part_E_rel = part_energy / (jet_energy + eps)
    return (part_E_rel,)


@register_processor
def get_part_dR(part_deta, part_dphi):
    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    return (delta_R,)


def normalize(x, means, std_devs):
    x = (x - means) / std_devs
    return x


@register_processor
def normalize_part_p4(part_energy, part_px, part_py, part_pz):
    m, s = _NORM["p4"]["mean"], _NORM["p4"]["std"]
    return (
        normalize(part_energy, m, s),
        normalize(part_px, m, s),
        normalize(part_py, m, s),
        normalize(part_pz, m, s),
    )


@register_processor
def normalize_nparticles(jet_nparticles):
    return (
        normalize(
            jet_nparticles,
            _NORM["jet_nparticles"]["mean"],
            _NORM["jet_nparticles"]["std"],
        ),
    )


@register_processor
def part_id_labeler(
    part_isChargedHadron: ak.Array,
    part_isNeutralHadron: ak.Array,
    part_isPhoton: ak.Array,
    part_isElectron: ak.Array,
    part_isMuon: ak.Array,
):
    # Stack and convert to single label
    part_labels = np.stack(
        [
            np.asarray(part_isChargedHadron + 0.0),
            np.asarray(part_isNeutralHadron + 0.0),
            np.asarray(part_isPhoton + 0.0),
            np.asarray(part_isElectron + 0.0),
            np.asarray(part_isMuon + 0.0),
        ],
        axis=-1,
    )
    part_id_label = np.argmax(part_labels, axis=-1)
    return (part_id_label,)


@register_processor
def get_ParT_inputs(
    # Particle kinematics
    part_energy: ak.Array,
    part_px: ak.Array,
    part_py: ak.Array,
    part_pz: ak.Array,
    part_deta: ak.Array,
    part_dphi: ak.Array,
    # Particle properties
    part_d0val: ak.Array,
    part_dzval: ak.Array,
    part_d0err: ak.Array,
    part_dzerr: ak.Array,
    part_isChargedHadron: ak.Array,
    part_isNeutralHadron: ak.Array,
    part_isPhoton: ak.Array,
    part_isElectron: ak.Array,
    part_isMuon: ak.Array,
    # Jet properties
    jet_energy: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
    # smearing factor
    smear_factor: float = -1,
):
    """Process all particle-level inputs for ParT using existing processors."""
    (
        norm_part_energy,
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_log_pt,
        norm_part_log_energy,
        norm_part_rel_pt,
        norm_part_log_rel_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
        jet_energy,
        jet_pt,
        jet_eta,
        jet_phi,
    ) = get_ParT_kinematics(
        part_energy=part_energy,
        part_px=part_px,
        part_py=part_py,
        part_pz=part_pz,
        part_deta=part_deta,
        part_dphi=part_dphi,
        jet_energy=jet_energy,
        jet_pt=jet_pt,
        jet_eta=jet_eta,
        jet_phi=jet_phi,
        smear_factor=smear_factor,
    )

    if smear_factor > 0:
        # Apply smearing to the particle properties
        part_d0val = gaussian_smear(part_d0val, smear_factor)
        part_dzval = gaussian_smear(part_dzval, smear_factor)
        part_d0err = gaussian_smear(part_d0err, smear_factor)
        part_dzerr = gaussian_smear(part_dzerr, smear_factor)

    # Track quality using existing tanh processor
    (tanh_d0,) = tanh(part_d0val)
    (tanh_dz,) = tanh(part_dzval)

    # Particle ID using existing one_hot processor
    (part_id_label,) = part_id_labeler(
        part_isChargedHadron=part_isChargedHadron,
        part_isNeutralHadron=part_isNeutralHadron,
        part_isPhoton=part_isPhoton,
        part_isElectron=part_isElectron,
        part_isMuon=part_isMuon,
    )

    return (
        norm_part_energy,
        norm_part_px,
        norm_part_py,
        norm_part_pz,
        norm_part_log_pt,
        norm_part_log_energy,
        norm_part_rel_pt,
        norm_part_log_rel_energy,
        norm_part_delta_R,
        norm_part_deta,
        norm_part_dphi,
        part_id_label,
        tanh_d0,
        tanh_dz,
        part_d0err,
        part_dzerr,
        jet_energy,
        jet_pt,
        jet_eta,
        jet_phi,
    )


@register_processor
def get_kinematics(
    part_px: ak.Array,
    part_py: ak.Array,
    part_deta: ak.Array,
    part_dphi: ak.Array,
):
    part_pt = np.sqrt(part_px**2 + part_py**2)
    part_log_pt = np.log(part_pt + eps)

    norm_part_log_pt = normalize(
        part_log_pt, _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])

    return (
        norm_part_log_pt,
        norm_part_deta,
        norm_part_dphi,
    )


def run_processors(config: _DataloaderConfig, data: dict[str, ak.Array]):
    """Execute all processors in sequence"""
    for proc_conf in config.transformations:
        proc_fn = processors[proc_conf.processor]
        kwargs = proc_conf.kwargs or {}
        outputs = proc_fn(*proc_conf.args, _data=data, **kwargs)
        for name, output in zip(proc_conf.outputs, outputs):
            data[name] = output
    return data

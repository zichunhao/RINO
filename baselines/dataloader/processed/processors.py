from collections.abc import Callable
from typing import Tuple

import torch
import sys
import math

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


def register_processor(processor: Callable[..., tuple[torch.Tensor, ...]]):
    def wrapper(*args, _data: dict[str, torch.Tensor], **kwargs):
        args = [_data[x] for x in args]
        # Handle special parameters that should be passed as literals
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


def get_jetEPtEtaPhi(
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
    part_energy: torch.Tensor,
):
    # sum up the 4-momentum components
    jet_energy = torch.sum(part_energy, dim=-1)
    jet_px = torch.sum(part_px, dim=-1)
    jet_py = torch.sum(part_py, dim=-1)
    jet_pz = torch.sum(part_pz, dim=-1)
    jet_pt = torch.sqrt(jet_px**2 + jet_py**2)
    jet_eta = torch.arcsinh(jet_pz / (jet_pt + eps))
    jet_phi = torch.arctan2(jet_py, jet_px)

    return jet_energy, jet_pt, jet_eta, jet_phi


@register_processor
def cyl_to_cart(pt: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor):
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return px, py, pz


@register_processor
def gaussian_smear(x: torch.Tensor, smear_factor: float):
    if smear_factor > 0:
        return x * (1 + torch.normal(0, smear_factor, size=x.shape, device=x.device))
    return x


@register_processor
def get_nparticles(part_is_real: torch.Tensor):
    # For toptagging
    nparticles = torch.sum(part_is_real, dim=-1)
    return (nparticles,)


@register_processor
def jetclass_labeler(
    label_QCD: torch.Tensor,
    label_Hbb: torch.Tensor,
    label_Hcc: torch.Tensor,
    label_Hgg: torch.Tensor,
    label_H4q: torch.Tensor,
    label_Hqql: torch.Tensor,
    label_Zqq: torch.Tensor,
    label_Wqq: torch.Tensor,
    label_Tbqq: torch.Tensor,
    label_Tbl: torch.Tensor,
):
    labels = torch.stack(
        [
            label_QCD,
            label_Hbb,
            label_Hcc,
            label_Hgg,
            label_H4q,
            label_Hqql,
            label_Zqq,
            label_Wqq,
            label_Tbqq,
            label_Tbl,
        ],
        dim=1,
    )

    # assert torch.all(torch.sum(labels, dim=1) == 1)
    label = torch.argmax(labels, dim=1)
    return (label,)


@register_processor
def jetclass_toptagging_labeler(
    label_QCD: torch.Tensor,
    label_Tbqq: torch.Tensor,
):
    labels = torch.stack(
        [
            label_QCD,
            label_Tbqq,
        ],
        dim=1,
    )

    # assert torch.all(torch.sum(labels, dim=1) == 1)
    label = torch.argmax(labels, dim=1)
    return (label,)


@register_processor
def jetnet_toptagging_labeler(
    label: torch.Tensor,
):
    label = label.clone()
    # {g, q} -> QCD
    label[(label == 0) | (label == 1)] = 0
    # top
    label[label == 2] = 1

    return (label,)


@register_processor
def tanh(x: torch.Tensor):
    return (torch.tanh(x),)


@register_processor
def log(x: torch.Tensor):
    return (torch.log(x + eps),)


@register_processor
def one_hot(*args: torch.Tensor):
    return tuple(arg.float() for arg in args)


@register_processor
def get_simplified_kinematics(
    part_energy: torch.Tensor,
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
    part_deta: torch.Tensor,
    part_dphi: torch.Tensor,
):
    part_pt = torch.sqrt(part_px**2 + part_py**2)

    part_log_pt = torch.log(part_pt + eps)
    part_log_energy = torch.log(part_energy + eps)

    # Normalizations
    norm_part_log_pt = normalize(
        part_log_pt, _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_part_log_energy = normalize(
        part_log_energy, _NORM["log_energy"]["mean"], _NORM["log_energy"]["std"]
    )

    delta_R = torch.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])
    norm_part_delta_R = normalize(
        delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]
    )

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
    part_energy: torch.Tensor,
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
    part_deta: torch.Tensor,
    part_dphi: torch.Tensor,
    jet_energy: torch.Tensor,
    jet_pt: torch.Tensor,
):
    """Kinematics for RINO (kT-clustering SSL).

    Combines absolute log-scale features (jet energy scale) with jet-relative
    log-scale features (cross-view consistency). The absolute features let the
    model learn what hard vs. soft jets look like; the relative features ensure
    that the same jet at different clustering scales (subjet-K vs. particle view)
    produces comparable inputs, since log(pT_i/pT_jet) is additive under
    kT merging (a subjet's fraction = sum of constituent fractions).

    No Cartesian 4-momentum components: those carry no information beyond
    (pT, geometry) once we have log-pT and (deta, dphi).

    Outputs (7 features):
        norm_log_pt          — log(pT_i),              normalized by particle stats
        norm_log_energy      — log(E_i),               normalized by particle stats
        norm_log_rel_pt      — log(pT_i / pT_jet),     normalized
        norm_log_rel_energy  — log(E_i / E_jet),       normalized
        norm_part_delta_R    — ΔR from jet axis,       normalized
        norm_part_deta       — Δη from jet axis,       normalized
        norm_part_dphi       — Δφ from jet axis,       normalized
    """
    part_pt = torch.sqrt(part_px**2 + part_py**2)
    jet_energy_ = jet_energy.unsqueeze(-1) + eps
    jet_pt_ = jet_pt.unsqueeze(-1) + eps

    # ── absolute log scale (jet hardness) ─────────────────────────────────
    norm_log_pt = normalize(
        torch.log(part_pt + eps), _NORM["log_pt"]["mean"], _NORM["log_pt"]["std"]
    )
    norm_log_energy = normalize(
        torch.log(part_energy + eps),
        _NORM["log_energy"]["mean"],
        _NORM["log_energy"]["std"],
    )

    # ── jet-relative log scale (cross-view consistency) ───────────────────
    norm_log_rel_pt = normalize(
        torch.log(part_pt / jet_pt_ + eps),
        _NORM["log_rel_pt"]["mean"],
        _NORM["log_rel_pt"]["std"],
    )
    norm_log_rel_energy = normalize(
        torch.log(part_energy / jet_energy_ + eps),
        _NORM["log_rel_energy"]["mean"],
        _NORM["log_rel_energy"]["std"],
    )

    # ── geometry (already jet-relative) ───────────────────────────────────
    delta_R = torch.sqrt(part_deta**2 + part_dphi**2)
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
    part_energy: torch.Tensor,
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
    part_deta: torch.Tensor,
    part_dphi: torch.Tensor,
    jet_energy: torch.Tensor,
    jet_pt: torch.Tensor,
    jet_eta: torch.Tensor,
    jet_phi: torch.Tensor,
    smear_factor: float = -1,
):
    if smear_factor > 0:
        part_px = gaussian_smear(part_px, smear_factor)
        part_py = gaussian_smear(part_py, smear_factor)
        part_pz = gaussian_smear(part_pz, smear_factor)

        part_p = torch.sqrt(part_px**2 + part_py**2 + part_pz**2)
        part_energy = gaussian_smear(part_energy, smear_factor)
        # clamp to msq
        part_energy = torch.clamp(part_energy, min=part_p + eps)

        # recompute jet_energy and jet_phi
        jet_energy, jet_pt, jet_eta, jet_phi = get_jetEPtEtaPhi(
            part_px, part_py, part_pz, part_energy
        )

        part_pt = torch.sqrt(part_px**2 + part_py**2)
        part_eta = torch.arcsinh(part_pz / (part_pt + eps))
        part_phi = torch.arctan2(part_py, part_px)

        part_deta = part_eta - jet_eta.unsqueeze(-1)
        part_dphi = part_phi - jet_phi.unsqueeze(-1)
        part_dphi = (part_dphi + math.pi) % (2 * math.pi) - math.pi
    else:
        part_pt = torch.sqrt(part_px**2 + part_py**2)

    part_log_pt = torch.log(part_pt + eps)
    part_rel_pt = part_pt / (jet_pt.unsqueeze(-1) + eps)
    part_log_rel_pt = torch.log(part_rel_pt + eps)

    part_log_energy = torch.log(part_energy + eps)
    part_rel_energy = part_energy / (jet_energy.unsqueeze(-1) + eps)
    part_log_rel_energy = torch.log(part_rel_energy + eps)

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

    delta_R = torch.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, _NORM["deta"]["mean"], _NORM["deta"]["std"])
    norm_part_dphi = normalize(part_dphi, _NORM["dphi"]["mean"], _NORM["dphi"]["std"])
    norm_part_delta_R = normalize(
        delta_R, _NORM["delta_R"]["mean"], _NORM["delta_R"]["std"]
    )

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
def get_jet_kinematics(
    jet_nparticles: torch.Tensor,
    jet_energy: torch.Tensor,
    jet_pt: torch.Tensor,
    jet_eta: torch.Tensor,
    jet_phi: torch.Tensor,
):
    (norm_jet_nparticles,) = normalize_nparticles(jet_nparticles)
    log_jet_energy = torch.log(jet_energy + eps)
    log_jet_pt = torch.log(jet_pt + eps)

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
def get_part_pt(part_px: torch.Tensor, part_py: torch.Tensor):
    part_pt = torch.sqrt(part_px**2 + part_py**2)
    return (part_pt,)


@register_processor
def get_pt(
    part_px: torch.Tensor,
    part_py: torch.Tensor,
):
    # Baselines-specific alias of get_part_pt; preserved because some baselines
    # yamls still reference the older name.
    part_pt = torch.sqrt(part_px**2 + part_py**2)
    return (part_pt,)


@register_processor
def get_part_pt_rel(part_px: torch.Tensor, part_py: torch.Tensor, jet_pt: torch.Tensor):
    part_pt = torch.sqrt(part_px**2 + part_py**2)
    part_rel_pt = part_pt / (jet_pt + eps)
    return (part_rel_pt,)


@register_processor
def get_part_E_rel(part_energy: torch.Tensor, jet_energy: torch.Tensor):
    part_E_rel = part_energy / (jet_energy + eps)
    return (part_E_rel,)


@register_processor
def get_part_dR(part_deta: torch.Tensor, part_dphi: torch.Tensor):
    delta_R = torch.sqrt(part_deta**2 + part_dphi**2)
    return (delta_R,)


def normalize(x: torch.Tensor, means: float, std_devs: float):
    x = (x - means) / std_devs
    return x


@register_processor
def normalize_part_p4(
    part_energy: torch.Tensor,
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
):
    m, s = _NORM["p4"]["mean"], _NORM["p4"]["std"]
    return (
        normalize(part_energy, m, s),
        normalize(part_px, m, s),
        normalize(part_py, m, s),
        normalize(part_pz, m, s),
    )


@register_processor
def normalize_nparticles(jet_nparticles: torch.Tensor):
    return (
        normalize(
            jet_nparticles,
            _NORM["jet_nparticles"]["mean"],
            _NORM["jet_nparticles"]["std"],
        ),
    )


@register_processor
def get_ParT_inputs(
    # Particle get_ParT_kinematics
    part_energy: torch.Tensor,
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_pz: torch.Tensor,
    part_deta: torch.Tensor,
    part_dphi: torch.Tensor,
    # Particle properties
    part_d0val: torch.Tensor,
    part_dzval: torch.Tensor,
    part_d0err: torch.Tensor,
    part_dzerr: torch.Tensor,
    part_isChargedHadron: torch.Tensor,
    part_isNeutralHadron: torch.Tensor,
    part_isPhoton: torch.Tensor,
    part_isElectron: torch.Tensor,
    part_isMuon: torch.Tensor,
    # Jet properties
    jet_energy: torch.Tensor,
    jet_pt: torch.Tensor,
    jet_eta: torch.Tensor,
    jet_phi: torch.Tensor,
    # smearing factor
    smear_factor: float = -1,
):
    """Process all particle-level inputs for ParT using existing processors."""
    # get_ParT_kinematics features using existing processor
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
    (
        part_isChargedHadron,
        part_isNeutralHadron,
        part_isPhoton,
        part_isElectron,
        part_isMuon,
    ) = one_hot(
        part_isChargedHadron,
        part_isNeutralHadron,
        part_isPhoton,
        part_isElectron,
        part_isMuon,
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
        part_isChargedHadron,
        part_isNeutralHadron,
        part_isPhoton,
        part_isElectron,
        part_isMuon,
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
    part_px: torch.Tensor,
    part_py: torch.Tensor,
    part_deta: torch.Tensor,
    part_dphi: torch.Tensor,
):
    part_pt = torch.sqrt(part_px**2 + part_py**2)
    part_log_pt = torch.log(part_pt + eps)

    norm_part_log_pt = normalize(part_log_pt, means=1.5, std_devs=1.5)
    norm_part_deta = normalize(part_deta, means=0.0, std_devs=0.3)
    norm_part_dphi = normalize(part_dphi, means=0.0, std_devs=0.3)

    return (
        norm_part_log_pt,
        norm_part_deta,
        norm_part_dphi,
    )


def run_processors(config: _DataloaderConfig, data: dict[str, torch.Tensor]):
    """Execute all processors in sequence"""
    for proc_conf in config.transformations:
        proc_fn = processors[proc_conf.processor]
        kwargs = proc_conf.kwargs or {}
        outputs = proc_fn(*proc_conf.args, _data=data, **kwargs)
        for name, output in zip(proc_conf.outputs, outputs):
            data[name] = output
    return data

from collections.abc import Callable

import awkward as ak
import numpy as np
import math
import sys

from .config import _DataloaderConfig

processors = {}
eps = sys.float_info.epsilon


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
    label_QCD: ak.Array,
    label_Hbb: ak.Array,
    label_Hcc: ak.Array,
    label_Hgg: ak.Array,
    label_H4q: ak.Array,
    label_Hqql: ak.Array,
    label_Zqq: ak.Array,
    label_Wqq: ak.Array,
    label_Tbqq: ak.Array,
    label_Tbl: ak.Array,
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
def tanh(x):
    return (np.tanh(x),)


@register_processor
def log(x):
    return (np.log(x + eps),)


@register_processor
def one_hot(*args):
    return tuple(arg + 0.0 for arg in args)


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
    smear_factor: float = -1, # Default to no smearing
):
    if smear_factor > 0:
        # Apply smearing to the particle 4-momentum
        part_energy = gaussian_smear(part_energy, smear_factor)
        part_px = gaussian_smear(part_px, smear_factor)
        part_py = gaussian_smear(part_py, smear_factor)
        part_pz = gaussian_smear(part_pz, smear_factor)
        part_msq = np.sqrt(part_px**2 + part_py**2 + part_pz**2)
        # Ensure energy is at least the invariant mass
        part_energy = np.clip(part_energy, part_msq + eps, None)
        
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
    norm_part_log_pt = normalize(part_log_pt, means=1.5, std_devs=1.5)
    norm_part_log_energy = normalize(part_log_energy, means=1.5, std_devs=1.5)
    
    norm_part_log_rel_pt = normalize(part_log_rel_pt, means=-4.5, std_devs=1.5)
    norm_part_log_rel_energy = normalize(part_log_rel_energy, means=-5, std_devs=1.5)

    delta_R = np.sqrt(part_deta**2 + part_dphi**2)
    norm_part_deta = normalize(part_deta, means=0.0, std_devs=0.3)
    norm_part_dphi = normalize(part_dphi, means=0.0, std_devs=0.3)
    norm_part_delta_R = normalize(delta_R, means=0.3, std_devs=0.2)

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
def get_jet_kinematics(
    jet_nparticles: ak.Array,
    jet_energy: ak.Array,
    jet_pt: ak.Array,
    jet_eta: ak.Array,
    jet_phi: ak.Array,
):
    norm_jet_nparticles, = normalize_nparticles(jet_nparticles)
    log_jet_energy = np.log(jet_energy + eps)
    log_jet_pt = np.log(jet_pt + eps)

    norm_log_jet_energy = normalize(log_jet_energy, means=7.0, std_devs=1.0)
    norm_log_jet_pt = normalize(log_jet_pt, means=6.5, std_devs=0.5)
    norm_jet_eta = normalize(jet_eta, means=0.0, std_devs=2.0)
    norm_jet_phi = normalize(jet_phi, means=0.0, std_devs=math.pi)

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
    means = 0.0
    std_devs = 25.0
    part_energy = normalize(part_energy, means, std_devs)
    part_px = normalize(part_px, means, std_devs)
    part_py = normalize(part_py, means, std_devs)
    part_pz = normalize(part_pz, means, std_devs)
    return part_energy, part_px, part_py, part_pz


@register_processor
def normalize_nparticles(jet_nparticles):
    means = 40.0
    std_devs = 15.0
    return (normalize(jet_nparticles, means, std_devs),)


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


def run_processors(config: _DataloaderConfig, data: dict[str, ak.Array]):
    """Execute all processors in sequence"""
    for proc_conf in config.transformations:
        proc_fn = processors[proc_conf.processor]
        kwargs = proc_conf.kwargs or {}
        outputs = proc_fn(*proc_conf.args, _data=data, **kwargs)
        for name, output in zip(proc_conf.outputs, outputs):
            data[name] = output
    return data

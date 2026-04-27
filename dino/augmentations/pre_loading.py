"""Per-view pre-loading augmentations for DINO.

Each view in the DINO multi-crop pipeline specifies a single augmentation
type (jetclr or lorentz). These functions take raw particle 4-momenta and
jet 4-momentum from the dataloader, apply one augmentation family, and emit
the 7 RINO-normalized features matching the default pipeline's ``sequence``
output.

Composition is intentionally NOT supported: each view carries one aug.
"""

import sys
import torch
import numpy as np

from .boost import get_random_boost_matrices
from .rotation import get_random_rotation_matrices

EPS = sys.float_info.epsilon

# RINO normalization statistics (JetClass training split, all classes).
# Must match dino/dataloader/jetclass/processors.py:_NORM.
_RINO_NORM = {
    "log_pt": {"mean": 1.7, "std": 1.8},
    "log_energy": {"mean": 2.0, "std": 1.8},
    "log_rel_pt": {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "deta": {"mean": 0.0, "std": 0.14},
    "dphi": {"mean": 0.0, "std": 0.14},
    "delta_R": {"mean": 0.14, "std": 0.25},
}


# ---------- JetCLR augmentations (operate on pT, deta, dphi; energy passive) ----------


def rotate_jets(pt, energy, deta, dphi):
    """Random rotation in eta-phi plane (shared per jet)."""
    B = pt.shape[0]
    theta = torch.rand(B, device=pt.device, dtype=pt.dtype) * 2 * np.pi
    c = torch.cos(theta).unsqueeze(1)
    s = torch.sin(theta).unsqueeze(1)
    return pt, energy, c * deta - s * dphi, s * deta + c * dphi


def distort_jets(pt, energy, deta, dphi, strength=0.1, pT_clip_min=0.1):
    """Smear eta/phi with noise inversely proportional to pT."""
    denom = pt.clamp(min=pT_clip_min)
    noise_eta = torch.nan_to_num(
        strength * torch.randn_like(pt) / denom, posinf=0.0, neginf=0.0
    )
    noise_phi = torch.nan_to_num(
        strength * torch.randn_like(pt) / denom, posinf=0.0, neginf=0.0
    )
    return pt, energy, deta + noise_eta, dphi + noise_phi


def collinear_fill_jets(pt, energy, deta, dphi, mask):
    """Split random real particles collinearly into empty slots (IRC-safe).

    Splits pt and energy by random fraction r and (1-r); deta/dphi are copied.
    """
    B, N = pt.shape
    device = pt.device

    is_nz = mask & (pt > 0.0)
    nz = is_nz.sum(dim=1)
    zs = N - nz
    n_splits = torch.minimum(zs, nz)

    if int(n_splits.max().item()) == 0:
        return pt, energy, deta, dphi

    max_s = int(n_splits.max().item())

    # Pick which real particles to split.
    noise = torch.rand(B, N, device=device)
    noise = noise.masked_fill(~is_nz, 1e9)
    perm = noise.argsort(dim=1)
    sel = perm[:, :max_s]  # (B, max_s) indices

    valid = torch.arange(max_s, device=device).unsqueeze(0) < n_splits.unsqueeze(1)

    sel_pt = torch.gather(pt, 1, sel)
    sel_energy = torch.gather(energy, 1, sel)
    sel_deta = torch.gather(deta, 1, sel)
    sel_dphi = torch.gather(dphi, 1, sel)

    rs = torch.rand(B, max_s, device=device, dtype=pt.dtype)
    scale = torch.where(valid, rs, torch.ones_like(rs))

    # Scale originals down by r.
    scale_full_pt = torch.ones(B, N, device=device, dtype=pt.dtype)
    scale_full_pt.scatter_(1, sel, scale)
    pt_out = pt * scale_full_pt
    energy_out = energy * scale_full_pt

    # New particles at positions nz, nz+1, ... with (1-r)*pt.
    new_pos = (
        nz.unsqueeze(1) + torch.arange(max_s, device=device).unsqueeze(0)
    ).clamp(max=N - 1)
    valid_f = valid.to(pt.dtype)
    new_pt = (1.0 - rs) * sel_pt * valid_f
    new_energy = (1.0 - rs) * sel_energy * valid_f
    new_deta = sel_deta * valid_f
    new_dphi = sel_dphi * valid_f

    pt_out = pt_out.scatter(1, new_pos, new_pt)
    energy_out = energy_out.scatter(1, new_pos, new_energy)
    deta_out = deta.scatter(1, new_pos, new_deta)
    dphi_out = dphi.scatter(1, new_pos, new_dphi)

    return pt_out, energy_out, deta_out, dphi_out


def translate_jets(pt, energy, deta, dphi, mask, width=1.0):
    """Shift jets in eta/phi by up to ``width`` times their eta/phi extent."""
    B, N = pt.shape
    mask_f = mask.to(pt.dtype)

    # Ignore padded entries when computing the extent.
    big = torch.full_like(deta, 1e9)
    small = torch.full_like(deta, -1e9)
    deta_min = torch.where(mask, deta, big).min(dim=1, keepdim=True)[0]
    deta_max = torch.where(mask, deta, small).max(dim=1, keepdim=True)[0]
    dphi_min = torch.where(mask, dphi, big).min(dim=1, keepdim=True)[0]
    dphi_max = torch.where(mask, dphi, small).max(dim=1, keepdim=True)[0]
    ptp_eta = (deta_max - deta_min).clamp(min=0.0)
    ptp_phi = (dphi_max - dphi_min).clamp(min=0.0)

    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.maximum(-width * ptp_phi, -np.pi - dphi_min)
    high_phi = torch.minimum(+width * ptp_phi, +np.pi - dphi_max)

    shift_eta = torch.rand(B, 1, device=pt.device, dtype=pt.dtype) * (
        high_eta - low_eta
    ) + low_eta
    shift_phi = torch.rand(B, 1, device=pt.device, dtype=pt.dtype) * (
        high_phi - low_phi
    ) + low_phi
    return pt, energy, deta + shift_eta * mask_f, dphi + shift_phi * mask_f


# ---------- RINO normalization ----------


def _to_rino(pt, energy, deta, dphi, jet_pt, jet_energy, mask):
    """Map (pT, energy, deta, dphi) → 7 RINO-normalized features."""
    eps = 1e-8
    N = _RINO_NORM
    jet_pt = jet_pt.unsqueeze(1) if jet_pt.dim() == 1 else jet_pt
    jet_energy = jet_energy.unsqueeze(1) if jet_energy.dim() == 1 else jet_energy

    log_pt = (torch.log(pt.clamp(min=eps)) - N["log_pt"]["mean"]) / N["log_pt"]["std"]
    log_energy = (
        torch.log(energy.clamp(min=eps)) - N["log_energy"]["mean"]
    ) / N["log_energy"]["std"]
    log_rel_pt = (
        torch.log((pt / jet_pt.clamp(min=eps)).clamp(min=eps))
        - N["log_rel_pt"]["mean"]
    ) / N["log_rel_pt"]["std"]
    log_rel_energy = (
        torch.log((energy / jet_energy.clamp(min=eps)).clamp(min=eps))
        - N["log_rel_energy"]["mean"]
    ) / N["log_rel_energy"]["std"]

    delta_R = torch.sqrt(deta**2 + dphi**2)
    norm_delta_R = (delta_R - N["delta_R"]["mean"]) / N["delta_R"]["std"]
    norm_deta = (deta - N["deta"]["mean"]) / N["deta"]["std"]
    norm_dphi = (dphi - N["dphi"]["mean"]) / N["dphi"]["std"]

    out = torch.stack(
        [log_pt, log_energy, log_rel_pt, log_rel_energy, norm_delta_R, norm_deta, norm_dphi],
        dim=-1,
    )
    return out * mask.unsqueeze(-1).to(out.dtype)


def _p4_to_cyl(raw_p4):
    """Raw (E, px, py, pz) → (pT, energy, eta, phi)."""
    E = raw_p4[..., 0]
    px = raw_p4[..., 1]
    py = raw_p4[..., 2]
    pz = raw_p4[..., 3]
    pt = torch.sqrt(px * px + py * py).clamp(min=EPS)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    return pt, E, eta, phi


# ---------- Per-view entry points ----------


def apply_jetclr_view(
    raw_p4: torch.Tensor,
    jet_p4: torch.Tensor,
    mask: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """Apply one stochastic JetCLR augmentation and return 7 RINO features.

    Args:
        raw_p4: (B, N, 4) particle 4-momenta (E, px, py, pz), unnormalized.
        jet_p4: (B, 4) jet 4-momentum (E, px, py, pz), unnormalized.
        mask: (B, N) bool.
        cfg: {"rot": bool, "cf": bool, "ptd": bool, "ptst": float,
              "ptcm": float, "trs": bool, "trsw": float}.
    """
    pt, energy, eta, phi = _p4_to_cyl(raw_p4)
    jet_pt, jet_energy, jet_eta, jet_phi = _p4_to_cyl(jet_p4)

    deta = eta - jet_eta.unsqueeze(1)
    dphi = phi - jet_phi.unsqueeze(1)
    dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi

    mask_f = mask.to(pt.dtype)
    pt = pt * mask_f
    energy = energy * mask_f
    deta = deta * mask_f
    dphi = dphi * mask_f

    if cfg.get("rot", True):
        pt, energy, deta, dphi = rotate_jets(pt, energy, deta, dphi)

    if cfg.get("cf", True):
        pt, energy, deta, dphi = collinear_fill_jets(pt, energy, deta, dphi, mask)

    if cfg.get("ptd", True):
        pt, energy, deta, dphi = distort_jets(
            pt, energy, deta, dphi,
            strength=cfg.get("ptst", 0.1),
            pT_clip_min=cfg.get("ptcm", 0.1),
        )

    if cfg.get("trs", False):
        pt, energy, deta, dphi = translate_jets(
            pt, energy, deta, dphi, mask, width=cfg.get("trsw", 1.0)
        )

    return _to_rino(pt, energy, deta, dphi, jet_pt, jet_energy, mask)


def apply_lorentz_view(
    raw_p4: torch.Tensor,
    jet_p4: torch.Tensor,
    mask: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """Apply one random (full 3D) Lorentz transformation and return 7 RINO features.

    Applies the same Lorentz matrix to every particle in a jet (and to the
    jet's own 4-momentum), then recomputes (pT, energy, deta, dphi) relative
    to the boosted jet axis.
    """
    B, N, _ = raw_p4.shape
    device, dtype = raw_p4.device, raw_p4.dtype

    # Restricted Lorentz = B · R is the minimal decomposition (any proper
    # orthochronous Lorentz transformation factors this way). The extra
    # left-rotation in get_random_lorentz_matrices is redundant.
    R = get_random_rotation_matrices(N=B, device=device, dtype=dtype)
    Bm = get_random_boost_matrices(
        N=B,
        sigma=cfg.get("sigma", 0.3),
        sample_mode=cfg.get("sample_mode", "beta"),
        beta_min=cfg.get("beta_min", -0.99),
        beta_max=cfg.get("beta_max", 0.99),
        device=device,
        dtype=dtype,
    )
    matrices = torch.bmm(Bm, R)  # (B, 4, 4) — rotate first, then boost

    # Boost particles and jet with the same matrix.
    p4_new = torch.einsum("bij,bnj->bni", matrices, raw_p4)
    jet_p4_new = torch.einsum("bij,bj->bi", matrices, jet_p4)

    # Mask pad rows so padded entries stay at zero after the linear transform.
    p4_new = p4_new * mask.unsqueeze(-1).to(p4_new.dtype)

    pt, energy, eta, phi = _p4_to_cyl(p4_new)
    jet_pt, jet_energy, jet_eta, jet_phi = _p4_to_cyl(jet_p4_new)

    deta = eta - jet_eta.unsqueeze(1)
    dphi = phi - jet_phi.unsqueeze(1)
    dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi

    mask_f = mask.to(pt.dtype)
    pt = pt * mask_f
    energy = energy * mask_f
    deta = deta * mask_f
    dphi = dphi * mask_f

    return _to_rino(pt, energy, deta, dphi, jet_pt, jet_energy, mask)

import sys
import numpy as np

import torch

EPS = sys.float_info.epsilon

# RINO normalization statistics (JetClass training split, all classes).
# Matches dino/dataloader/jetclass/processors.py:_NORM.
_RINO_NORM = {
    "log_pt": {"mean": 1.7, "std": 1.8},
    "log_energy": {"mean": 2.0, "std": 1.8},
    "log_rel_pt": {"mean": -4.7, "std": 1.8},
    "log_rel_energy": {"mean": -4.7, "std": 1.8},
    "deta": {"mean": 0.0, "std": 0.14},
    "dphi": {"mean": 0.0, "std": 0.14},
    "delta_R": {"mean": 0.14, "std": 0.25},
}


def translate_jets(batch, width=1.0, pt_idx=0, eta_idx=1, phi_idx=2):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    dim pt_idx: pT, dim eta_idx: eta, dim phi_idx: phi
    Output: batch of eta-phi translated jets, same shape as input
    """
    mask = batch[:, :, pt_idx] > 0  # (batchsize, n_constit) - 1 for non-zero pT

    ptp_eta = (
        batch[:, :, eta_idx].max(dim=-1, keepdim=True)[0]
        - batch[:, :, eta_idx].min(dim=-1, keepdim=True)[0]
    )  # (batchsize, 1)
    ptp_phi = (
        batch[:, :, phi_idx].max(dim=-1, keepdim=True)[0]
        - batch[:, :, phi_idx].min(dim=-1, keepdim=True)[0]
    )  # (batchsize, 1)

    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.maximum(
        -width * ptp_phi, -np.pi - batch[:, :, phi_idx].min(dim=1, keepdim=True)[0]
    )
    high_phi = torch.minimum(
        +width * ptp_phi, +np.pi - batch[:, :, phi_idx].max(dim=1, keepdim=True)[0]
    )

    shift_eta = (
        torch.rand(batch.shape[0], 1, device=batch.device) * (high_eta - low_eta)
        + low_eta
    )  # (batchsize, 1)
    shift_phi = (
        torch.rand(batch.shape[0], 1, device=batch.device) * (high_phi - low_phi)
        + low_phi
    )  # (batchsize, 1)

    out = batch.clone()
    out[:, :, eta_idx] += (shift_eta.expand(-1, batch.shape[1]) * mask.float())
    out[:, :, phi_idx] += (shift_phi.expand(-1, batch.shape[1]) * mask.float())
    return out


def rotate_jets(batch, pt_idx=0, eta_idx=1, phi_idx=2):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    dim pt_idx: pT, dim eta_idx: eta, dim phi_idx: phi
    Output: batch of jets rotated independently in eta-phi, same shape as input
    """
    rot_angle = torch.rand(batch.shape[0], device=batch.device) * 2 * np.pi
    c = torch.cos(rot_angle).view(-1, 1, 1)  # (batchsize, 1, 1)
    s = torch.sin(rot_angle).view(-1, 1, 1)

    eta = batch[:, :, eta_idx : eta_idx + 1]  # (batchsize, n_constit, 1)
    phi = batch[:, :, phi_idx : phi_idx + 1]

    eta_rot = c * eta - s * phi
    phi_rot = s * eta + c * phi

    out = batch.clone()
    out[:, :, eta_idx : eta_idx + 1] = eta_rot
    out[:, :, phi_idx : phi_idx + 1] = phi_rot
    return out


def normalise_pts(batch, pt_idx=0):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    """
    batch_norm = batch.clone()
    pt_sum = batch_norm[:, :, pt_idx].sum(dim=1, keepdim=True)  # (batchsize, 1)
    batch_norm[:, :, pt_idx] = torch.nan_to_num(
        batch_norm[:, :, pt_idx] / (pt_sum + EPS), posinf=0.0, neginf=0.0
    )
    return batch_norm


def rescale_pts(batch, pt_idx=0):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    """
    out = batch.clone()
    out[:, :, pt_idx] = batch[:, :, pt_idx] / 600.0
    return out


def crop_jets(batch, nc):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, nc, D)
    """
    return batch[:, 0:nc, :].clone()


def distort_jets(batch, strength=0.1, pT_clip_min=0.1, pt_idx=0, eta_idx=1, phi_idx=2):
    """
    Input: batch of jets, shape (batchsize, n_constit, D)
    Output: batch of jets with each constituent's position shifted independently,
            shifts drawn from normal with mean 0, std strength/pT, same shape as input
    """
    pT = batch[:, :, pt_idx]  # (batchsize, n_constit)

    shift_eta = torch.nan_to_num(
        strength
        * torch.randn(batch.shape[0], batch.shape[1], device=batch.device)
        / pT.clamp(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )

    shift_phi = torch.nan_to_num(
        strength
        * torch.randn(batch.shape[0], batch.shape[1], device=batch.device)
        / pT.clamp(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )

    out = batch.clone()
    out[:, :, eta_idx] += shift_eta
    out[:, :, phi_idx] += shift_phi
    return out


def collinear_fill_jets(batch, pt_idx=0, split_idxs=None):
    """
    Fully vectorized collinear fill — no Python loops over the batch.

    Input: batch of jets, shape (batchsize, n_constit, D)
    dim pt_idx ordering: pT used for real-particle detection and splitting.
    split_idxs: additional feature indices to split by the same ratio as pT
                (e.g., [1] for energy). These features are scaled by r for the
                original particle and (1-r) for the new collinear daughter.
    Output: batch with collinear splittings filling zero-padded entries.

    For each jet:
      - Count non-zero (real) constituents nz and zero-padded slots zs = N - nz.
      - Randomly select k = min(nz, zs) real constituents to split.
      - Each selected constituent's pT (and split_idxs features) is split by
        a random ratio r:
          original slot  → r  * val  (same eta/phi)
          new zero slot  → (1-r) * val  (same eta/phi, collinear)
    """
    if split_idxs is None:
        split_idxs = []

    device = batch.device
    B, N, D = batch.shape

    is_nz = batch[:, :, pt_idx] > 0.0  # (B, N) — True for real particles
    nz = is_nz.sum(dim=1)  # (B,)  — number of real particles
    zs = N - nz  # (B,)  — number of empty slots
    n_splits = torch.minimum(zs, nz)  # (B,)  — splits to perform

    if n_splits.max() == 0:
        return batch.clone()

    max_s = int(n_splits.max().item())
    out = batch.clone()

    # --- Random permutation of real-particle indices (argsort trick) ---
    noise = torch.rand(B, N, device=device)
    noise.masked_fill_(~is_nz, 1e9)
    perm = noise.argsort(dim=1)  # (B, N): real particles first, pads last

    sel = perm[:, :max_s]  # (B, max_s) — constituent indices to split

    # Validity mask: is split i active for jet b?
    valid = torch.arange(max_s, device=device).unsqueeze(0) < n_splits.unsqueeze(
        1
    )  # (B, max_s)

    # --- Gather features of selected constituents ---
    sel_D = sel.unsqueeze(-1).expand(-1, -1, D)
    sel_feats = torch.gather(batch, 1, sel_D)  # (B, max_s, D)

    # --- Random split ratios ---
    rs = torch.rand(B, max_s, device=device)  # (B, max_s)

    # --- Scale pT (and split_idxs features) of original selected particles ---
    all_split_idxs = [pt_idx] + list(split_idxs)
    scale_vals = torch.where(valid, rs, torch.ones_like(rs))  # (B, max_s)
    for idx in all_split_idxs:
        feat_scale = torch.ones(B, N, device=device)
        feat_scale.scatter_(1, sel, scale_vals)
        out[:, :, idx] = batch[:, :, idx] * feat_scale

    # --- Place new collinear constituents in zero-padded slots ---
    new_pos = (
        nz.unsqueeze(1) + torch.arange(max_s, device=device).unsqueeze(0)
    ).clamp(max=N - 1)  # (B, max_s)

    new_feats = sel_feats.clone()
    # Split features: (1-r) * original value
    for idx in all_split_idxs:
        new_feats[:, :, idx] = (1.0 - rs) * sel_feats[:, :, idx]
    # Zero out invalid splits across all features
    new_feats = new_feats * valid.unsqueeze(-1).float()

    new_pos_D = new_pos.unsqueeze(-1).expand(-1, -1, D)
    out.scatter_(1, new_pos_D, new_feats)

    return out


def rino_post_normalize(
    x: torch.Tensor,
    jet_pt: torch.Tensor,
    jet_energy: torch.Tensor,
    mask: torch.Tensor,
    pt_idx: int = 0,
    energy_idx: int = 1,
    eta_idx: int = 2,
    phi_idx: int = 3,
) -> torch.Tensor:
    """Post-augmentation normalization producing 7 RINO-compatible features.

    Takes raw (pT, energy, deta, dphi) after augmentation and computes:
        0: norm_log_pt          — log(pT),             z-scored
        1: norm_log_energy      — log(E),              z-scored
        2: norm_log_rel_pt      — log(pT / jet_pT),    z-scored
        3: norm_log_rel_energy  — log(E / jet_E),      z-scored
        4: norm_delta_R         — sqrt(deta^2+dphi^2),  z-scored
        5: norm_deta            — deta,                z-scored
        6: norm_dphi            — dphi,                z-scored

    Args:
        x: (B, N, D) raw features after augmentation.
        jet_pt: (B,) jet transverse momentum.
        jet_energy: (B,) jet energy.
        mask: (B, N) boolean mask (True for real particles).
        pt_idx, energy_idx, eta_idx, phi_idx: feature channel indices in x.

    Returns:
        (B, N, 7) normalized features, zero-masked for padding.
    """
    eps = 1e-8
    N = _RINO_NORM

    pt = x[:, :, pt_idx]
    energy = x[:, :, energy_idx]
    deta = x[:, :, eta_idx]
    dphi = x[:, :, phi_idx]

    # Ensure jet quantities are (B, 1) for broadcasting
    if jet_pt.dim() == 1:
        jet_pt = jet_pt.unsqueeze(1)
    if jet_energy.dim() == 1:
        jet_energy = jet_energy.unsqueeze(1)

    # Absolute log-scale features
    log_pt = (torch.log(pt.clamp(min=eps)) - N["log_pt"]["mean"]) / N["log_pt"]["std"]
    log_energy = (
        torch.log(energy.clamp(min=eps)) - N["log_energy"]["mean"]
    ) / N["log_energy"]["std"]

    # Jet-relative log-scale features
    log_rel_pt = (
        torch.log((pt / jet_pt.clamp(min=eps)).clamp(min=eps))
        - N["log_rel_pt"]["mean"]
    ) / N["log_rel_pt"]["std"]
    log_rel_energy = (
        torch.log((energy / jet_energy.clamp(min=eps)).clamp(min=eps))
        - N["log_rel_energy"]["mean"]
    ) / N["log_rel_energy"]["std"]

    # Geometry
    delta_R = torch.sqrt(deta**2 + dphi**2)
    norm_delta_R = (delta_R - N["delta_R"]["mean"]) / N["delta_R"]["std"]
    norm_deta = (deta - N["deta"]["mean"]) / N["deta"]["std"]
    norm_dphi = (dphi - N["dphi"]["mean"]) / N["dphi"]["std"]

    out = torch.stack(
        [log_pt, log_energy, log_rel_pt, log_rel_energy, norm_delta_R, norm_deta, norm_dphi],
        dim=-1,
    )  # (B, N, 7)

    # Zero out padding positions
    out = out * mask.unsqueeze(-1).float()
    return out

"""Positional-encoding bin-collision rates and the r_scale / r_max derivation.

Backs the appendix table ``tab:pe-collision`` and the polar-PE parameters in
"Architecture Details, Positional encoding".

The script loads (dEta, dPhi) of every constituent, standardizes them with the
training normalization (``_NORM`` in ``dino/dataloader/jetclass/processors.py``)
and truncates each jet to ``--max-particles`` (the pretraining
``max_seq_length``). It then

1. prints percentiles of the standardized radius r = sqrt(dEta^2 + dPhi^2):
   ``r_scale`` is the median and ``r_max`` the 99th percentile;
2. assigns every constituent a patch index for each scheme and grid size with
   the release encoders (``BinnedFeatures`` for Cartesian and Log-Cartesian,
   ``PolarBinnedFeatures`` for Polar), so the bins are exactly the training
   ones. Cartesian and Log-Cartesian grids span the same square
   [-b, b]^2, where b is the larger 99th percentile of |dEta| and |dPhi|;
   Log-Cartesian applies sign(x) log(1 + |x| / (r_scale / 2)) before binning;
3. reports, per scheme and grid,
   - per-particle rate: fraction of constituents sharing a patch with at least
     one other constituent of the same jet;
   - per-jet rate: fraction of jets with at least one such collision;
   - mask-mask rate: fraction of masked constituents sharing a patch with
     another masked constituent of the same jet. Each jet with n >= 2
     constituents has k of them masked uniformly at random, with
     ``--mask-count round``: k = round(ratio * n) (an exact 25% mask, as in
     the paper table), or ``floor``: k = int(ratio * n) (the iBOT training
     rule of ``iBOTLoss.create_particle_mask``); k is clamped to [1, n - 1].
     Masks are drawn once (seeded) and shared by all schemes.

Example (the paper uses the 5x10^5 QCD jets of the JetClass validation split)::

    python studies/physics/pe_collision.py \\
        --inputs "PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root" \\
        --output-dir PROJECT_ROOT/experiments/studies/physics/pe_collision
"""

from __future__ import annotations

import argparse
import csv

import awkward as ak
import numpy as np
import torch

from _common import (
    expand_inputs,
    load_particle_arrays,
    load_release_module,
    resolve_path,
    write_json,
)
from dataloader.jetclass.processors import _NORM

# The PE module used by the backbone (dino/models/positional_encoding.py).
_pe = load_release_module(
    "models/positional_encoding.py", "release_positional_encoding"
)
BinnedFeatures, PolarBinnedFeatures = _pe.BinnedFeatures, _pe.PolarBinnedFeatures

SCHEMES = ("cartesian", "log-cartesian", "polar")
SCHEME_LABELS = {
    "cartesian": "Cartesian",
    "log-cartesian": "Log-Cartesian",
    "polar": "Polar",
}
METRICS = {
    "per_particle": "Per-particle collision rate (%)",
    "per_jet": "Per-jet collision rate (%)",
    "mask_mask": "Mask-mask collision rate (% of masked particles)",
}


def load_standardized_coords(
    files, max_jets: int | None, max_particles: int, h5_label: int | None
) -> tuple[ak.Array, ak.Array]:
    """Jagged standardized (dEta, dPhi), truncated to ``max_particles``."""
    arrays = load_particle_arrays(
        files, ["part_deta", "part_dphi"], max_jets=max_jets, h5_label=h5_label
    )
    deta = arrays.part_deta[:, :max_particles]
    dphi = arrays.part_dphi[:, :max_particles]
    deta = (deta - _NORM["deta"]["mean"]) / _NORM["deta"]["std"]
    dphi = (dphi - _NORM["dphi"]["mean"]) / _NORM["dphi"]["std"]
    return deta, dphi


def coordinate_percentiles(deta: np.ndarray, dphi: np.ndarray) -> dict:
    """Percentiles of the radius and of |dEta|, |dPhi| over all constituents."""
    radial = [10, 25, 50, 75, 90, 95, 99, 99.9]
    axis = [50, 75, 90, 95, 99, 99.9]

    def table(values: np.ndarray, levels: list[float]) -> dict[str, float]:
        return dict(zip(map(str, levels), map(float, np.percentile(values, levels))))

    return {
        "radius": table(np.hypot(deta, dphi), radial),
        "abs_deta": table(np.abs(deta), axis),
        "abs_dphi": table(np.abs(dphi), axis),
    }


def build_encoder(scheme: str, n_bins: int, params: dict) -> torch.nn.Module:
    """Release PE module whose ``forward`` returns each particle's patch index.

    The learned lookup table is replaced by the identity (row i holds the value
    i), so the module's own binning code produces the index.
    """
    bound = params["axis_bound"]
    if scheme == "cartesian":
        encoder = BinnedFeatures(out_features=1, bins=[[-bound, bound, n_bins]] * 2)
        table = encoder.embedding
    elif scheme == "log-cartesian":
        encoder = BinnedFeatures(
            out_features=1,
            bins=[[-bound, bound, n_bins]] * 2,
            log_bins=True,
            log_scale=params["log_scale"],
        )
        table = encoder.embedding
    elif scheme == "polar":
        encoder = PolarBinnedFeatures(
            input_indices=[0, 1],
            out_features=1,
            n_r_bins=n_bins,
            n_phi_bins=n_bins,
            r_max=params["r_max"],
            r_scale=params["r_scale"],
        )
        table = encoder._binned.embedding
    else:
        raise ValueError(f"Unknown scheme {scheme!r}; choose from {SCHEMES}")
    with torch.no_grad():
        table.weight.copy_(
            torch.arange(table.num_embeddings, dtype=table.weight.dtype)[:, None]
        )
    return encoder.eval()


@torch.no_grad()
def patch_indices(encoder: torch.nn.Module, coords: torch.Tensor) -> np.ndarray:
    """Patch index of every particle; ``coords`` has shape ``(P, 2)``."""
    return encoder(coords[None]).reshape(-1).round().long().numpy()


def sample_masks(
    n_particles: np.ndarray, mask_ratio: float, mask_count: str, seed: int
) -> np.ndarray:
    """Flat boolean mask over all particles (row-major jet order).

    Each jet with n >= 2 particles gets k masked particles chosen uniformly
    without replacement, k = round(ratio * n) or int(ratio * n) depending on
    ``mask_count``, clamped to [1, n - 1]; jets with fewer particles are not
    masked.
    """
    to_int = {"round": np.round, "floor": np.floor}[mask_count]
    k = np.clip(to_int(n_particles * mask_ratio), 1, n_particles - 1)
    k = np.where(n_particles >= 2, k, 0).astype(np.int64)

    jet_id = np.repeat(np.arange(len(n_particles)), n_particles)
    scores = np.random.default_rng(seed).random(len(jet_id))
    # Rank the random scores within each jet and mask the k lowest.
    order = np.lexsort((scores, jet_id))
    starts = np.repeat(np.cumsum(n_particles) - n_particles, n_particles)
    rank = np.empty(len(jet_id), dtype=np.int64)
    rank[order] = np.arange(len(jet_id)) - starts
    return rank < np.repeat(k, n_particles)


def collision_rates(
    patch: np.ndarray,
    jet_id: np.ndarray,
    n_jets: int,
    n_patches: int,
    masked: np.ndarray,
) -> dict[str, float]:
    """Per-particle, per-jet and mask-mask collision rates for one scheme."""

    def shares_patch(keys: np.ndarray) -> np.ndarray:
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        return counts[inverse] > 1

    key = jet_id * n_patches + patch  # unique per (jet, patch)
    colliding = shares_patch(key)
    return {
        "per_particle": float(colliding.mean()),
        "per_jet": float(np.unique(jet_id[colliding]).size / n_jets),
        "mask_mask": float(shares_patch(key[masked]).mean()),
    }


def print_table(rows: list[dict], grids: list[int]) -> None:
    """Print the rates (%) with one block per metric, as in the paper table."""
    width = 14
    print(f"\n{'Scheme':<{width}}" + "".join(f"{f'{g}x{g}':>10}" for g in grids))
    print(f"{'(patches)':<{width}}" + "".join(f"{g * g:>10,}" for g in grids))
    for metric, title in METRICS.items():
        print(f"\n{title}")
        for scheme in SCHEMES:
            values = {r["grid"]: r[metric] for r in rows if r["scheme"] == scheme}
            print(
                f"{SCHEME_LABELS[scheme]:<{width}}"
                + "".join(f"{100 * values[g]:>10.2f}" for g in grids)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root"],
        help="QCD jet files (ROOT, or padded HDF5), directories or glob patterns.",
    )
    parser.add_argument(
        "--max-jets", type=int, default=0, help="Jets to read (<= 0: all)."
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=90,
        help="Constituents kept per jet (pretraining max_seq_length).",
    )
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[20, 50, 80, 100, 120],
        help="Bins per axis; each scheme uses a grid x grid patch table.",
    )
    parser.add_argument(
        "--r-scale",
        type=float,
        default=None,
        help="Polar r_scale (standardized units). Default: median radius.",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Polar r_max (standardized units). Default: 99th-percentile radius.",
    )
    parser.add_argument(
        "--axis-bound",
        type=float,
        default=None,
        help="Cartesian and Log-Cartesian grids span [-bound, bound] per axis. "
        "Default: the larger 99th percentile of |dEta| and |dPhi|.",
    )
    parser.add_argument(
        "--log-scale",
        type=float,
        default=None,
        help="Log-Cartesian scale in sign(x) log(1 + |x| / scale). "
        "Default: r_scale / 2.",
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=0.25, help="Masked fraction per jet."
    )
    parser.add_argument(
        "--mask-count",
        choices=["round", "floor"],
        default="round",
        help="Masked count per jet: round(ratio * n) (paper table) or "
        "int(ratio * n) (iBOT training rule).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Mask sampling seed.")
    parser.add_argument(
        "--h5-label",
        type=int,
        default=None,
        help="HDF5 inputs only: keep jets whose 'label' equals this (QCD = 0).",
    )
    parser.add_argument(
        "--output-dir",
        default="PROJECT_ROOT/experiments/studies/physics/pe_collision",
        help="Directory for pe_collision.json and pe_collision.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    deta, dphi = load_standardized_coords(
        files,
        max_jets=args.max_jets if args.max_jets > 0 else None,
        max_particles=args.max_particles,
        h5_label=args.h5_label,
    )
    n_particles = ak.to_numpy(ak.num(deta, axis=1)).astype(np.int64)
    n_jets = len(n_particles)
    deta_flat = ak.to_numpy(ak.flatten(deta)).astype(np.float64)
    dphi_flat = ak.to_numpy(ak.flatten(dphi)).astype(np.float64)
    print(f"{n_jets:,} jets, {len(deta_flat):,} constituents")

    percentiles = coordinate_percentiles(deta_flat, dphi_flat)

    def pick(value: float | None, default: float) -> float:
        # Data-derived defaults, used unless given on the command line.
        return default if value is None else value

    r_scale = pick(args.r_scale, percentiles["radius"]["50"])
    params = {
        "r_scale": r_scale,
        "r_max": pick(args.r_max, percentiles["radius"]["99"]),
        "axis_bound": pick(
            args.axis_bound,
            max(percentiles["abs_deta"]["99"], percentiles["abs_dphi"]["99"]),
        ),
        "log_scale": pick(args.log_scale, r_scale / 2),
    }
    raw_std = _NORM["deta"]["std"]
    print("\nStandardized radius percentiles:")
    for p, value in percentiles["radius"].items():
        print(f"  p{p:>5}  r = {value:.4f}")
    print(
        f"r_scale = {params['r_scale']:.4f} (raw {params['r_scale'] * raw_std:.4f}), "
        f"r_max = {params['r_max']:.4f} (raw {params['r_max'] * raw_std:.4f})"
    )
    print("Binning parameters: " + ", ".join(f"{k}={v:.4g}" for k, v in params.items()))

    masked = sample_masks(n_particles, args.mask_ratio, args.mask_count, args.seed)
    print(f"Masked fraction of constituents: {masked.mean():.2%} ({args.mask_count})\n")
    jet_id = np.repeat(np.arange(n_jets, dtype=np.int64), n_particles)
    coords = torch.from_numpy(np.stack([deta_flat, dphi_flat], axis=1))

    rows = []
    for scheme in SCHEMES:
        for grid in args.grids:
            patch = patch_indices(build_encoder(scheme, grid, params), coords)
            rates = collision_rates(patch, jet_id, n_jets, grid * grid, masked)
            rows.append(
                {"scheme": scheme, "grid": grid, "n_patches": grid * grid, **rates}
            )
            print(
                f"  {SCHEME_LABELS[scheme]:<14} {grid:>3}x{grid:<3} "
                + "  ".join(f"{k}={100 * v:.2f}%" for k, v in rates.items())
            )
    print_table(rows, args.grids)

    out_dir = resolve_path(args.output_dir)
    write_json(
        {
            "n_jets": n_jets,
            "n_particles": int(len(deta_flat)),
            "max_particles": args.max_particles,
            "mask_ratio": args.mask_ratio,
            "mask_count": args.mask_count,
            "masked_fraction": float(masked.mean()),
            "seed": args.seed,
            "normalization_std": {"deta": raw_std, "dphi": _NORM["dphi"]["std"]},
            "percentiles": percentiles,
            "binning_params": params,
            "rates": rows,
            "inputs": [str(f) for f in files],
        },
        out_dir / "pe_collision.json",
    )
    with open(out_dir / "pe_collision.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir / 'pe_collision.csv'}")


if __name__ == "__main__":
    main()

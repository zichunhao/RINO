"""Physical scales of the exclusive-kT views (Appendix table ``tab:kt-scales``).

For each QCD jet with more than ``--min-constituents`` constituents, the
constituents are reclustered with the exclusive kT algorithm (radius ``--R``,
the same FastJet setup as ``dino/preprocess/jetclass/cluster.py``). For every
view N in the ladder the script records

- the merge scale ``q_N = sqrt(d_merge(N))``, where ``d_merge(N)`` is the kT
  distance (``d_ij`` or ``d_iB``) of the clustering step from N+1 to N subjets
  (FastJet ``exclusive_dmerge(N)``);
- the softest-subjet momentum fraction ``pT_soft / pT_jet``: the smallest pT
  among the N exclusive subjets over the pT of the constituent sum.

It reports the median [IQR] of both per N, plus how often the q_N ladder is
strictly decreasing, only weakly decreasing (ties), or inverted.

Example (JetClass QCD = ``ZJetsToNuNu`` files; the default reads the first
50k jets of the training split, as in the paper)::

    python studies/physics/kt_scales.py \\
        --inputs "PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root" \\
        --max-jets 50000 \\
        --output PROJECT_ROOT/experiments/studies/physics/kt_scales.json
"""

from __future__ import annotations

import argparse
import warnings

import awkward as ak
import fastjet
import numpy as np

from _common import expand_inputs, load_particle_arrays, median_iqr, write_json

DEFAULT_LADDER = [2, 3, 4, 6, 8, 16]
P4_BRANCHES = ["part_px", "part_py", "part_pz", "part_energy"]


def cluster_chunk(
    particles: ak.Array, ladder: list[int], radius: float
) -> dict[str, np.ndarray]:
    """Exclusive-kT scales for one chunk of jets.

    Args:
        particles: Jagged record array with ``part_px/py/pz/energy`` fields.
            Every jet must have more than ``max(ladder)`` constituents.
        ladder: Numbers of exclusive subjets N.
        radius: kT jet radius R.

    Returns:
        ``q`` and ``soft_frac``, each an ``(n_jets, len(ladder))`` array, and
        ``jet_pt`` of shape ``(n_jets,)``.
    """
    p4 = ak.zip(
        {
            "px": ak.values_astype(particles.part_px, np.float64),
            "py": ak.values_astype(particles.part_py, np.float64),
            "pz": ak.values_astype(particles.part_pz, np.float64),
            "E": ak.values_astype(particles.part_energy, np.float64),
        },
        with_name="Momentum4D",
    )
    jet_pt = np.hypot(
        ak.to_numpy(ak.sum(p4.px, axis=1)), ak.to_numpy(ak.sum(p4.py, axis=1))
    )
    sequence = fastjet.ClusterSequence(
        p4, fastjet.JetDefinition(fastjet.kt_algorithm, radius)
    )

    q = np.empty((len(jet_pt), len(ladder)))
    soft_pt = np.empty_like(q)
    for col, n_subjets in enumerate(ladder):
        d_merge = ak.to_numpy(sequence.exclusive_dmerge(njets=n_subjets))
        with np.errstate(invalid="ignore"):
            q[:, col] = np.where(d_merge > 0, np.sqrt(d_merge), np.nan)
        subjets = sequence.exclusive_jets(n_jets=n_subjets)
        soft_pt[:, col] = ak.to_numpy(ak.min(np.hypot(subjets.px, subjets.py), axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        soft_frac = soft_pt / jet_pt[:, None]
    return {"q": q, "soft_frac": soft_frac, "jet_pt": jet_pt}


def ladder_ordering(q: np.ndarray) -> dict[str, float]:
    """Fractions of jets whose q_N ladder is strict, tied, or inverted.

    ``strict``: q_N > q_N' for every adjacent pair N < N'; ``ties``: not
    strict but never increasing; ``inverted``: at least one increase.
    """
    steps = np.diff(q, axis=1)  # want every step < 0
    strict = np.all(steps < 0, axis=1)
    non_increasing = np.all(steps <= 0, axis=1)
    return {
        "strict": float(strict.mean()),
        "ties": float((non_increasing & ~strict).mean()),
        "inverted": float((~non_increasing).mean()),
    }


def compute_kt_scales(
    particles: ak.Array,
    ladder: list[int],
    radius: float,
    min_constituents: int,
    chunk_size: int,
) -> dict:
    """Cluster all eligible jets and summarize q_N and pT_soft/pT_jet per N."""
    if min_constituents < max(ladder):
        raise ValueError(
            f"--min-constituents ({min_constituents}) must be >= max N "
            f"({max(ladder)}) so every view is defined"
        )
    n_const = ak.to_numpy(ak.num(particles.part_energy, axis=1))
    eligible = particles[n_const > min_constituents]
    print(
        f"{len(eligible):,} of {len(particles):,} jets have more than "
        f"{min_constituents} constituents"
    )

    parts = []
    for start in range(0, len(eligible), chunk_size):
        parts.append(
            cluster_chunk(eligible[start : start + chunk_size], ladder, radius)
        )
        print(f"  clustered {min(start + chunk_size, len(eligible)):,} jets")
    q = np.concatenate([p["q"] for p in parts])
    soft_frac = np.concatenate([p["soft_frac"] for p in parts])
    jet_pt = np.concatenate([p["jet_pt"] for p in parts])

    # Keep jets whose whole ladder is defined (finite, positive d_merge).
    valid = np.all(np.isfinite(q), axis=1) & (jet_pt > 0)
    q, soft_frac = q[valid], soft_frac[valid]

    return {
        "n_jets_read": int(len(particles)),
        "n_jets_used": int(valid.sum()),
        "ladder": ladder,
        "radius": radius,
        "min_constituents_exclusive": min_constituents,
        "per_view": {
            str(n): {
                "q_N_GeV": median_iqr(q[:, col]),
                "pt_soft_over_pt_jet": median_iqr(soft_frac[:, col]),
            }
            for col, n in enumerate(ladder)
        },
        "q_N_ladder_ordering": ladder_ordering(q),
    }


def _fmt(stats: dict[str, float], spec: str) -> str:
    return f"{stats['median']:{spec}} [{stats['q25']:{spec}}, {stats['q75']:{spec}}]"


def print_summary(result: dict) -> None:
    """Print the per-N table and the ladder-ordering fractions."""
    print(f"\nJets used: {result['n_jets_used']:,}")
    print(f"{'N':>3} | {'q_N [GeV], median [IQR]':>28} | {'pT_soft/pT_jet':>28}")
    for n in result["ladder"]:
        row = result["per_view"][str(n)]
        print(
            f"{n:>3} | {_fmt(row['q_N_GeV'], '.2f'):>28} | "
            f"{_fmt(row['pt_soft_over_pt_jet'], '.2g'):>28}"
        )
    order = result["q_N_ladder_ordering"]
    print(
        "\nq_N ladder: strictly decreasing {:.2%}, ties only {:.2%}, "
        "inverted {:.3%}".format(order["strict"], order["ties"], order["inverted"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root"],
        help="QCD jet files (ROOT, or padded HDF5), directories or glob patterns.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=50_000,
        help="Number of jets to read before the constituent cut (<= 0: all).",
    )
    parser.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=DEFAULT_LADDER,
        help="Numbers of exclusive subjets N (the kT views).",
    )
    parser.add_argument("--R", type=float, default=0.8, help="kT jet radius.")
    parser.add_argument(
        "--min-constituents",
        type=int,
        default=16,
        help="Keep jets with strictly more constituents than this.",
    )
    parser.add_argument(
        "--h5-label",
        type=int,
        default=None,
        help="HDF5 inputs only: keep jets whose 'label' equals this (QCD = 0).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20_000, help="Jets per FastJet call."
    )
    parser.add_argument(
        "--output",
        default="PROJECT_ROOT/experiments/studies/physics/kt_scales.json",
        help="Summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ladder = sorted(set(args.ladder))
    files = expand_inputs(args.inputs)
    particles = load_particle_arrays(
        files,
        P4_BRANCHES,
        max_jets=args.max_jets if args.max_jets > 0 else None,
        h5_label=args.h5_label,
    )
    # FastJet warns about exclusive jets for every call; kT is well defined.
    warnings.filterwarnings("ignore", message="dcut and exclusive jets")
    result = compute_kt_scales(
        particles, ladder, args.R, args.min_constituents, args.chunk_size
    )
    result["inputs"] = [str(f) for f in files]
    print_summary(result)
    write_json(result, args.output)


if __name__ == "__main__":
    main()

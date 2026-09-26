"""Count constituents per jet for the multiplicity CDF (``fig:nparticles-cdf``).

Harvester for ``nparticles_cdf.py``: for every split it counts the non-zero
``part_energy`` entries of each jet (the constituent multiplicity) and stores
the counts in an HDF5 file with one int16 dataset per split plus
``combined`` (all splits concatenated in the given order).

Example (JetClass QCD = ``ZJetsToNuNu`` files of the train, validation and
test splits, 12.5M jets in total)::

    python studies/physics/nparticles_count.py \\
        --split train="PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root" \\
        --split val="PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root" \\
        --split test="PROJECT_ROOT/data/JetClass/raw/test_20M/ZJetsToNuNu_*.root" \\
        --output PROJECT_ROOT/experiments/studies/physics/nparticles_qcd.h5
"""

from __future__ import annotations

import argparse

import awkward as ak
import h5py
import numpy as np

from _common import expand_inputs, load_particle_arrays, resolve_path

DEFAULT_SPLITS = [
    "train=PROJECT_ROOT/data/JetClass/raw/train_100M/ZJetsToNuNu_*.root",
    "val=PROJECT_ROOT/data/JetClass/raw/val_5M/ZJetsToNuNu_*.root",
    "test=PROJECT_ROOT/data/JetClass/raw/test_20M/ZJetsToNuNu_*.root",
]


def count_constituents(files, h5_label: int | None = None) -> np.ndarray:
    """Number of non-zero ``part_energy`` entries per jet, file by file."""
    counts = []
    for path in files:
        energy = load_particle_arrays([path], ["part_energy"], h5_label=h5_label)
        counts.append(ak.to_numpy(ak.count_nonzero(energy.part_energy, axis=1)))
    return np.concatenate(counts).astype(np.int16)


def parse_split(spec: str) -> tuple[str, str]:
    """Parse ``NAME=PATTERN`` into ``(NAME, PATTERN)``."""
    name, sep, pattern = spec.partition("=")
    if not sep or not name or not pattern:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATTERN, got {spec!r}")
    return name, pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        type=parse_split,
        default=None,
        metavar="NAME=PATTERN",
        help="Split name and its files (ROOT or padded HDF5, glob allowed). "
        "Repeat per split. Default: JetClass QCD train, val and test.",
    )
    parser.add_argument(
        "--h5-label",
        type=int,
        default=None,
        help="HDF5 inputs only: keep jets whose 'label' equals this (QCD = 0).",
    )
    parser.add_argument(
        "--output",
        default="PROJECT_ROOT/experiments/studies/physics/nparticles_qcd.h5",
        help="Output HDF5 file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = args.splits or [parse_split(s) for s in DEFAULT_SPLITS]

    counts = {}
    sources = {}
    for name, pattern in splits:
        files = expand_inputs([pattern])
        print(f"[{name}] {len(files)} files")
        counts[name] = count_constituents(files, args.h5_label)
        sources[name] = [f.name for f in files]
        print(f"[{name}] {len(counts[name]):,} jets")

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for name, values in counts.items():
            dataset = handle.create_dataset(name, data=values, compression="gzip")
            dataset.attrs["files"] = sources[name]
        handle.create_dataset(
            "combined", data=np.concatenate(list(counts.values())), compression="gzip"
        )
        handle.attrs["splits"] = list(counts)
        handle.attrs["count_definition"] = "number of non-zero part_energy per jet"
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

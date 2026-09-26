"""Shared helpers for the physics/data studies in ``studies/physics``.

Every script in this directory imports this module, which:

- puts ``dino/`` on ``sys.path`` so release modules (``utils``, ``models``,
  ``losses``, ``dataloader``) import exactly as they do in training;
- resolves the ``PROJECT_ROOT`` placeholder in CLI paths with the same
  ``process_placeholder`` used by the training configs;
- loads jagged per-particle arrays from JetClass-style ROOT files, or from
  padded HDF5 files, with an optional cap on the number of jets.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DINO_DIR = REPO_ROOT / "dino"
if str(DINO_DIR) not in sys.path:
    sys.path.insert(0, str(DINO_DIR))

import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402

from utils.ckpt import process_placeholder  # noqa: E402
from utils.logger import LOGGER  # noqa: E402

# Keep script output readable: release modules log their construction at INFO.
LOGGER.setLevel("WARNING")

HDF5_SUFFIXES = (".h5", ".hdf5")


def load_release_module(relative_path: str, name: str) -> ModuleType:
    """Import a single file under ``dino/`` without its package ``__init__``.

    The studies need only a few self-contained components (e.g. the
    positional-encoding binning); loading the file alone keeps them
    independent of the full model package and its backbone dependencies.
    """
    spec = importlib.util.spec_from_file_location(name, DINO_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(path: str | Path) -> Path:
    """Resolve the ``PROJECT_ROOT`` placeholder in *path*."""
    return Path(process_placeholder(str(path), {}, None))


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    """Expand files, directories and glob patterns into a sorted file list.

    ``PROJECT_ROOT`` is resolved first. A directory expands to the ROOT and
    HDF5 files directly inside it. Each pattern's matches are sorted, and the
    patterns keep the order they were given in.
    """
    files: list[Path] = []
    for pattern in patterns:
        resolved = str(resolve_path(pattern))
        if Path(resolved).is_dir():
            matches = [
                p
                for p in Path(resolved).iterdir()
                if p.suffix.lower() in (".root",) + HDF5_SUFFIXES
            ]
        else:
            matches = [Path(p) for p in glob.glob(resolved, recursive=True)]
        if not matches:
            raise FileNotFoundError(f"No input files match {pattern!r}")
        files.extend(sorted(matches))
    return files


def _load_root(
    path: Path, branches: Sequence[str], n_max: int | None, tree: str
) -> ak.Array:
    import uproot

    with uproot.open(path) as f:
        return f[tree].arrays(list(branches), entry_stop=n_max)


def _load_hdf5(
    path: Path,
    branches: Sequence[str],
    n_max: int | None,
    label: int | None,
    real_key: str,
) -> ak.Array:
    """Read padded ``(n_jets, max_particles)`` datasets and strip the padding.

    Real particles are those with a non-zero ``real_key`` entry (the same
    convention as ``dino/preprocess/jetclass/cluster.py``). With ``label``
    set, only jets whose ``label`` dataset equals it are kept.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if label is not None:
            rows = np.flatnonzero(f["label"][:] == label)
            if n_max is not None:
                rows = rows[:n_max]
        else:
            n_rows = f[real_key].shape[0]
            rows = np.arange(n_rows if n_max is None else min(n_max, n_rows))
        if rows.size == 0:
            raise ValueError(f"No jets selected from {path}")

        def take(key: str) -> np.ndarray:
            # Read the covering slice once, then select rows in memory
            # (much faster than h5py point selection for many rows).
            block = f[key][rows[0] : rows[-1] + 1]
            return block[rows - rows[0]]

        real = take(real_key) != 0
        counts = real.sum(axis=1)
        # Row-major boolean selection keeps each jet's real particles in
        # order; unflatten restores the per-jet (jagged) structure.
        return ak.zip(
            {key: ak.unflatten(take(key)[real], counts) for key in branches},
            depth_limit=1,
        )


def load_particle_arrays(
    files: Sequence[Path],
    branches: Sequence[str],
    max_jets: int | None = None,
    tree: str = "tree",
    h5_label: int | None = None,
    h5_real_key: str = "part_energy",
) -> ak.Array:
    """Load per-particle branches as one jagged array, file by file.

    Args:
        files: ROOT or HDF5 files, read in the given order.
        branches: Branch (ROOT) or dataset (HDF5) names to load.
        max_jets: Stop after this many jets in total (``None`` reads all).
        tree: ROOT tree name.
        h5_label: HDF5 only: keep jets whose ``label`` dataset equals this.
        h5_real_key: HDF5 only: dataset whose non-zero entries mark real
            particles in the padded arrays.

    Returns:
        Record array with one jagged field per branch.
    """
    chunks = []
    n_loaded = 0
    for path in files:
        remaining = None if max_jets is None else max_jets - n_loaded
        if remaining is not None and remaining <= 0:
            break
        print(f"  reading {path}")
        if path.suffix.lower() in HDF5_SUFFIXES:
            chunk = _load_hdf5(path, branches, remaining, h5_label, h5_real_key)
        else:
            chunk = _load_root(path, branches, remaining, tree)
        chunks.append(chunk)
        n_loaded += len(chunk)
    if not chunks:
        raise ValueError("No jets were loaded")
    return ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def median_iqr(values: np.ndarray) -> dict[str, float]:
    """Median and interquartile range (25th, 75th percentiles)."""
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75)}


def write_json(obj: dict, path: str | Path) -> Path:
    """Write *obj* as indented JSON, creating parent directories."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)
    print(f"Wrote {path}")
    return path

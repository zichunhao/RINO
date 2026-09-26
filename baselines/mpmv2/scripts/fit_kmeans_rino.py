"""Fit the K-means codebook of the masked K-means task on RINO kinematics.

The upstream codebook ``resources/kmeans_7.pkl`` was fit on MPMv2's
quantile-normalised (pT, deta, dphi, d0, d0err, dz, dzerr) features. The RINO
HDF5 files written by ``scripts/make_jetclass.py`` hold the 7 affine-normalised
RINO kinematics instead, and the RINO datamodules apply no further
preprocessing, so this script fits the codebook directly on the valid
constituents stored in those files.

Jets are read as random contiguous blocks spread over the input files. The
result is a pickled ``src.models.kmeans.KMeansCodebook`` that ``KmeansTask``
loads through ``model.tasks.kmeans.kmeans_path``.

Usage:
    python scripts/fit_kmeans_rino.py \
        --input PROJECT_ROOT/data/JetClass/mpm-rino/train_100M_combined_QCD.h5 \
        --eval_input PROJECT_ROOT/data/JetClass/mpm-rino/val_5M_combined_QCD.h5
"""

import argparse
import glob
import logging
import math
from pathlib import Path

import h5py
import numpy as np
import rootutils
import torch as T

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from src.models.kmeans import KMeansCodebook  # noqa: E402

log = logging.getLogger(__name__)


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand file paths and glob patterns into a sorted list of HDF5 files."""
    paths = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No files match {patterns}")
    return paths


def sample_constituents(
    paths: list[Path],
    n_jets: int,
    block_size: int,
    n_csts: int,
    csts_dim: int,
    rng: np.random.Generator,
) -> T.Tensor:
    """Read the valid constituents of about ``n_jets`` jets.

    The jets of all files are split into contiguous blocks of ``block_size``
    and ``ceil(n_jets / block_size)`` blocks are drawn without replacement.
    Only the first ``n_csts`` constituents and ``csts_dim`` features are kept,
    matching the slicing of the datamodule.

    Returns
    -------
    T.Tensor
        Float32 tensor of shape ``(n_constituents, csts_dim)``.
    """
    blocks = []
    for path in paths:
        with h5py.File(path, "r") as f:
            n_file = len(f["mask"])
        blocks += [
            (path, start, min(start + block_size, n_file))
            for start in range(0, n_file, block_size)
        ]
    n_blocks = min(math.ceil(n_jets / block_size), len(blocks))
    chosen = sorted(rng.choice(len(blocks), size=n_blocks, replace=False))

    csts = []
    for i in chosen:
        path, start, stop = blocks[i]
        with h5py.File(path, "r") as f:
            block = f["csts"][start:stop, :n_csts, :csts_dim]
            mask = f["mask"][start:stop, :n_csts].astype(bool)
        csts.append(block[mask])
    n_read = sum(blocks[i][2] - blocks[i][1] for i in chosen)
    csts = np.concatenate(csts).astype(np.float32)
    log.info(f"Read {len(csts):,} constituents from {n_read:,} jets")
    return T.from_numpy(csts)


def codebook_usage(labels: T.Tensor, n_clusters: int) -> tuple[int, float]:
    """Number of used clusters and perplexity of the cluster assignment."""
    counts = T.bincount(labels, minlength=n_clusters).double()
    probs = counts[counts > 0] / counts.sum()
    perplexity = T.exp(-(probs * probs.log()).sum()).item()
    return int((counts > 0).sum()), perplexity


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="HDF5 files or glob patterns with 'csts' and 'mask' datasets",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: resources/kmeans_rino_<csts_dim>.pkl)",
    )
    parser.add_argument("--n_clusters", type=int, default=16384)
    parser.add_argument(
        "--n_jets", type=int, default=300_000, help="Number of jets to fit on"
    )
    parser.add_argument(
        "--block_size", type=int, default=10_000, help="Jets per contiguous read"
    )
    parser.add_argument("--n_csts", type=int, default=128)
    parser.add_argument("--csts_dim", type=int, default=7)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--init", choices=["kmeans++", "random"], default="kmeans++")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Points per nearest-centroid chunk (bounds peak memory)",
    )
    parser.add_argument("--device", default="cuda" if T.cuda.is_available() else "cpu")
    parser.add_argument(
        "--eval_input",
        nargs="+",
        default=None,
        help="Held-out HDF5 files or glob patterns to report codebook usage on",
    )
    parser.add_argument("--eval_jets", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = get_args()
    output = Path(
        args.output or root / "resources" / f"kmeans_rino_{args.csts_dim}.pkl"
    )

    rng = np.random.default_rng(args.seed)
    points = sample_constituents(
        expand_paths(args.input),
        args.n_jets,
        args.block_size,
        args.n_csts,
        args.csts_dim,
        rng,
    ).to(args.device)

    generator = T.Generator(device=args.device).manual_seed(args.seed)
    kmeans = KMeansCodebook(args.n_clusters, args.csts_dim, args.chunk_size)
    kmeans = kmeans.to(args.device)
    labels = kmeans.fit(
        points,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        generator=generator,
    )
    used, perplexity = codebook_usage(labels, args.n_clusters)
    log.info(
        f"Fit sample: {used}/{args.n_clusters} clusters used, "
        f"perplexity {perplexity:.1f}"
    )

    if args.eval_input:
        eval_points = sample_constituents(
            expand_paths(args.eval_input),
            args.eval_jets,
            args.block_size,
            args.n_csts,
            args.csts_dim,
            rng,
        ).to(args.device)
        used, perplexity = codebook_usage(
            kmeans.predict(eval_points.T), args.n_clusters
        )
        log.info(
            f"Held-out sample: {used}/{args.n_clusters} clusters used, "
            f"perplexity {perplexity:.1f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    T.save(kmeans.cpu(), output)
    log.info(f"Saved codebook to {output}")


if __name__ == "__main__":
    main()

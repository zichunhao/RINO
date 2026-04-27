"""Combine multiple pre-shuffled JetClass HDF5 files into one contiguous file.

The input files under `shuffled{seed}_kt/{split}_{label}/mixed_*.h5` are already
globally shuffled, so we just concatenate them in filename order. The resulting
single-file layout is much friendlier to CephFS (one metadata open per split
instead of N), and enables lazy per-slice reads from dataloaders.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger("combine")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_dir", type=str, required=True,
                   help="Input directory containing the per-file shards.")
    p.add_argument("--out", dest="out_file", type=str, required=True,
                   help="Output HDF5 file path (will be overwritten).")
    p.add_argument("--pattern", type=str, default="*.h5",
                   help="Glob pattern for input files (default: *.h5).")
    p.add_argument("--chunk-rows", type=int, default=1000,
                   help="Chunk size (rows) for output datasets. "
                        "Match to training batch size for best read perf.")
    p.add_argument("--compression", type=str, default=None,
                   help="HDF5 compression filter (e.g. 'lzf', 'gzip'). "
                        "None = uncompressed (fastest reads).")
    p.add_argument("--log", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = get_args()
    logging.basicConfig(
        level=args.log.upper(),
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    in_dir = Path(args.in_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {in_dir}")
    LOGGER.info(f"Found {len(files)} input files in {in_dir}")

    # Pass 1: read row counts so we can preallocate exactly.
    sizes: list[int] = []
    for f in tqdm(files, desc="metadata"):
        with h5py.File(f, "r") as h:
            first_key = next(iter(h.keys()))
            sizes.append(h[first_key].shape[0])
    total_rows = int(np.sum(sizes))
    LOGGER.info(f"Total rows: {total_rows:,}")

    # Peek at the first file to get schema.
    with h5py.File(files[0], "r") as h:
        schema = {k: (h[k].shape[1:], h[k].dtype) for k in h.keys()}
    LOGGER.info(f"Schema: {len(schema)} datasets")

    # Create output datasets with final shape preallocated.
    LOGGER.info(f"Creating {out_file}")
    with h5py.File(out_file, "w") as out:
        for key, (tail_shape, dtype) in schema.items():
            shape = (total_rows, *tail_shape)
            chunks = (min(args.chunk_rows, total_rows), *tail_shape)
            out.create_dataset(
                key,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compression=args.compression,
            )

        # Pass 2: copy each file contiguously into the output.
        row = 0
        for f, nrows in zip(tqdm(files, desc="copy"), sizes):
            with h5py.File(f, "r") as h:
                missing = schema.keys() - h.keys()
                if missing:
                    LOGGER.warning(f"{f.name} missing keys: {sorted(missing)}")
                extra = h.keys() - schema.keys()
                if extra:
                    LOGGER.warning(f"{f.name} has extra keys (skipped): {sorted(extra)}")
                for key in schema:
                    if key not in h:
                        continue
                    out[key][row : row + nrows] = h[key][:]
            row += nrows

        assert row == total_rows, f"row mismatch: wrote {row}, expected {total_rows}"

    LOGGER.info(f"Wrote {total_rows:,} rows to {out_file} ({out_file.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()

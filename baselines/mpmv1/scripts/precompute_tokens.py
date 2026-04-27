"""Precompute VQ-VAE token labels for RINO-preprocessed combined HDF5 files.

Runs the frozen SharedVQVAE once over each combined H5 and writes a sibling
file `{basename}_tokens.h5` containing a single `code_labels` (N, n_nodes)
int32 dataset. RINOH5Iterator detects the sibling file and yields the
precomputed tokens, so Bert.preprocess_inputs can skip the runtime
tokenize() call entirely.

Why this exists: MPMv1 training was deadlocking under DDP because the
runtime vqvae.tokenize() call has data-dependent tensor shapes
(`latents[mask]`), which trigger KeOps / vqtorch kernel JIT with
variable compile times per rank. The resulting per-rank runtime drift
deadlocks NCCL allreduce at the first backward. Precomputing tokens
offline removes the VQ call from training_step entirely.

Usage:
    python baselines/mpmv1/scripts/precompute_tokens.py \\
        --data-dir PROJECT_ROOT/data/JetClass/mpm-rino \\
        --ckpt PROJECT_ROOT/experiments/vqvae/vqvae-shared/checkpoints/last.ckpt
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Add PARCEL root + baselines to sys.path so we can import SharedVQVAELightning.
_script_dir = Path(__file__).resolve().parent          # .../baselines/mpmv1/scripts/
_mpmv1_dir = _script_dir.parent                         # .../baselines/mpmv1/
_baselines_dir = _mpmv1_dir.parent                      # .../baselines/
_project_dir = _baselines_dir.parent                    # PARCEL root
sys.path.insert(0, str(_project_dir))


def precompute_file(
    h5_path: Path,
    out_path: Path,
    vqvae,
    device: torch.device,
    chunk_size: int,
    n_nodes: int,
) -> None:
    """Tokenize every jet in h5_path, writing (N, n_nodes) int32 to out_path."""
    with h5py.File(h5_path, "r") as fin:
        n = len(fin["csts"])
        print(f"[{h5_path.name}] {n} jets — tokenizing in chunks of {chunk_size}")
        t_start = time.time()

        with h5py.File(out_path, "w") as fout:
            code_labels = fout.create_dataset(
                "code_labels",
                shape=(n, n_nodes),
                dtype=np.int32,
                chunks=(min(chunk_size, n), n_nodes),
                compression="lzf",
            )
            fout.attrs["source"] = str(h5_path.name)
            fout.attrs["n_nodes"] = n_nodes

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)

                csts = fin["csts"][start:end, :n_nodes, :7].astype(np.float32)
                mask = fin["mask"][start:end, :n_nodes].astype(bool)

                csts_t = torch.from_numpy(csts).to(device, non_blocking=True)
                mask_t = torch.from_numpy(mask).to(device, non_blocking=True)

                with torch.no_grad():
                    codes = vqvae.tokenize(csts_t, mask_t)  # (B, N), -1 for padding

                code_labels[start:end] = codes.cpu().numpy().astype(np.int32)

                # Progress print every ~1% or every 100 chunks, whichever is coarser.
                chunks_done = (start // chunk_size) + 1
                if chunks_done == 1 or chunks_done % max(1, n // (chunk_size * 100)) == 0:
                    elapsed = time.time() - t_start
                    rate = end / max(elapsed, 1e-6)
                    eta = (n - end) / max(rate, 1e-6)
                    print(
                        f"[{h5_path.name}] {end}/{n} ({100 * end / n:.1f}%) "
                        f"— {rate:.0f} jets/s, ETA {eta / 60:.1f} min",
                        flush=True,
                    )

        print(
            f"[{h5_path.name}] done in {(time.time() - t_start) / 60:.1f} min → {out_path.name}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="PROJECT_ROOT/data/JetClass/mpm-rino",
        help="Directory containing the combined HDF5 files.",
    )
    parser.add_argument(
        "--ckpt",
        default="PROJECT_ROOT/experiments/vqvae/vqvae-shared/checkpoints/last.ckpt",
        help="Path to the shared VQ-VAE checkpoint.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Number of jets to tokenize per GPU batch.",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=128,
        help="Number of particles per jet (must match training config).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "train_100M_combined_QCD.h5",
            "val_5M_combined_QCD.h5",
            "test_20M_combined_QCD.h5",
        ],
        help="H5 filenames to process (relative to --data-dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing *_tokens.h5 sibling files.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading VQ-VAE from {args.ckpt}")
    from baselines.vqvae import SharedVQVAELightning  # noqa: E402

    module = SharedVQVAELightning.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    )
    vqvae = module.model.to(device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    data_dir = Path(args.data_dir)
    for fname in args.files:
        h5_path = data_dir / fname
        if not h5_path.exists():
            print(f"[skip] {h5_path} does not exist")
            continue

        out_path = h5_path.with_name(h5_path.stem + "_tokens.h5")
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path.name} already exists (use --force to overwrite)")
            continue

        precompute_file(
            h5_path=h5_path,
            out_path=out_path,
            vqvae=vqvae,
            device=device,
            chunk_size=args.chunk_size,
            n_nodes=args.n_nodes,
        )

    print("All done.")


if __name__ == "__main__":
    main()

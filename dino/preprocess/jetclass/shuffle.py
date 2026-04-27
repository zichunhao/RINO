import argparse
from pathlib import Path
import random
import uproot
import numpy as np
import awkward as ak
import h5py
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict


@dataclass
class FileInfo:
    path: Path
    entries: int
    current_pos: int = 0

    @property
    def remaining_entries(self) -> int:
        return self.entries - self.current_pos

    @property
    def is_complete(self) -> bool:
        return self.current_pos >= self.entries


@dataclass
class ChunkData:
    file_id: int
    data: Dict
    chunk_size: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mix ROOT/HDF5 files with parallel processing"
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="List of input files (ROOT or HDF5)"
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for mixed files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of entries to process at once from each file (default: 1000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of parallel reading threads (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mixing (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of entries within output files",
    )
    parser.add_argument(
        "--compression",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        default=4,
        help="Compression level (0-9, default: 4)",
    )
    parser.add_argument(
        "--dR-filter",
        action="store_true",
        default=False,
        help="If set, filter out events where any particle has dR = sqrt(deta^2 + dphi^2) > 0.8",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=-1,
        help="If > 1, truncate each event to at most this many particles (default: -1, no truncation)",
    )
    parser.add_argument(
        "--output-digits",
        type=int,
        default=2,
        help="Minimum number of digits for output file index padding (default: 2). "
        "Expands automatically if the file count requires more.",
    )
    parser.add_argument(
        "--format",
        choices=["root", "hdf5"],
        default=None,
        help="Output format. Defaults to same as input format.",
    )
    return parser.parse_args()


def shuffle_entries(data):
    """Shuffle entries in the data dictionary while maintaining correlations"""
    n_entries = len(data[list(data.keys())[0]])
    indices = np.random.permutation(n_entries)
    return {branch: data[branch][indices] for branch in data.keys()}


def _detect_format(files):
    """Return 'hdf5' if all files have HDF5 extensions, else 'root'."""
    hdf5_exts = {".h5", ".hdf5"}
    if all(Path(f).suffix.lower() in hdf5_exts for f in files):
        return "hdf5"
    return "root"


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _pad_array(arr, max_len, pad_value=0.0):
    """Pad or truncate a 2D numpy array to shape (N, max_len)."""
    n = arr.shape[0]
    out = np.full((n, max_len), pad_value, dtype=arr.dtype)
    actual = min(arr.shape[1], max_len)
    out[:, :actual] = arr[:, :actual]
    return out


def _ak_to_numpy(branch_data, max_particles):
    """Convert an awkward array to a flat numpy array.

    For 1-D arrays (per-jet scalars) returns shape (N,).
    For 2-D jagged arrays (per-particle) returns shape (N, max_particles),
    padded/truncated to max_particles.
    """
    if branch_data.ndim == 1:
        return ak.to_numpy(branch_data)
    # jagged → pad to max_particles
    padded = ak.pad_none(branch_data, max_particles, clip=True)
    filled = ak.fill_none(padded, 0.0)
    return ak.to_numpy(filled).astype(np.float32)


def write_hdf5(path, data, compression):
    """Write a data dict to an HDF5 file. All values must already be numpy arrays."""
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            opts = (
                dict(compression="gzip", compression_opts=compression)
                if compression > 0
                else {}
            )
            f.create_dataset(key, data=arr, **opts)


# ---------------------------------------------------------------------------
# ROOT mixer (existing logic, refactored to a class method)
# ---------------------------------------------------------------------------


class ParallelMixer:
    def __init__(self, files, chunk_size, n_threads, tree_name="tree"):
        self.chunk_size = chunk_size
        self.tree_name = tree_name
        self.n_threads = n_threads
        self.active_files = []
        self.branch_names = None
        self.total_entries = 0
        self.skipped_files = []
        self.failed_files = []

        print("Reading file information...")
        for f in tqdm(files, desc="Scanning input files"):
            try:
                with uproot.open(f) as root_file:
                    tree = root_file[tree_name]
                    entries = tree.num_entries
                    if entries > 0:
                        if self.branch_names is None:
                            self.branch_names = list(tree.keys())
                        self.active_files.append(FileInfo(f, entries))
                        self.total_entries += entries
                    else:
                        print(f"  WARNING: skipping empty file: {f}")
                        self.skipped_files.append(Path(f))
            except Exception as e:
                print(f"  WARNING: failed to scan {f}: {e}")
                self.skipped_files.append(Path(f))

        self.executor = ThreadPoolExecutor(max_workers=n_threads)

    def read_chunk(self, file_id: int, file_info: FileInfo) -> Optional[ChunkData]:
        try:
            current_chunk_size = min(file_info.remaining_entries, self.chunk_size)
            if current_chunk_size <= 0:
                return None

            with uproot.open(file_info.path) as root_file:
                tree = root_file[self.tree_name]
                data = tree.arrays(
                    self.branch_names,
                    library="ak",
                    entry_start=file_info.current_pos,
                    entry_stop=file_info.current_pos + current_chunk_size,
                )
                file_info.current_pos += current_chunk_size
                return ChunkData(file_id, data, current_chunk_size)

        except Exception as e:
            print(f"Error reading file {file_info.path}: {e}")
            return None

    def get_next_chunks(self):
        if not self.active_files:
            return None

        file_indices = list(range(len(self.active_files)))
        random.shuffle(file_indices)

        future_to_idx = {
            self.executor.submit(self.read_chunk, idx, self.active_files[idx]): idx
            for idx in file_indices
        }

        chunks_data = {branch: [] for branch in self.branch_names}
        files_to_remove = set()
        chunks_read = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk = future.result()
            if chunk is not None:
                for branch in self.branch_names:
                    chunks_data[branch].append(chunk.data[branch])
                chunks_read += 1

                if self.active_files[chunk.file_id].is_complete:
                    files_to_remove.add(chunk.file_id)
            else:
                failed_path = self.active_files[idx].path
                print(f"  WARNING: dropping failed file: {failed_path}")
                self.failed_files.append(failed_path)
                files_to_remove.add(idx)

        for idx in sorted(files_to_remove, reverse=True):
            del self.active_files[idx]

        if chunks_read > 0:
            return {
                branch: ak.concatenate(chunks) for branch, chunks in chunks_data.items()
            }
        return None

    def shutdown(self):
        self.executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# HDF5 mixer
# ---------------------------------------------------------------------------


class HDF5Mixer:
    """Mixer that reads from HDF5 files (already-numpy, fixed- or variable-length)."""

    def __init__(self, files, chunk_size, n_threads):
        self.chunk_size = chunk_size
        self.n_threads = n_threads
        self.active_files = []
        self.branch_names = None
        self.total_entries = 0
        self.skipped_files = []
        self.failed_files = []

        print("Reading file information...")
        for f in tqdm(files, desc="Scanning input files"):
            f = Path(f)
            try:
                with h5py.File(f, "r") as hf:
                    keys = list(hf.keys())
                    if not keys:
                        print(f"  WARNING: skipping empty file: {f}")
                        self.skipped_files.append(f)
                        continue
                    entries = hf[keys[0]].shape[0]
                    if entries > 0:
                        if self.branch_names is None:
                            self.branch_names = keys
                        self.active_files.append(FileInfo(f, entries))
                        self.total_entries += entries
                    else:
                        print(f"  WARNING: skipping empty file: {f}")
                        self.skipped_files.append(f)
            except Exception as e:
                print(f"  WARNING: failed to scan {f}: {e}")
                self.skipped_files.append(f)

        self.executor = ThreadPoolExecutor(max_workers=n_threads)

    def read_chunk(self, file_id: int, file_info: FileInfo) -> Optional[ChunkData]:
        try:
            current_chunk_size = min(file_info.remaining_entries, self.chunk_size)
            if current_chunk_size <= 0:
                return None

            start = file_info.current_pos
            stop = start + current_chunk_size
            with h5py.File(file_info.path, "r") as hf:
                data = {key: hf[key][start:stop] for key in self.branch_names}
            file_info.current_pos += current_chunk_size
            return ChunkData(file_id, data, current_chunk_size)

        except Exception as e:
            print(f"Error reading file {file_info.path}: {e}")
            return None

    def get_next_chunks(self):
        if not self.active_files:
            return None

        file_indices = list(range(len(self.active_files)))
        random.shuffle(file_indices)

        future_to_idx = {
            self.executor.submit(self.read_chunk, idx, self.active_files[idx]): idx
            for idx in file_indices
        }

        chunks_data = {branch: [] for branch in self.branch_names}
        files_to_remove = set()
        chunks_read = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk = future.result()
            if chunk is not None:
                for branch in self.branch_names:
                    chunks_data[branch].append(chunk.data[branch])
                chunks_read += 1

                if self.active_files[chunk.file_id].is_complete:
                    files_to_remove.add(chunk.file_id)
            else:
                failed_path = self.active_files[idx].path
                print(f"  WARNING: dropping failed file: {failed_path}")
                self.failed_files.append(failed_path)
                files_to_remove.add(idx)

        for idx in sorted(files_to_remove, reverse=True):
            del self.active_files[idx]

        if chunks_read > 0:
            return {
                branch: np.concatenate(chunks, axis=0)
                for branch, chunks in chunks_data.items()
            }
        return None

    def shutdown(self):
        self.executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Main mix function
# ---------------------------------------------------------------------------


def mix_files(
    input_files,
    output_dir,
    chunk_size,
    n_threads=4,
    seed=42,
    tree_name="tree",
    shuffle=True,
    compression=4,
    max_particles=-1,
    output_format="root",
    output_digits=2,
    dr_filter=False,
):
    random.seed(seed)
    np.random.seed(seed)

    input_files = [Path(f) for f in input_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_format = _detect_format(input_files)

    if output_format == "hdf5" and input_format == "root" and max_particles < 1:
        raise ValueError(
            "--max-particles must be set when converting ROOT → HDF5, "
            "so jagged particle arrays can be padded to a fixed length."
        )

    if input_format == "hdf5":
        mixer = HDF5Mixer(input_files, chunk_size, n_threads)
    else:
        mixer = ParallelMixer(input_files, chunk_size, n_threads, tree_name)

    if not mixer.active_files:
        print("No valid input files found!")
        return

    try:
        iteration = 0
        total_processed = 0
        total_filtered = 0
        n_valid = len(mixer.active_files)
        pbar = tqdm(total=mixer.total_entries, desc="Processing entries")

        n_digits = output_digits
        ext = ".h5" if output_format == "hdf5" else ".root"

        while True:
            chunks = mixer.get_next_chunks()
            if chunks is None:
                break

            # dR filter: keep events where all particles satisfy 0 <= dR <= 0.8.
            # dR = sqrt(part_deta^2 + part_dphi^2); lower bound >= 0 is always true.
            if dr_filter and "part_deta" in chunks and "part_dphi" in chunks:
                dR = np.sqrt(chunks["part_deta"] ** 2 + chunks["part_dphi"] ** 2)
                if input_format == "root":
                    event_mask = ak.all(dR <= 0.8, axis=1)
                    n_removed = int(ak.sum(~event_mask))
                else:
                    event_mask = np.all(dR <= 0.8, axis=1)
                    n_removed = int(np.sum(~event_mask))
                if n_removed > 0:
                    total_filtered += n_removed
                    chunks = {branch: arr[event_mask] for branch, arr in chunks.items()}
                if len(chunks[mixer.branch_names[0]]) == 0:
                    continue

            # Truncate / pad particle arrays
            if max_particles > 1:
                if input_format == "root" and output_format == "hdf5":
                    # Convert jagged awkward → fixed-length numpy (pads short events,
                    # truncates long ones) so HDF5 gets a rectangular array.
                    chunks = {
                        branch: _ak_to_numpy(arr, max_particles)
                        for branch, arr in chunks.items()
                    }
                elif input_format == "root":
                    # ROOT output: just truncate; jagged arrays are fine
                    chunks = {
                        branch: arr[:, :max_particles] if arr.ndim > 1 else arr
                        for branch, arr in chunks.items()
                    }
                else:
                    # HDF5 input: numpy arrays — pad and truncate to exact shape
                    chunks = {
                        key: _pad_array(arr, max_particles) if arr.ndim > 1 else arr
                        for key, arr in chunks.items()
                    }

            if shuffle:
                chunks = shuffle_entries(chunks)

            output_path = output_dir / f"mixed_{iteration:0{n_digits}d}{ext}"
            if output_format == "hdf5":
                write_hdf5(output_path, chunks, compression)
            else:
                with uproot.recreate(
                    output_path, compression=uproot.ZLIB(compression)
                ) as out_file:
                    out_file[tree_name] = chunks

            current_entries = len(chunks[mixer.branch_names[0]])
            total_processed += current_entries
            pbar.update(current_entries)
            iteration += 1

            del chunks

        pbar.close()

        n_input = n_valid + len(mixer.skipped_files)
        n_ok = n_valid - len(mixer.failed_files)
        print(f"\n{'='*60}")
        print(f"Mixing complete!")
        print(f"  Output format         : {output_format}")
        print(f"  Output files created  : {iteration}")
        print(f"  Total entries written : {total_processed:,}")
        print(f"  Entries filtered (dR) : {total_filtered:,}")
        print(f"  Input files total     : {n_input}")
        print(f"  Successfully read     : {n_ok} / {n_valid}")

        if mixer.skipped_files:
            print(
                f"\n  WARNING: {len(mixer.skipped_files)} file(s) skipped at scan time:"
            )
            for p in mixer.skipped_files:
                print(f"    - {p}")

        if mixer.failed_files:
            print(
                f"\n  WARNING: {len(mixer.failed_files)} file(s) dropped due to read errors:"
            )
            for p in mixer.failed_files:
                print(f"    - {p}")

        if not mixer.skipped_files and not mixer.failed_files:
            print("  All input files processed successfully.")
        print("=" * 60)

    finally:
        mixer.shutdown()


def main():
    args = parse_args()

    input_format = _detect_format(args.input)
    output_format = args.format if args.format is not None else input_format

    if args.max_particles > 1:
        print(
            f"Truncating particle arrays to max {args.max_particles} particles per event"
        )

    mix_files(
        args.input,
        args.out_dir,
        args.chunk_size,
        args.threads,
        args.seed,
        shuffle=not args.no_shuffle,
        compression=args.compression,
        max_particles=args.max_particles,
        output_format=output_format,
        output_digits=args.output_digits,
        dr_filter=args.dR_filter,
    )


if __name__ == "__main__":
    main()

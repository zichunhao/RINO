import argparse
from pathlib import Path
import random
import uproot
import numpy as np
import awkward as ak
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
        description="Mix ROOT files with parallel processing"
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="List of input ROOT files"
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
    return parser.parse_args()


def shuffle_entries(data):
    """Shuffle entries in the data dictionary while maintaining correlations"""
    n_entries = len(data[list(data.keys())[0]])
    indices = np.random.permutation(n_entries)
    return {branch: data[branch][indices] for branch in data.keys()}


class ParallelMixer:
    def __init__(self, files, chunk_size, n_threads, tree_name="tree"):
        self.chunk_size = chunk_size
        self.tree_name = tree_name
        self.n_threads = n_threads
        self.active_files = []
        self.branch_names = None
        self.total_entries = 0

        # Initialize file information
        print("Reading file information...")
        for f in tqdm(files, desc="Scanning input files"):
            with uproot.open(f) as root_file:
                tree = root_file[tree_name]
                entries = tree.num_entries
                if entries > 0:
                    if self.branch_names is None:
                        self.branch_names = list(tree.keys())
                    self.active_files.append(FileInfo(f, entries))
                    self.total_entries += entries

        self.executor = ThreadPoolExecutor(max_workers=n_threads)

    def read_chunk(self, file_id: int, file_info: FileInfo) -> Optional[ChunkData]:
        """Read a chunk from a file"""
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
        """Get next chunk from each file using parallel processing"""
        if not self.active_files:
            return None

        # Randomize file order
        file_indices = list(range(len(self.active_files)))
        random.shuffle(file_indices)

        # Submit tasks for each file
        future_to_idx = {
            self.executor.submit(self.read_chunk, idx, self.active_files[idx]): idx
            for idx in file_indices
        }

        # Collect results
        chunks_data = {branch: [] for branch in self.branch_names}
        files_to_remove = set()
        chunks_read = 0

        for future in as_completed(future_to_idx):
            chunk = future.result()
            if chunk is not None:
                for branch in self.branch_names:
                    chunks_data[branch].append(chunk.data[branch])
                chunks_read += 1

                if self.active_files[chunk.file_id].is_complete:
                    files_to_remove.add(chunk.file_id)

        # Remove completed files
        for idx in sorted(files_to_remove, reverse=True):
            del self.active_files[idx]

        # Return concatenated chunks if we have any data
        if chunks_read > 0:
            return {
                branch: ak.concatenate(chunks) for branch, chunks in chunks_data.items()
            }
        return None

    def shutdown(self):
        """Shutdown the mixer"""
        self.executor.shutdown(wait=True)


def mix_files(
    input_files,
    output_dir,
    chunk_size,
    n_threads=4,
    seed=42,
    tree_name="tree",
    shuffle=True,
    compression=4,
):
    """Mix ROOT files with parallel processing"""
    random.seed(seed)
    np.random.seed(seed)

    # Setup paths
    input_files = [Path(f) for f in input_files]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mixer
    mixer = ParallelMixer(input_files, chunk_size, n_threads, tree_name)
    if not mixer.active_files:
        print("No valid input files found!")
        return

    try:
        iteration = 0
        total_processed = 0
        pbar = tqdm(total=mixer.total_entries, desc="Processing entries")

        while True:
            # Get chunks from all files
            chunks = mixer.get_next_chunks()
            if chunks is None:
                break

            # Shuffle entries if enabled
            if shuffle:
                chunks = shuffle_entries(chunks)

            # Write to output file with compression
            output_path = output_dir / f"mixed_{iteration}.root"
            with uproot.recreate(
                output_path, compression=uproot.ZLIB(compression)
            ) as out_file:
                out_file[tree_name] = chunks

            # Update progress
            current_entries = len(chunks[mixer.branch_names[0]])
            total_processed += current_entries
            pbar.update(current_entries)
            iteration += 1

            # Clean up
            del chunks

        pbar.close()
        print(
            f"\nMixing complete! Created {iteration} files with {total_processed:,} total entries."
        )

    finally:
        mixer.shutdown()


def main():
    args = parse_args()
    mix_files(
        args.input,
        args.out_dir,
        args.chunk_size,
        args.threads,
        args.seed,
        shuffle=not args.no_shuffle,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()

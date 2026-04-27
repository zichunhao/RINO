"""HDF5-based iterator for RINO-preprocessed JetClass data.

Reads from combined HDF5 files produced by baselines/mpmv2/scripts/make_jetclass.py
(main_rino_kinematics). Much faster than ROOT-based rino_iterator.py.

Produces the same (high, nodes, mask, label) tuple format as RINOJetClassIterator
for compatibility with the IteratorWrapper / RINOIteratorWrapper pipeline.

**Important performance note**: the naive implementation (one `h5[idx]` lookup
per sample) is pathologically slow on contended CephFS — ~0.03 it/s per DDP
rank, ~25 days per epoch. The root cause is that each per-sample indexed read
triggers a separate CephFS round trip. This version instead reads a
**contiguous chunk** of `chunk_size` rows per h5 call and yields samples from
the in-memory chunk, reducing h5 calls by ~chunk_size×. Since the combined
HDF5 files are already globally shuffled at preprocess time (via shuffle.py),
contiguous slices remain representative — same argument as the DINO dataloader.
"""

import h5py
import numpy as np


class RINOH5Iterator:
    """Iterator that reads RINO features from a combined HDF5 file in
    contiguous chunks.

    Returns per-sample tuples of (high, nodes, mask, label) where:
        - high: (4,) jet-level features (zeros if not available)
        - nodes: (n_csts, 7) RINO particle features
        - mask: (n_csts,) bool mask
        - label: (1,) class index
    """

    def __init__(
        self,
        dset: str,
        path: str = "PROJECT_ROOT/data/JetClass/mpm-rino/",
        n_nodes: int = 128,
        n_load: int = 20,  # unused, kept for interface compat
        processes: list = None,  # unused, HDF5 is already QCD-only
        max_files: int = None,  # unused
        features: list = None,  # unused
        chunk_size: int = 1000,  # contiguous rows per h5 call
    ):
        self.dset = dset
        self.n_nodes = n_nodes
        self.features = None  # interface compat
        self.chunk_size = chunk_size

        if dset == "train":
            h5_path = f"{path}/train_100M_combined_QCD.h5"
            self.n_samples = 10_000_000
        elif dset == "test":
            h5_path = f"{path}/test_20M_combined_QCD.h5"
            self.n_samples = 2_000_000
        else:
            h5_path = f"{path}/val_5M_combined_QCD.h5"
            self.n_samples = 500_000

        self.h5_path = h5_path
        self.file = None
        # Sibling *_tokens.h5 file with precomputed VQ-VAE code labels.
        # If present, __next__ yields a 5-tuple with the extra code_labels
        # array so Bert.preprocess_inputs can skip the runtime tokenize()
        # call (see baselines/mpmv1/scripts/precompute_tokens.py).
        import os
        _base, _ext = os.path.splitext(h5_path)
        self.tokens_path = f"{_base}_tokens{_ext}"
        self.tokens_file = None
        self.has_tokens = os.path.exists(self.tokens_path)
        with h5py.File(h5_path, "r") as f:
            self.n_jets = len(f["csts"])
        self.n_samples = min(self.n_samples, self.n_jets)

        # Under DDP, each rank should iterate only 1/world_size of the data
        # per epoch. The H5 is already globally shuffled and each rank picks
        # random contiguous chunks, so dividing n_samples is sufficient —
        # no per-rank file sharding needed. This brings step count from
        # 20,000 (all ranks × full data) down to ~3,333 (matching MPMv2).
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
            self.n_samples = self.n_samples // world_size

        # Contiguous-chunk cache
        self._chunk_csts = None
        self._chunk_mask = None
        self._chunk_labels = None
        self._chunk_code_labels = None  # (B, n_nodes) int32 when has_tokens
        self._chunk_pos = 0  # index into the current chunk
        self._chunk_len = 0  # number of rows currently in the chunk

        # For val/test: linear chunk start positions; for train: random starts.
        self._next_linear_start = 0
        # Per-worker RNG (seeded deterministically so DDP ranks get different
        # streams, but reproducible across runs given the same rank).
        self._rng = np.random.default_rng()

        # DEBUG instrumentation
        self._total_samples_yielded = 0
        self._total_chunks_loaded = 0

        print(
            f"[RINOH5Iterator] Opened {h5_path}: {self.n_jets} jets, "
            f"chunk_size={chunk_size}, has_tokens={self.has_tokens}",
            flush=True,
        )

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r", swmr=True)
        if self.has_tokens and self.tokens_file is None:
            self.tokens_file = h5py.File(self.tokens_path, "r", swmr=True)

    def get_nclasses(self):
        return 10

    def _load_chunk(self):
        """Load a contiguous chunk of rows from h5 into memory."""
        import os
        import time
        t0 = time.time()
        self._ensure_open()
        if self._total_chunks_loaded < 5 or self._total_chunks_loaded % 50 == 0:
            print(
                f"[RINOH5Iterator dset={self.dset} pid={os.getpid()}] "
                f"_load_chunk #{self._total_chunks_loaded} starting",
                flush=True,
            )

        if self.dset == "train":
            # Random offset anywhere in the file. Since shuffle.py already
            # globally mixed events, contiguous chunks are representative.
            max_start = max(0, self.n_jets - self.chunk_size)
            start = int(self._rng.integers(0, max_start + 1))
        else:
            # Linear iteration for val/test — wrap around.
            if self._next_linear_start >= self.n_jets:
                self._next_linear_start = 0
            start = self._next_linear_start
            self._next_linear_start = start + self.chunk_size

        end = min(start + self.chunk_size, self.n_jets)

        # Single contiguous slice per field — far fewer round trips to CephFS
        # than per-sample indexed reads.
        self._chunk_csts = (
            self.file["csts"][start:end, : self.n_nodes, :7].astype(np.float32)
        )
        self._chunk_mask = (
            self.file["mask"][start:end, : self.n_nodes].astype(bool)
        )
        self._chunk_labels = self.file["labels"][start:end].astype(np.float32)
        if self.has_tokens:
            self._chunk_code_labels = (
                self.tokens_file["code_labels"][start:end, : self.n_nodes].astype(
                    np.int64
                )
            )

        self._chunk_pos = 0
        self._chunk_len = end - start
        self._total_chunks_loaded += 1
        if self._total_chunks_loaded <= 5 or self._total_chunks_loaded % 50 == 0:
            import os, time as _t
            print(
                f"[RINOH5Iterator dset={self.dset} pid={os.getpid()}] "
                f"loaded chunk #{self._total_chunks_loaded} "
                f"rows [{start}:{end}] ({end-start}) in {_t.time()-t0:.2f}s",
                flush=True,
            )

    def __next__(self):
        if self._chunk_pos >= self._chunk_len:
            self._load_chunk()

        i = self._chunk_pos
        self._chunk_pos += 1

        nodes = self._chunk_csts[i]
        mask = self._chunk_mask[i]
        label = self._chunk_labels[i : i + 1]  # (1,) slice keeps 1D shape

        # Jet-level features (zeros — VQ-VAE and MPM don't use high_dim)
        high = np.zeros(0, dtype=np.float32)

        self._total_samples_yielded += 1
        if self._total_samples_yielded in (1, 10, 100, 1000, 10000, 100000):
            import os
            print(
                f"[RINOH5Iterator dset={self.dset} pid={os.getpid()}] "
                f"yielded {self._total_samples_yielded} samples "
                f"(chunk_pos={self._chunk_pos}/{self._chunk_len})",
                flush=True,
            )

        if self.has_tokens:
            code_labels = self._chunk_code_labels[i]  # (n_nodes,) int64
            return high, nodes, mask, label, code_labels
        return high, nodes, mask, label

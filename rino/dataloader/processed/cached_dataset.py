import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .config import _DataloaderConfig
from .processors import run_processors
from .dataset import (
    Batch,
    class_and_aux,
    load_raw_data,
    parse_requested_keys,
    sequence_and_mask,
    process_views,
    identity_collate,
)
from utils.logger import LOGGER
from accelerate import Accelerator

CACHE_LOCK = threading.Lock()


class CachedJetDataset(Dataset):
    """
    Cached version of JetDataset for preprocessed jet data stored in HDF5 format.

    This dataset loads data from HDF5 files with memory caching for improved performance.
    Files are preloaded into memory based on available cache size, with thread-safe
    operations and optional parallel loading.
    """

    def __init__(
        self,
        config: _DataloaderConfig,
        split: str = "train",
        cache_size: float = 32.0,
        size_multiplier: float = 1.5,  # HDF5 files may expand more in memory
        preload_workers: int = 1,
        accelerator: Accelerator | None = None,
        **kwargs,
    ):
        """
        Create a cached dataset for preprocessed jet data.

        Parameters
        ----------
        config : _DataloaderConfig
            The configuration of the dataloader
        split : str, optional
            Which split to use, by default 'train'
        cache_size : float, optional
            Maximum cache size in gigabytes, by default 32.0
        size_multiplier: float, optional
            Multiplier to estimate memory size from file size, default 1.5
        preload_workers: int, optional
            Number of worker threads for preloading data. If <= 1, load sequentially.
        **kwargs
            Overwrite the configuration with these values
        """
        config = copy.deepcopy(config)
        config.update(kwargs)
        self.config = config
        self.preload_workers = max(1, preload_workers)
        
        if accelerator is not None:
            self.is_main_process = accelerator.is_main_process
            self.process_index = accelerator.process_index
            self.num_processes = accelerator.num_processes
        else:
            self.is_main_process = True
            self.process_index = 0
            self.num_processes = 1

        # Setup paths
        _path = config.paths[split]
        if isinstance(_path, str):
            paths = []
            for p in config.patterns:
                paths.extend(list(Path(_path).glob(p)))
            self.paths = sorted(paths)
        elif isinstance(_path, list):
            self.paths = sorted([Path(x) for x in _path])
        else:
            raise ValueError(f"Invalid path type: {type(_path)}")

        # Initialize cache with thread-safe structures
        self.cache_size_bytes = int(cache_size * 1024 * 1024 * 1024)
        self.size_multiplier = size_multiplier
        self.cache: dict[Path, dict[str, torch.Tensor]] = {}
        self.cache_sizes: dict[Path, int] = {}
        self.total_cache_size = 0

        # Setup dataset
        self.sizes = self._cache_dataset_sizes(config, self.paths)
        self.batch_indices = self._init_batch_indices(config, self.paths, self.sizes)
        self._len = len(self.batch_indices)
        self.require_names = parse_requested_keys(config)

        # Preload data based on file sizes
        if self.is_main_process:
            # Main process loads everything normally
            LOGGER.info(f"Main process {self.process_index}: Preloading dataset with {len(self.paths)} files")
            self._preload_data()
            LOGGER.info(f"Main process loaded {len(self.cache)} files into cache ({self.total_cache_size/1e9:.2f}GB)")
        else:
            # Worker processes: empty dataset, no loading
            self.cache = {}
            self.cache_sizes = {}
            self.total_cache_size = 0
            LOGGER.info(f"Worker process {self.process_index}: Using empty dataset (dispatch_batches=True mode)")

    def _estimate_memory_size(self, path: Path) -> int:
        """Estimate memory requirements based on file size"""
        file_size = os.path.getsize(path)
        estimated_size = int(file_size * self.size_multiplier)
        return estimated_size

    def _preload_single_file(
        self, path: Path, estimated_size: int
    ) -> Optional[tuple[Path, dict[str, torch.Tensor], int]]:
        """Preload a single file and return its data if successful"""
        try:
            with CACHE_LOCK:
                if (self.total_cache_size + estimated_size) > self.cache_size_bytes:
                    LOGGER.debug(
                        f"Skip {path}: estimate={estimated_size/1e9:.2f}GB > "
                        f"remaining={((self.cache_size_bytes - self.total_cache_size)/1e9):.2f}GB"
                    )
                    return None

            # Load the entire file into memory
            with h5py.File(path, "r") as f:
                # Get the file size (number of samples)
                first_key = list(f.keys())[0]
                total_samples = f[first_key].shape[0]

                # Load all required data
                data = {}
                for name in self.require_names:
                    if name in f:
                        # Load entire dataset and convert to tensor
                        data_array = f[name][:]
                        data[name] = torch.from_numpy(data_array)
                    else:
                        LOGGER.warning(f"Required key '{name}' not found in {path}")

                # Calculate actual memory usage
                actual_size = sum(tensor.nbytes for tensor in data.values())

                with CACHE_LOCK:
                    if (self.total_cache_size + actual_size) <= self.cache_size_bytes:
                        return (path, data, actual_size)
                    else:
                        LOGGER.warning(
                            f"Skip {path}: actual={actual_size/1e9:.2f}GB > "
                            f"remaining={((self.cache_size_bytes - self.total_cache_size)/1e9):.2f}GB"
                        )
                        return None

        except Exception as e:
            LOGGER.error(f"Failed to preload {path}: {str(e)}")
            return None

    def _preload_data(self):
        """Preload data using file size based estimation with optional threading"""
        # Sort files by size for better cache utilization
        paths_with_sizes = [(p, self._estimate_memory_size(p)) for p in self.paths]
        paths_with_sizes.sort(key=lambda x: x[1])  # Sort by estimated size

        if self.preload_workers <= 1:
            # Sequential loading
            for path, estimated_size in paths_with_sizes:
                result = self._preload_single_file(path, estimated_size)
                if result:
                    path, data, actual_size = result
                    self.cache[path] = data
                    self.cache_sizes[path] = actual_size
                    self.total_cache_size += actual_size
        else:
            # Parallel loading with ThreadPoolExecutor
            LOGGER.info(f"Preloading data with {self.preload_workers} workers")
            with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
                future_to_path = {
                    executor.submit(self._preload_single_file, path, size): path
                    for path, size in paths_with_sizes
                }

                for future in as_completed(future_to_path):
                    result = future.result()
                    if result:
                        path, data, actual_size = result
                        with CACHE_LOCK:
                            self.cache[path] = data
                            self.cache_sizes[path] = actual_size
                            self.total_cache_size += actual_size
                            LOGGER.debug(
                                f"Cached {path} (Size: {actual_size/1e9:.2f}GB, "
                                f"Total: {self.total_cache_size/1e9:.2f}/{self.cache_size_bytes/1e9:.2f}GB)"
                            )

        LOGGER.info(
            f"Total cached: {self.total_cache_size/1e9:.2f}GB / {self.cache_size_bytes/1e9:.2f}GB "
            f"({len(self.cache)}/{len(self.paths)} files)"
        )

    def _get_data(
        self, path: Path, idx_start: int, idx_end: int
    ) -> dict[str, torch.Tensor]:
        """Get data either from cache or load from disk"""
        if path in self.cache:
            return {k: v[idx_start:idx_end] for k, v in self.cache[path].items()}
        return load_raw_data(path, self.require_names, idx_start, idx_end, skip_missing=True)

    def __getitem__(self, idx: int) -> Batch:
        """Load and process a batch of data."""
        path, f_start, f_end = self.batch_indices[idx]

        # Get data either from cache or disk
        data = self._get_data(path, f_start, f_end)

        # Run processors
        data = run_processors(self.config, data)

        # Create sequence, mask, class, aux, and views
        seq, mask, seq_len = sequence_and_mask(self.config, data)
        class_, aux = class_and_aux(self.config, data)
        views = process_views(self.config, data, seq_len)

        # Create and return batch
        return Batch(sequence=seq, mask=mask, class_=class_, aux=aux, views=views)

    def __len__(self) -> int:
        return self._len

    @staticmethod
    def _init_batch_indices(
        config: _DataloaderConfig, paths: list[Path], sizes: NDArray[np.int64]
    ) -> list[tuple[Path, int, int]]:
        """Initialize batch indices, ensuring each batch comes from a single file."""
        batch_indices: list[tuple[Path, int, int]] = []

        for path, file_size in zip(paths, sizes):
            # Create batches for this file
            num_full_batches = file_size // config.batch_size
            remainder = file_size % config.batch_size

            # Add full batches
            for batch_idx in range(num_full_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = start_idx + config.batch_size
                batch_indices.append((path, start_idx, end_idx))

            # Add remainder as incomplete batch if it exists
            if remainder > 0:
                start_idx = num_full_batches * config.batch_size
                end_idx = start_idx + remainder
                batch_indices.append((path, start_idx, end_idx))

        return batch_indices

    @staticmethod
    def _cache_dataset_sizes(
        config: _DataloaderConfig, paths: list[Path]
    ) -> NDArray[np.int64]:
        """Cache the size of each dataset file."""
        sizes = np.empty(len(paths), dtype=np.int64)
        for i, p in enumerate(paths):
            with h5py.File(p, "r") as f:
                # Use the first dataset's length
                first_key = list(f.keys())[0]
                sizes[i] = f[first_key].shape[0]
        return sizes

    def __repr__(self) -> str:
        cached_files = len(self.cache)
        total_files = len(self.paths)
        return (
            f"CachedJetDataset(config={self.config}, "
            f"cache_size={self.total_cache_size/1e9:.2f}GB, "
            f"cached_files={cached_files}/{total_files}, "
            f"preload_workers={self.preload_workers})"
        )

    def __str__(self) -> str:
        return f"CachedJetDataset(split={self.config.paths}, cached={len(self.cache)}/{len(self.paths)} files)"


class CachedJetDataLoader:
    def __new__(cls, dataset: CachedJetDataset, **kwargs):
        # Force batch_size=1 and identity collate function
        kwargs["batch_size"] = 1
        kwargs["collate_fn"] = identity_collate
        return torch.utils.data.DataLoader(dataset, **kwargs)
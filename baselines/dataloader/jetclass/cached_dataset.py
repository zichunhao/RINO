import copy
from math import ceil, floor
from pathlib import Path
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import awkward as ak
import numpy as np
import uproot as up
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .config import _DataloaderConfig
from .processors import run_processors
from .dataset import (
    Batch,
    class_and_aux,
    load_raw_data,
    maybe_concat_batch_slices,
    parse_requested_keys,
    sequence_and_mask,
    process_views,
)
from utils.logger import LOGGER
from accelerate import Accelerator

CACHE_LOCK = threading.Lock()


class CachedJetDataset(Dataset):
    def __init__(
        self,
        config: _DataloaderConfig,
        split="train",
        cache_size: float = 32.0,
        size_multiplier: float = 1.2,  # ROOT files are typically compressed
        preload_workers: int = 1,
        accelerator: Accelerator | None = None,
        **kwargs,
    ):
        """Create a dataset for the jet data with memory caching.

        Parameters
        ----------
        config : DataloaderConfig
            The configuration of the dataloader
        split : str, optional
            Which split to use, by default 'train'
        cache_size : float, optional
            Maximum cache size in gigabytes, by default 32.0
        size_multiplier: float, optional
            Multiplier to estimate memory size from file size, default 1.2
        preload_workers: int, optional
            Number of worker threads for preloading data. If <= 1, load sequentially.
        **kwargs
            Overwrite the configuration with these values
        """
        config = copy.deepcopy(config)
        config.update(kwargs)
        self.config = config
        self.preload_workers = max(1, preload_workers)
        
        # Store accelerator info (for pickling compatibility)
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
            # self.paths = sorted(Path(_path).glob(config.pattern))
            for p in config.patterns:
                paths.extend(list(Path(_path).glob(p)))
            self.paths = sorted(paths)
        elif isinstance(_path, list):
            self.paths = sorted([Path(x) for x in _path])

        # Initialize cache with thread-safe structures
        self.cache_size_bytes = int(cache_size * 1024 * 1024 * 1024)
        self.size_multiplier = size_multiplier
        self.cache: dict[Path, dict[str, ak.Array]] = {}
        self.cache_sizes: dict[Path, int] = {}
        self.total_cache_size = 0

        # Setup dataset
        self.sizes = self.cache_dataset_sizes(config, self.paths)
        self.batch_split_idx = self.init_batch_split(config, self.paths, self.sizes)
        self._len = len(self.batch_split_idx)
        self.require_names = parse_requested_keys(config)

        # Preload data based on file sizes
        # DISPATCH BATCHES LOGIC: Only main process loads data
        if self.is_main_process:
            # Main process loads everything normally
            self._preload_data()
            LOGGER.info(f"Main process loaded {len(self.cache)} files into cache ({self.total_cache_size/1e9:.2f}GB)")
        else:
            # Worker processes: empty dataset, no loading
            self.cache = {}
            self.cache_sizes = {}
            self.total_cache_size = 0
            LOGGER.info(f"Worker process {self.process_index}: Using empty dataset (dispatch_batches=True mode)")
            
        # Synchronize all processes
        if accelerator is not None:
            accelerator.wait_for_everyone()

    def _estimate_memory_size(self, path: Path) -> int:
        """Estimate memory requirements based on file size"""
        file_size = os.path.getsize(path)
        estimated_size = int(file_size * self.size_multiplier)
        return estimated_size

    def _preload_single_file(
        self, path: Path, estimated_size: int
    ) -> Optional[tuple[Path, dict[str, ak.Array], int]]:
        """Preload a single file and return its data if successful"""
        try:
            with CACHE_LOCK:
                if (self.total_cache_size + estimated_size) > self.cache_size_bytes:
                    LOGGER.debug(
                        f"Skip {path}: estimate={estimated_size/1e9:.2f}GB > "
                        f"remaining={((self.cache_size_bytes - self.total_cache_size)/1e9):.2f}GB"
                    )
                    return None

            with up.open(path) as f:
                total_events = f[self.config.length_from]._members["fEntries"]
                data = load_raw_data(path, self.require_names, 0, int(total_events))
                actual_size = sum(arr.nbytes for arr in data.values())

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
    ) -> dict[str, ak.Array]:
        """Get data either from cache or load from disk"""
        if path in self.cache:
            return {k: v[idx_start:idx_end] for k, v in self.cache[path].items()}
        return load_raw_data(path, self.require_names, idx_start, idx_end)

    def __getitem__(self, idx):
        if not self.is_main_process:
            raise RuntimeError(
                f"Worker process {self.process_index} should not call __getitem__ "
                "when using dispatch_batches=True. Only main process loads data."
            )
        
        batch_split_idx = self.batch_split_idx[idx]
        batch_slices: list[Batch] = []

        for path, f_start, f_end in batch_split_idx:
            data = self._get_data(path, f_start, f_end)
            data = run_processors(self.config, data)

            # Get sequence, mask, and sequence length
            seq, mask, seq_len = sequence_and_mask(self.config, data)
            class_, aux = class_and_aux(self.config, data)
            # Pass sequence length to process_views
            views = process_views(self.config, data, seq_len)
            _batch = Batch(
                sequence=seq,
                mask=mask,
                class_=class_,
                aux=aux,
                views=views,
            )
            batch_slices.append(_batch)

        return maybe_concat_batch_slices(batch_slices)

    def __len__(self):
        return self._len

    @staticmethod
    def init_batch_split(
        config: _DataloaderConfig, paths: list[Path], sizes: NDArray[np.int64]
    ):
        if config.drop_last == 2:
            sizes = sizes // config.batch_size_atomic * config.batch_size_atomic

        cum_sizes = np.concatenate([[0], np.cumsum(sizes)])
        if config.drop_last == 0:
            _len = floor(cum_sizes[-1] / config.batch_size_atomic)
        else:
            _len = ceil(cum_sizes[-1] / config.batch_size_atomic)

        batch_splits = []
        batch_starts = np.arange(_len) * config.batch_size_atomic
        batch_ends = batch_starts + config.batch_size_atomic

        for start, end in zip(batch_starts, batch_ends):
            f_idx0 = np.searchsorted(cum_sizes, start, side="right") - 1
            f_idx1 = np.searchsorted(cum_sizes, end, side="left") - 1
            f_idx0 = max(0, f_idx0)

            _splits = []
            for f_idx in range(f_idx0, f_idx1 + 1):
                f_start = max(0, start - cum_sizes[f_idx])
                f_end = min(end - cum_sizes[f_idx], sizes[f_idx])
                _splits.append((paths[f_idx], f_start, f_end))

            batch_splits.append(tuple(_splits))

        return batch_splits

    @staticmethod
    def cache_dataset_sizes(config: _DataloaderConfig, paths) -> NDArray[np.int64]:
        sizes = np.empty(len(paths), dtype=np.int64)
        for i, p in enumerate(paths):
            with up.open(p) as f:
                sizes[i] = f[config.length_from]._members["fEntries"]
        return sizes

    def __repr__(self):
        cached_files = len(self.cache)
        total_files = len(self.paths)
        return (
            f"CachedJetDataset(config={self.config}, "
            f"cache_size={self.total_cache_size/1e9:.2f}GB, "
            f"cached_files={cached_files}/{total_files}, "
            f"preload_workers={self.preload_workers})"
        )

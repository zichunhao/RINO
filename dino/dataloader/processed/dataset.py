import copy
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import torch
import numpy as np
import h5py
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .config import _DataloaderConfig
from .processors import run_processors


class Batch(TypedDict):
    sequence: torch.Tensor
    mask: torch.Tensor
    class_: torch.Tensor
    aux: dict[str, torch.Tensor]
    views: dict[str, dict[str, torch.Tensor]]


def clip_and_pad(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Clip or pad the tensor to the specified sequence length."""
    if tensor.shape[1] < seq_len:
        # Pad the tensor (to the right) to the required sequence length
        tensor = torch.nn.functional.pad(tensor, (0, seq_len - tensor.shape[1]))
    elif tensor.shape[1] > seq_len:
        # Clip the tensor to the required sequence length
        tensor = tensor[:, :seq_len]
    return tensor


def parse_requested_keys(config: _DataloaderConfig) -> list[str]:
    """Get the list of keys that are required to be loaded from the raw data"""
    have_name = set()
    require_names = set()

    for processor in config.transformations:
        all_requires = processor.args + list(processor.kwargs.values())
        all_additional_requires = [x for x in all_requires if x not in have_name]
        require_names.update(all_additional_requires)
        have_name.update(all_additional_requires)
        have_name.update(processor.outputs)

    for name in (
        config.outputs.sequence
        + [config.outputs.sequence_mask]
        + config.outputs.class_
        + config.outputs.aux
    ):
        if name not in have_name:
            require_names.add(name)

    # Add view-related keys
    if hasattr(config.outputs, "views") and config.outputs.views:
        for view_config in config.outputs.views.values():
            require_names.add(view_config["mask"])
            for feature in view_config["features"]:
                if feature not in have_name:
                    require_names.add(feature)
            # Add jets if specified in view config
            if "jets" in view_config and view_config["jets"] is not None:
                for jet_feature in view_config["jets"]:
                    if jet_feature not in have_name:
                        require_names.add(jet_feature)

    return list(require_names)


def load_raw_data(
    path: Path,
    require_name: list[str],
    idx_start: int,
    idx_end: int,
    skip_missing: bool = False,
) -> dict[str, torch.Tensor]:
    """Load from HDF5 file and return the data as tensors. No processing is done here."""
    buf = {}
    with h5py.File(path, "r") as f:
        for name in require_name:
            if name in f:
                # Load data slice and convert to tensor
                data_slice = f[name][idx_start:idx_end]
                buf[name] = torch.from_numpy(data_slice)
            else:
                # could be from processor
                if skip_missing:
                    pass
                else:
                    raise KeyError(f"Required key '{name}' not found in file {path}")

    return buf


def process_views(
    config: _DataloaderConfig, data: dict[str, torch.Tensor], seq_len: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Process views data and return dictionary with features, masks, and optionally jets."""
    if not hasattr(config.outputs, "views") or not config.outputs.views:
        return {}

    # Get batch size from the main sequence data
    batch_size = None
    if config.outputs.sequence:
        first_seq_key = config.outputs.sequence[0]
        batch_size = data[first_seq_key].shape[0]

    views_dict = {}
    for view_name, view_config in config.outputs.views.items():
        view_data = {}

        # Use view-specific seq_len if provided, otherwise use global seq_len
        # Handle both dict and dataclass access patterns
        if isinstance(view_config, dict):
            view_seq_len = view_config.get("seq_len", seq_len)
            features_key = view_config["features"]
            mask_key = view_config["mask"]
            jets_key = view_config.get("jets")
        else:
            view_seq_len = getattr(view_config, "seq_len", None) or seq_len
            features_key = view_config.features
            mask_key = view_config.mask
            jets_key = getattr(view_config, "jets", None)

        # Get features
        feature_tensors = []
        for feature in features_key:
            tensor = data[feature]

            if tensor.dim() > 1:
                if tensor.shape[1] < view_seq_len:
                    pad_size = view_seq_len - tensor.shape[1]
                    tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=False)
                elif tensor.shape[1] > view_seq_len:
                    tensor = tensor[:, :view_seq_len]

            feature_tensors.append(tensor)

        features = torch.stack(feature_tensors, dim=-1)
        view_data["features"] = features

        # Get mask
        mask = data[mask_key]

        # Slice mask to match batch size first
        if batch_size is not None and mask.shape[0] > batch_size:
            mask = mask[:batch_size]

        # Handle sequence length for mask
        if mask.dim() > 1:
            if mask.shape[1] < view_seq_len:
                pad_size = view_seq_len - mask.shape[1]
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
            elif mask.shape[1] > view_seq_len:
                mask = mask[:, :view_seq_len]

        mask = mask.bool()
        view_data["mask"] = mask

        # Get jets if specified
        if jets_key is not None:
            jet_tensors = []
            for jet_feature in jets_key:
                tensor = data[jet_feature]
                # Jets are typically at the event level, so no sequence padding needed
                # But we may need to handle batch size consistency
                if batch_size is not None and tensor.shape[0] > batch_size:
                    tensor = tensor[:batch_size]

                # Ensure tensor has proper dimensionality
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() == 1 and batch_size is not None:
                    # If it's 1D but we have a batch, ensure it matches batch size
                    if tensor.shape[0] != batch_size:
                        # This might indicate a configuration issue
                        raise ValueError(
                            f"Jet feature {jet_feature} has shape {tensor.shape} "
                            f"but expected batch size {batch_size}"
                        )

                jet_tensors.append(tensor)

            if jet_tensors:
                jets = torch.stack(jet_tensors, dim=-1)
                view_data["jets"] = jets

        views_dict[view_name] = view_data

    return views_dict


def sequence_and_mask(
    config: _DataloaderConfig,
    data: dict[str, torch.Tensor],
    update_config_seq_len: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Concatenate the sequence data and get mask, return the sequence, mask, and sequence length"""
    if len(config.outputs.sequence) == 0:
        return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.bool), 0

    var0 = config.outputs.sequence[0]
    if config.seq_pad_strategy.name == "fixed":
        seq_len = config.max_seq_length
    elif config.seq_pad_strategy.name == "max":
        seq_len = min(data[var0].shape[1], config.max_seq_length)
    else:
        # no truncation, use the maximum sequence length from the data
        seq_len = max(data[var0].shape[1], config.max_seq_length)
        if update_config_seq_len and seq_len > config.max_seq_length:
            config.max_seq_length = seq_len

    tensors = []
    for key in config.outputs.sequence:
        tensor = data[key][:, :seq_len]  # This preserves the batch dimension
        tensor = clip_and_pad(tensor, seq_len)
        tensors.append(tensor)

    # Stack sequence tensors
    seq = torch.stack(tensors, dim=-1)  # This creates [batch, seq_len, features]

    # Get mask and clip to seq_len
    mask = data[config.outputs.sequence_mask]
    mask = clip_and_pad(mask, seq_len)

    return seq, mask, seq_len


def class_and_aux(
    config: _DataloaderConfig, data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Concatenate the class data, and fill the aux data into a dictionary; return the class and aux"""
    class_tensors = []
    for key in config.outputs.class_:
        tensor = data[key]
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        class_tensors.append(tensor)

    auxs = {key: data[key] for key in config.outputs.aux}

    if class_tensors:
        class_output = torch.stack(class_tensors, dim=-1)
    else:
        class_output = torch.empty(0)

    return class_output, auxs


class JetDataset(Dataset):
    """
    Dataset for preprocessed jet data stored in HDF5 format.

    This dataset loads data from HDF5 files in batches, with each batch coming from
    a single file only. No shuffling is done here - users should shuffle the order
    in which the dataset is accessed (e.g., by using a DataLoader with shuffle=True).

    Every time __getitem__ is called, the dataset opens the HDF5 file, loads and
    processes the data, with no persistent file handlers kept.
    """

    def __init__(self, config: _DataloaderConfig, split: str = "train", **kwargs):
        """
        Create a dataset for preprocessed jet data.

        Parameters
        ----------
        config : _DataloaderConfig
            The configuration of the dataloader
        split : str, optional
            Which split to use, by default 'train'. The split should have a key
            in the `paths` dictionary of the configuration.
        **kwargs
            Overwrite the configuration with these values
        """
        config = copy.deepcopy(config)
        config.update(kwargs)
        self.config = config

        # Get file paths
        _path = config.paths[split]
        if isinstance(_path, str):
            paths = []
            for p in config.patterns:
                paths.extend(list(Path(_path).glob(p)))
            self.paths = sorted(paths)
        elif isinstance(_path, list):
            self.paths = [Path(x) for x in _path]
        else:
            raise ValueError(f"Invalid path type: {type(_path)}")

        # Cache dataset information
        self.sizes = self._cache_dataset_sizes(config, self.paths)
        self.batch_indices = self._init_batch_indices(config, self.paths, self.sizes)
        self._len = len(self.batch_indices)
        self.require_names = parse_requested_keys(config)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Batch:
        """Load and process a batch of data from a single file."""
        path, f_start, f_end = self.batch_indices[idx]

        # Load raw data from HDF5
        data = load_raw_data(path, self.require_names, f_start, f_end)

        # Run processors
        data = run_processors(self.config, data)

        # Create sequence, mask, class, aux, and views
        seq, mask, seq_len = sequence_and_mask(self.config, data)
        class_, aux = class_and_aux(self.config, data)
        views = process_views(self.config, data, seq_len)

        # Create and return batch
        return Batch(sequence=seq, mask=mask, class_=class_, aux=aux, views=views)

    def __repr__(self) -> str:
        return f"JetDataset(config={self.config}, paths={self.paths})"

    def __str__(self) -> str:
        return f"JetDataset(split={self.config.paths}, num_files={len(self.paths)})"

    @staticmethod
    def _init_batch_indices(
        config: _DataloaderConfig, paths: list[Path], sizes: NDArray[np.int64]
    ) -> list[tuple[Path, int, int]]:
        """Initialize batch indices, ensuring each batch comes from a single file.

        When ``batch_size_atomic`` is active (0 < atomic < batch_size), indices are
        generated with the atomic size so the DataLoader can concatenate multiple
        __getitem__ results into a full batch via ``maybe_concat_batch_slices``.
        """
        atomic = _effective_atomic(config)
        batch_indices: list[tuple[Path, int, int]] = []

        for path, file_size in zip(paths, sizes):
            num_full = file_size // atomic
            remainder = file_size % atomic

            for batch_idx in range(num_full):
                start_idx = batch_idx * atomic
                batch_indices.append((path, start_idx, start_idx + atomic))

            if remainder > 0:
                start_idx = num_full * atomic
                batch_indices.append((path, start_idx, start_idx + remainder))

        return batch_indices

    @staticmethod
    def _cache_dataset_sizes(
        config: _DataloaderConfig, paths: list[Path]
    ) -> NDArray[np.int64]:
        """Cache the size of each dataset file."""
        sizes = np.empty(len(paths), dtype=np.int64)
        for i, p in enumerate(paths):
            with h5py.File(p, "r") as f:
                # Fallback: use the first dataset's length
                first_key = list(f.keys())[0]
                sizes[i] = f[first_key].shape[0]
        return sizes


def _effective_atomic(config: _DataloaderConfig) -> int:
    """Return the effective atomic batch size.

    Returns ``config.batch_size`` when the atomic path is disabled
    (``batch_size_atomic < 0``, ``== batch_size``, or ``> batch_size``).
    """
    bsa = getattr(config, "batch_size_atomic", -1)
    if bsa < 0 or bsa >= config.batch_size:
        return config.batch_size
    return bsa


def maybe_concat_batch_slices(batch_slices: Sequence[Batch]) -> Batch:
    """Concatenate atomic batch slices into a single batch.

    If only one slice is provided it is returned unchanged.
    Sequences/masks with differing lengths are zero-padded on the right.
    """
    assert len(batch_slices) > 0
    if len(batch_slices) == 1:
        return batch_slices[0]

    seqs = [x["sequence"] for x in batch_slices]
    masks = [x["mask"] for x in batch_slices]

    max_seq_len = max(s.shape[1] for s in seqs)
    padded_seqs, padded_masks = [], []
    for seq, mask in zip(seqs, masks):
        pad = max_seq_len - seq.shape[1]
        if pad > 0:
            seq = torch.nn.functional.pad(seq, (0, 0, 0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad), value=False)
        padded_seqs.append(seq)
        padded_masks.append(mask)

    new_seq = torch.cat(padded_seqs, dim=0)
    new_mask = torch.cat(padded_masks, dim=0)
    new_class_ = torch.cat([x["class_"] for x in batch_slices], dim=0)

    keys = list(batch_slices[0]["aux"].keys())
    new_aux = {k: torch.cat([x["aux"][k] for x in batch_slices], dim=0) for k in keys}

    new_views: dict[str, dict[str, torch.Tensor]] = {}
    if batch_slices[0]["views"]:
        for view_name in batch_slices[0]["views"]:
            vfeats = [x["views"][view_name]["features"] for x in batch_slices]
            vmasks = [x["views"][view_name]["mask"] for x in batch_slices]

            max_vlen = max(f.shape[1] for f in vfeats)
            padded_vf, padded_vm = [], []
            for feat, vmask in zip(vfeats, vmasks):
                pad = max_vlen - feat.shape[1]
                if pad > 0:
                    feat = torch.nn.functional.pad(feat, (0, 0, 0, pad))
                    vmask = torch.nn.functional.pad(vmask, (0, pad), value=False)
                padded_vf.append(feat)
                padded_vm.append(vmask)

            new_view_data: dict[str, torch.Tensor] = {
                "features": torch.cat(padded_vf, dim=0),
                "mask": torch.cat(padded_vm, dim=0),
            }
            if "jets" in batch_slices[0]["views"][view_name]:
                new_view_data["jets"] = torch.cat(
                    [x["views"][view_name]["jets"] for x in batch_slices], dim=0
                )
            new_views[view_name] = new_view_data

    return Batch(
        sequence=new_seq, mask=new_mask, class_=new_class_, aux=new_aux, views=new_views
    )


def identity_collate(batches):
    """Return the first (and only) item without adding batch dimension"""
    return batches[0]


class JetDataLoader:
    def __new__(cls, dataset: JetDataset, **kwargs):
        atomic = _effective_atomic(dataset.config)
        if atomic == dataset.config.batch_size:
            # Short path: one __getitem__ == one full batch
            kwargs["batch_size"] = 1
            kwargs["collate_fn"] = identity_collate
        else:
            kwargs["batch_size"] = dataset.config.batch_size // atomic
            kwargs["collate_fn"] = maybe_concat_batch_slices
        return DataLoader(dataset, **kwargs)

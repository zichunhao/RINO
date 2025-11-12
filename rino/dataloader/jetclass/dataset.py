import copy
from collections.abc import Sequence
from functools import partial
from math import ceil, floor
from pathlib import Path
from typing import TypedDict

import awkward as ak
import numpy as np
import uproot as up
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from .config import _DataloaderConfig
from .processors import run_processors


class Batch(TypedDict):
    sequence: np.ndarray
    mask: NDArray[np.bool_]
    class_: np.ndarray
    aux: dict[str, np.ndarray | ak.Array]
    views: dict[str, dict[str, np.ndarray]]  # New field


def parse_requested_keys(config: _DataloaderConfig):
    """Get the list of keys that are required to be loaded from the raw data"""

    have_name = set()
    require_names = set()
    for processor in config.transformations:
        all_requires = processor.args + list(processor.kwargs.values())
        all_additional_requires = [x for x in all_requires if x not in have_name]
        require_names.update(all_additional_requires)
        have_name.update(all_additional_requires)
        have_name.update(processor.outputs)

    for name in config.outputs.sequence + config.outputs.class_ + config.outputs.aux:
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
    path: Path, require_name: list[str], idx_start: int, idx_end: int
) -> dict[str, ak.Array]:
    "Load from .root file and return the data as a dictionary. No processing is done here."
    buf = {}
    with up.open(path) as f:  # type: ignore
        for name in require_name:
            buf[name] = f[name].array(entry_start=idx_start, entry_stop=idx_end)  # type: ignore
    return buf


def process_views(
    config: _DataloaderConfig, data: dict[str, ak.Array], seq_len: int
) -> dict[str, dict[str, np.ndarray]]:
    """Process views data and return dictionary with features, masks, and optionally jets.

    Parameters
    ----------
    config : _DataloaderConfig
        The configuration of the dataloader
    data : dict[str, ak.Array]
        The raw data dictionary
    seq_len : int
        The sequence length to pad to (matching particle sequence length)

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Dictionary containing padded features, masks, and optionally jets for each view
    """
    if not hasattr(config.outputs, "views") or not config.outputs.views:
        return {}

    views_dict = {}
    for view_name, view_config in config.outputs.views.items():
        view_data = {}
        
        # Get features
        feature_arrays = []
        for feature in view_config["features"]:
            arr = data[feature]
            if isinstance(arr, ak.Array):
                # Pad the array to seq_len
                arr = ak.pad_none(arr, target=seq_len, clip=True)
                arr = arr.to_numpy()
                # Replace NaN values with 0
                arr = np.nan_to_num(arr, nan=0.0)
            feature_arrays.append(arr)
        features = np.stack(feature_arrays, axis=-1)
        view_data["features"] = features

        # Get mask
        mask = data[view_config["mask"]]
        if isinstance(mask, ak.Array):
            # Pad the mask to seq_len
            mask = ak.pad_none(mask, target=seq_len, clip=True) + 0.0
            mask = (mask.to_numpy() > 0.01).astype(np.bool_)
        view_data["mask"] = mask

        # Get jets if specified
        if "jets" in view_config and view_config["jets"] is not None:
            jet_arrays = []
            for jet_feature in view_config["jets"]:
                arr = data[jet_feature]
                if isinstance(arr, ak.Array):
                    # Convert to numpy - jets are typically event-level, no sequence padding
                    arr = arr.to_numpy()
                elif isinstance(arr, np.ndarray):
                    # Already numpy array
                    pass
                elif np.isscalar(arr):
                    # Convert scalar to array
                    arr = np.array([arr])
                else:
                    # Fallback conversion
                    arr = np.array(arr)
                
                # Ensure proper shape - jets should be event-level features
                if arr.ndim == 0:
                    arr = arr.reshape(-1)  # Convert 0D to 1D
                elif arr.ndim > 1:
                    # If multi-dimensional, flatten to event-level
                    arr = arr.reshape(arr.shape[0], -1)
                
                jet_arrays.append(arr)
            
            if jet_arrays:
                # Stack jet features along last dimension
                jets = np.stack(jet_arrays, axis=-1)
                view_data["jets"] = jets

        views_dict[view_name] = view_data

    return views_dict


def sequence_and_mask(
    config: _DataloaderConfig, data: dict[str, ak.Array]
) -> tuple[np.ndarray, np.ndarray, int]:
    "Concatenate and pad the sequence data, return the sequence, mask, and sequence length"

    if len(config.outputs.sequence) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.bool_), 0

    var0 = config.outputs.sequence[0]
    if config.seq_pad_strategy.name == "fixed":
        seq_len = config.max_seq_length
    elif config.seq_pad_strategy.name == "max":
        seq_len = int(min(np.max(ak.num(data[var0])), config.max_seq_length))
    else:
        seq_len = int(max(np.max(ak.num(data[var0])), config.max_seq_length))

    arrs = []
    for key in config.outputs.sequence:
        arr = ak.pad_none(data[key], target=seq_len, clip=True).to_numpy()
        value = np.nan_to_num(arr.data, nan=0.0)
        arrs.append(value)
        mask = ~arr.mask  # real: 1; padded: 0
    seq = np.stack(arrs, axis=-1)
    if not isinstance(mask, np.ndarray):  # type: ignore
        mask = np.ones_like(seq, dtype=np.bool_)
    return seq, mask, seq_len  # type: ignore


def class_and_aux(
    config: _DataloaderConfig, data: dict[str, ak.Array]
) -> tuple[np.ndarray, dict[str, np.ndarray | ak.Array]]:
    """Concatenate the class data, and fill the aux data into a dictionary; return the class and aux"""

    class_s = []
    for key in config.outputs.class_:
        if isinstance(data[key], np.ndarray):
            class_s.append(data[key])
        elif hasattr(data[key], "to_numpy"):
            class_s.append(data[key].to_numpy())
        elif np.isscalar(data[key]):
            class_s.append(np.array([data[key]]))
        else:
            # Fallback for other types, try converting to numpy array
            class_s.append(np.array(data[key]))

    auxs = {key: data[key] for key in config.outputs.aux}

    return np.stack(class_s, axis=-1), auxs  # type: ignore


def maybe_concat_batch_slices(batch_slices: Sequence[Batch]):
    "Concatenate the batch slices if there are more than one, otherwise return the only batch slice"

    assert len(batch_slices) > 0
    if len(batch_slices) == 1:
        return batch_slices[0]

    # Handle sequences and masks
    seqs = [x["sequence"] for x in batch_slices]
    masks = [x["mask"] for x in batch_slices]

    lengths = [x.shape[0] for x in seqs]
    cum_lengths = np.concatenate([[0], np.cumsum(lengths)])

    new_seq_d0 = np.sum(lengths)
    new_seq_d1 = np.max([x.shape[1] for x in seqs])
    new_seq_d2 = seqs[0].shape[2]
    new_seq = np.zeros((new_seq_d0, new_seq_d1, new_seq_d2), dtype=seqs[0].dtype)
    new_mask = np.zeros((new_seq_d0, new_seq_d1), dtype=np.bool_)

    for i, (seq, mask) in enumerate(zip(seqs, masks)):
        i0, i1 = cum_lengths[i], cum_lengths[i + 1]
        new_seq[i0:i1, : seq.shape[1], :] = seq
        new_mask[i0:i1, : seq.shape[1]] = mask

    # Handle class labels
    new_class_ = np.concatenate([x["class_"] for x in batch_slices], axis=0)

    # Handle auxiliary data
    auxs = [x["aux"] for x in batch_slices]
    keys = list(auxs[0].keys())
    new_aux = {k: np.concatenate([x[k] for x in auxs], axis=0) for k in keys}

    # Handle views
    views = [x["views"] for x in batch_slices]
    new_views = {}
    
    if views[0]:  # If views exist in first batch
        view_names = list(views[0].keys())
        for view_name in view_names:
            view_features = [x[view_name]["features"] for x in views]
            view_masks = [x[view_name]["mask"] for x in views]

            # Calculate dimensions for features
            max_seq_len = np.max([x.shape[1] for x in view_features])
            feature_dim = view_features[0].shape[2]
            batch_lengths = [x.shape[0] for x in view_features]
            cum_batch_lengths = np.concatenate([[0], np.cumsum(batch_lengths)])
            total_samples = cum_batch_lengths[-1]

            # Initialize feature and mask arrays
            new_features = np.zeros(
                (total_samples, max_seq_len, feature_dim), dtype=view_features[0].dtype
            )
            new_view_mask = np.zeros((total_samples, max_seq_len), dtype=np.bool_)

            # Fill feature and mask arrays
            for i, (features, mask) in enumerate(zip(view_features, view_masks)):
                i0, i1 = cum_batch_lengths[i], cum_batch_lengths[i + 1]
                seq_len = features.shape[1]
                new_features[i0:i1, :seq_len] = features
                new_view_mask[i0:i1, :seq_len] = mask

            new_view_data = {"features": new_features, "mask": new_view_mask}

            # Handle jets if present
            if "jets" in views[0][view_name]:
                view_jets = [x[view_name]["jets"] for x in views]
                jet_dim = view_jets[0].shape[-1]
                
                # Initialize jet array
                new_jets = np.zeros((total_samples, jet_dim), dtype=view_jets[0].dtype)
                
                # Fill jet array
                for i, jets in enumerate(view_jets):
                    i0, i1 = cum_batch_lengths[i], cum_batch_lengths[i + 1]
                    new_jets[i0:i1] = jets
                
                new_view_data["jets"] = new_jets

            new_views[view_name] = new_view_data

    return Batch(
        sequence=new_seq, 
        mask=new_mask, 
        class_=new_class_, 
        aux=new_aux, 
        views=new_views
    )


class JetDataset(Dataset):
    def __init__(self, config: _DataloaderConfig, split="train", **kwargs):
        """Create a dataset for the jet data. Keys of the configuration can be overwritten by kwargs.

        The dataset directs loads the data from the .root files **in batches**, sequentially in the order of files. No shuffling is done here. The user **should** shuffle the order in which the dataset is accessed (e.g. by using a `DataLoader` with `shuffle=True`).

        Everytime __getitem__ is called, the dataset will open the .root file(s) and load & process the data, and no presistent file handler is kept.


        Parameters
        ----------
        config : DataloaderConfig
            The configuration of the dataloader
        split : str, optional
            Which split to use, by default 'train'. The split should have a key in the `paths` dictionary of the configuration.
        **kwargs
            Overwrite the configuration with these values
        """

        config = copy.deepcopy(config)
        config.update(kwargs)
        self.config = config
        _path = config.paths[split]
        if isinstance(_path, str):
            paths = []
            for p in config.patterns:
                paths.extend(list(Path(_path).glob(p)))
            self.paths = sorted(paths)
        elif isinstance(_path, list):
            self.paths = [Path(x) for x in _path]
        self.sizes = self.cache_dataset_sizes(config, self.paths)
        self.batch_split_idx = self.init_batch_split(config, self.paths, self.sizes)
        self._len = len(self.batch_split_idx)
        self.require_names = parse_requested_keys(config)

    def __len__(self):
        return self._len

    @staticmethod
    def init_batch_split(
        config: _DataloaderConfig, paths: list[Path], sizes: NDArray[np.int64]
    ):
        if config.drop_last == 2:
            sizes = sizes // config.batch_size_atomic * config.batch_size_atomic
        cum_sizes = np.cumsum(np.concatenate([[0], sizes]))
        if config.drop_last == 0:
            _len = floor(cum_sizes[-1] / config.batch_size_atomic)
        else:
            _len = ceil(cum_sizes[-1] / config.batch_size_atomic)

        batch_split_idx: list[tuple[tuple[Path, int, int], ...]] = []
        for i in range(_len):
            g_evt0 = i * config.batch_size_atomic
            g_evt1 = g_evt0 + config.batch_size_atomic

            f_idx0 = np.searchsorted(cum_sizes, g_evt0, side="right") - 1
            f_idx1 = np.searchsorted(cum_sizes, g_evt1, side="left") - 1
            if f_idx0 < 0:
                f_idx0 = 0

            _batch_split: list[tuple[Path, int, int]] = []
            for f_idx in range(f_idx0, f_idx1 + 1):
                f_start0 = cum_sizes[f_idx]
                f_start = g_evt0 - f_start0
                f_end = min(g_evt1 - f_start0, sizes[f_idx])
                _batch_split.append((paths[f_idx], f_start, f_end))
                g_evt0 = cum_sizes[f_idx + 1]

            batch_split_idx.append(tuple(_batch_split))
        return batch_split_idx

    @staticmethod
    def cache_dataset_sizes(config: _DataloaderConfig, paths) -> NDArray[np.int64]:
        sizes = np.empty(len(paths), dtype=np.int64)
        for i, p in enumerate(paths):
            with up.open(p) as f:  # type: ignore
                sizes[i] = f[config.length_from]._members["fEntries"]  # type: ignore
        return sizes

    def __getitem__(self, idx):
        batch_split_idx = self.batch_split_idx[idx]
        batch_slices: list[Batch] = []
        for path, f_start, f_end in batch_split_idx:
            data = load_raw_data(path, self.require_names, f_start, f_end)
            data = run_processors(self.config, data)

            seq, mask, seq_len = sequence_and_mask(self.config, data)
            class_, aux = class_and_aux(self.config, data)
            views = process_views(self.config, data, seq_len)
            _batch = Batch(sequence=seq, mask=mask, class_=class_, aux=aux, views=views)
            batch_slices.append(_batch)
        return maybe_concat_batch_slices(batch_slices)

    def __repr__(self):
        return f"JetDataset(config={self.config}, paths={self.paths})"

    def __str__(self):
        return f"JetDataset(config={self.config})"


JetDataLoader = partial(DataLoader, collate_fn=maybe_concat_batch_slices)
"A simple wrapper of `torch.utils.data.DataLoader` with `collate_fn` set to `lambda x: x` and `batch_size` set to `None`."
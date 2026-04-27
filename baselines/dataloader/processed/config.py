import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

pad_length_t = Enum("pad_length_t", "max fixed all")


@dataclass
class TransformationSchema:
    processor: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()


@dataclass
class ViewSchema:
    mask: str
    features: list[str]
    jets: Optional[list[str]] = None
    seq_len: Optional[int] = None


@dataclass
class OutputSchema:
    sequence: list[str]
    sequence_mask: str
    class_: list[str]
    aux: list[str]
    views: Optional[dict[str, ViewSchema]] = None


@dataclass
class _DataloaderConfig:
    paths: dict[str, Any]
    dtype: str
    patterns: list[str]
    batch_size: int
    batch_size_atomic: int = -1  # -1 means disabled (use batch_size directly)
    drop_last: int = 0
    max_seq_length: int = 128
    seq_pad_strategy: pad_length_t = pad_length_t.fixed
    transformations: list[TransformationSchema] = field(default_factory=list)
    outputs: OutputSchema = field(default_factory=lambda: OutputSchema(sequence=[], sequence_mask="", class_=[], aux=[]))

    def update(self, updates: dict): ...

    def __post_init__(self):
        if self.batch_size_atomic > self.batch_size:
            warnings.warn(
                f"batch_size_atomic ({self.batch_size_atomic}) > batch_size ({self.batch_size}); "
                "taking short path (using batch_size directly)."
            )
        elif 0 < self.batch_size_atomic < self.batch_size:
            assert (
                self.batch_size % self.batch_size_atomic == 0
            ), "batch_size must be divisible by batch_size_atomic"


class DataloaderConfig:
    def __new__(cls, path: str | Path):
        _dataloader_config = OmegaConf.load(path)
        dataloader_schema = OmegaConf.structured(_DataloaderConfig)
        dataloader_schema.merge_with(_dataloader_config.dataloader)
        dataloader_config: _DataloaderConfig = dataloader_schema
        return dataloader_config

    def __init__(self, path: str | Path): ...
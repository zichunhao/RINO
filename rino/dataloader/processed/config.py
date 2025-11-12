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
    drop_last: int
    max_seq_length: int
    seq_pad_strategy: pad_length_t
    transformations: list[TransformationSchema]
    outputs: OutputSchema

    def update(self, updates: dict): ...

    def __post_init__(self): ...


class DataloaderConfig:
    def __new__(cls, path: str | Path):
        _dataloader_config = OmegaConf.load(path)
        dataloader_schema = OmegaConf.structured(_DataloaderConfig)
        dataloader_schema.merge_with(_dataloader_config.dataloader)
        dataloader_config: _DataloaderConfig = dataloader_schema
        return dataloader_config

    def __init__(self, path: str | Path): ...

"""Shared helpers for the representation studies (domain shift, scale structure).

The studies reuse the release code in ``dino/`` for configs, dataloaders and
models. Importing this module puts ``dino/`` on ``sys.path`` so that the
release modules (``utils``, ``models``, ``dataloader``) resolve the same way
they do for ``dino/dino_inference.py``.

Three kinds of pretrained encoders are supported (``--model-type``):

* ``dino``     -- a DINO/iBOT pretraining checkpoint (RINO and its ablations).
                  The EMA *teacher* backbone and teacher DINO head are used.
* ``jetclr``   -- a single-network contrastive checkpoint (JetCLR, JetCLR-scale)
                  saved by ``dino/jetclr_train.py``; backbone + projection head.
* ``backbone`` -- a bare ``{"backbone": state_dict}`` file, e.g. a baseline
                  converted by ``baselines/scripts/convert_checkpoint.py``.
"""

from __future__ import annotations

import contextlib
import copy
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# The cuDNN SDPA backend fails with "cuDNN Frontend error: No execution plans support
# the graph" on several GPU/driver combinations; the release training and inference
# entry points disable it the same way (see dino/dino_train.py).
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINO_DIR = PROJECT_ROOT / "dino"
if str(DINO_DIR) not in sys.path:
    sys.path.insert(0, str(DINO_DIR))

from models import JetTransformerEncoder  # noqa: E402
from utils.ckpt import get_checkpoints_path, process_placeholder  # noqa: E402
from utils.device import check_bf16_support, get_available_device  # noqa: E402
from utils.logger import LOGGER, configure_logger  # noqa: E402
from utils.producers import (  # noqa: E402
    get_config,
    get_dataloader_and_config,
    get_models,
    get_models_single,
)
from utils.producers.model import load_weight  # noqa: E402

MODEL_TYPES = ("dino", "jetclr", "backbone")
POOLINGS = ("native", "mean")

__all__ = [
    "PROJECT_ROOT",
    "LOGGER",
    "MODEL_TYPES",
    "POOLINGS",
    "configure_logger",
    "load_config",
    "resolve_checkpoint",
    "get_part_dim",
    "load_encoder",
    "load_backbone_weights",
    "build_dataloader",
    "resolve_device",
    "autocast",
    "batch_inputs",
    "encode",
    "normalize",
]


# --------------------------------------------------------------------------- #
# Configs and checkpoints
# --------------------------------------------------------------------------- #


def _resolve_placeholders(obj: Any, config: dict) -> Any:
    """Recursively replace PROJECT_ROOT/JOBNAME in every string of ``obj``.

    EPOCHNUM is left untouched; it is resolved when a checkpoint path is built.
    """
    if isinstance(obj, str):
        return process_placeholder(s=obj, config=config, epoch_num=None)
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, config) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(v, config) for v in obj]
    return obj


def load_config(path: str | Path) -> dict:
    """Load a YAML config and resolve its PROJECT_ROOT/JOBNAME placeholders."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} does not contain a YAML mapping")
    return _resolve_placeholders(config, config)


def resolve_checkpoint(
    config: dict, checkpoint: str | None, load_epoch: str | int | None
) -> Path:
    """Return the checkpoint to load.

    An explicit ``checkpoint`` path wins. Otherwise the path is built from the
    config's ``training.checkpoints_dir``/``checkpoints_filename`` for
    ``load_epoch`` (an integer or ``"best"``), as in ``dino/dino_inference.py``.
    """
    if checkpoint is not None:
        path = Path(process_placeholder(s=str(checkpoint), config=config, epoch_num=None))
    elif load_epoch is not None:
        epoch = int(load_epoch) if str(load_epoch).isdigit() else load_epoch
        path = get_checkpoints_path(config=config, epoch_num=epoch)
    else:
        raise ValueError("Specify either a checkpoint path or a load epoch")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def get_part_dim(data_config: dict, mode: str, split: str) -> int:
    """Number of particle features produced by the dataloader of ``split``."""
    split_cfg = data_config[mode]["dataloader"][split]
    dataloader_config = get_config(
        config=data_config,
        mode=mode,
        split=split,
        preprocessed=split_cfg.get("preprocessed", False),
    )
    return len(dataloader_config.outputs.sequence)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def load_backbone_weights(
    backbone: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    allow_missing_keys: bool = False,
) -> None:
    """Load ``state_dict`` into ``backbone``.

    By default the load is strict (``load_weight`` from the release, which also
    strips ``module.``/``_orig_mod.`` prefixes). With ``allow_missing_keys``,
    parameters absent from ``state_dict`` keep their initialisation, which is
    how the finetuning loader treats a converted OmniJet-alpha backbone (no
    continuous input layer). Unexpected keys are always an error, and so is a
    state dict that matches no parameter at all, so a wrongly keyed checkpoint
    can never silently leave the backbone at random initialisation.
    """
    if not allow_missing_keys:
        load_weight(backbone, state_dict)
        return

    for prefix in ("_orig_mod.module.", "module.", "_orig_mod."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            break
    result = backbone.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys in backbone state dict: {result.unexpected_keys[:10]}"
        )
    if not state_dict:
        raise RuntimeError("Backbone state dict is empty; nothing was loaded")
    if result.missing_keys:
        LOGGER.warning(
            f"{len(result.missing_keys)} backbone tensors are not in the checkpoint "
            f"and keep their initialisation: {result.missing_keys}"
        )
    LOGGER.info(f"Loaded {len(state_dict)} backbone tensors")


def _inference_config(model_config: dict, checkpoint: Path) -> dict:
    """Copy of ``model_config`` that loads ``checkpoint`` in inference mode."""
    config = copy.deepcopy(model_config)
    config["compile_model"] = False
    inference = config.setdefault("inference", {}) or {}
    config["inference"] = inference
    inference["load_path"] = str(checkpoint)
    inference.pop("load_epoch", None)
    inference["rep"] = {"penultimate_layer": False}
    return config


def load_encoder(
    model_type: str,
    model_config: dict,
    checkpoint: Path,
    part_dim: int,
    device: torch.device,
    allow_missing_keys: bool = False,
) -> tuple[torch.nn.Module, torch.nn.Module | None]:
    """Build a pretrained encoder and load its weights.

    Args:
        model_type: One of ``MODEL_TYPES`` (see the module docstring).
        model_config: Config with a ``models.backbone`` section. For ``dino`` it
            is the pretraining config (with ``models.dino_head``); for
            ``jetclr`` the contrastive pretraining config (with
            ``models.projection_head``); for ``backbone`` only
            ``models.backbone`` is read.
        checkpoint: Checkpoint file.
        part_dim: Number of particle input features.
        device: Device to place the modules on.
        allow_missing_keys: ``backbone`` type only; see ``load_backbone_weights``.

    Returns:
        ``(backbone, head)`` in eval mode. ``head`` is the teacher DINO head
        (``dino``), the projection head (``jetclr``) or ``None`` (``backbone``).
    """
    if model_type == "dino":
        config = _inference_config(model_config, checkpoint)
        _student, (backbone, head, _ibot_head), _pos_emb, _scale_emb = get_models(
            part_dim=part_dim, config=config, mode="inference", device=device
        )
    elif model_type == "jetclr":
        config = _inference_config(model_config, checkpoint)
        backbone, head, _ibot_head, _pos_emb = get_models_single(
            part_dim=part_dim, config=config, mode="inference", device=device
        )
    elif model_type == "backbone":
        backbone_cfg = model_config["models"]["backbone"]
        backbone_type = backbone_cfg.get("type", "JetTransformerEncoder")
        if backbone_type != "JetTransformerEncoder":
            raise NotImplementedError(f"Unsupported backbone type: {backbone_type}")
        backbone = JetTransformerEncoder(
            part_dim=part_dim, **copy.deepcopy(backbone_cfg.get("params", {}))
        )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "backbone" not in ckpt:
            raise KeyError(
                f"{checkpoint} has no 'backbone' entry; expected the output of "
                "baselines/scripts/convert_checkpoint.py"
            )
        load_backbone_weights(backbone, ckpt["backbone"], allow_missing_keys)
        backbone = backbone.to(device)
        head = None
    else:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")

    backbone.eval()
    if head is not None:
        head.eval()
    return backbone, head


# --------------------------------------------------------------------------- #
# Data and forward passes
# --------------------------------------------------------------------------- #


def build_dataloader(
    data_config: dict, mode: str, split: str, batch_size: int | None = None
):
    """Deterministic (unshuffled) dataloader for ``data_config[mode].dataloader[split]``."""
    config = copy.deepcopy(data_config)
    if split not in config.get(mode, {}).get("dataloader", {}):
        raise KeyError(f"Split '{split}' not found under {mode}.dataloader")
    config.setdefault("shuffle", {})
    config["shuffle"] = dict(config["shuffle"] or {}, **{mode: False})
    if batch_size is not None:
        split_cfg = config[mode]["dataloader"][split]
        kwargs = split_cfg.setdefault("kwargs", {}) or {}
        split_cfg["kwargs"] = kwargs
        kwargs["batch_size"] = batch_size
        if "batch_size_atomic" in kwargs:
            kwargs["batch_size_atomic"] = min(kwargs["batch_size_atomic"], batch_size)
    dataloader, _ = get_dataloader_and_config(config=config, mode=mode, split=split)
    return dataloader


def resolve_device(name: str) -> torch.device:
    """``"auto"`` picks the release default (CUDA, then MPS, then CPU)."""
    return get_available_device() if name == "auto" else torch.device(name)


@contextlib.contextmanager
def autocast(device: torch.device, use_bf16: bool) -> Iterator[None]:
    """bfloat16 autocast on CUDA when requested and supported; no-op otherwise."""
    enabled = use_bf16 and device.type == "cuda" and check_bf16_support(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        yield


def batch_inputs(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(particles, jets, mask)`` tensors of a release dataloader batch."""
    particles = torch.as_tensor(np.asarray(batch["sequence"]), dtype=torch.float32)
    jets = torch.as_tensor(np.asarray(batch["class_"]), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(batch["mask"]), dtype=torch.bool)
    return particles.to(device), jets.to(device), mask.to(device)


def encode(
    backbone: torch.nn.Module,
    particles: torch.Tensor,
    mask: torch.Tensor,
    jets: torch.Tensor | None,
    pooling: str = "native",
) -> torch.Tensor:
    """Pooled jet representation of one batch.

    Args:
        pooling: ``"native"`` returns the backbone's own pooled output (the CLS
            token for ``cls_token`` backbones such as RINO and JetCLR-scale; the
            masked mean over all tokens for ``mean`` backbones such as the MPM
            baselines). ``"mean"`` returns the masked mean of the particle
            output tokens only (CLS and register tokens excluded), as
            ``dino/diagnostics/scale_invariance_analysis.py --pooling mean``.
    """
    rep, particles_out = backbone(particles, mask=mask, jets=jets)
    if pooling == "native":
        return rep
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(particles_out.dtype)
        return (particles_out * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")


def normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise rows (float32)."""
    return F.normalize(x.float(), dim=-1)

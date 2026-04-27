"""Self-contained model factory for baselines finetuning.

Loads pretrained backbone from a generic checkpoint (key='backbone'),
NOT from DINO's teacher/student checkpoint structure.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

# Shared model classes (not DINO-specific)
from models import AssembledModel
from models.jet_transformer_encoder import JetTransformerEncoder
from models.head import MLPHead

LOGGER = logging.getLogger(__name__)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_models_finetune(
    part_dim: int,
    config: dict,
    mode: str = "training",
    device: torch.device | None = None,
    train_head: bool = True,
) -> AssembledModel:
    """Create backbone + classification head and load pretrained weights.

    For baselines (JetCLR, etc.), the pretrained checkpoint stores
    backbone weights under the 'backbone' key — no teacher/student split.
    """
    models_config = config["models"]

    # --- Backbone ---
    backbone_config = models_config["backbone"]
    backbone_params = backbone_config.get("params", {})
    backbone = JetTransformerEncoder(part_dim=part_dim, **backbone_params)
    if device:
        backbone = backbone.to(device)
    LOGGER.info(f"Backbone parameters: {_count_parameters(backbone):,}")

    # --- Head ---
    head_config = models_config["head"]
    head_params = head_config.get("params", {})
    head_input_dim = (
        backbone.rep_dim if hasattr(backbone, "rep_dim") else backbone.d_model
    )
    head = MLPHead(input_dim=head_input_dim, **head_params)
    if device:
        head = head.to(device)
    LOGGER.info(f"Head parameters: {_count_parameters(head):,}")

    # --- Assemble ---
    model = AssembledModel(embedding=None, backbone=backbone, heads=head)

    # --- Load weights ---
    if train_head:
        _load_backbone_weights(backbone, head, config, mode)
    else:
        _load_assembled_model(model, config, mode)

    LOGGER.info(f"Total parameters: {_count_parameters(model):,}")
    return model


def _resolve_path(path_str: str, config: dict) -> Path:
    """Resolve PROJECT_ROOT and JOBNAME placeholders."""
    from utils.ckpt import process_placeholder

    return Path(process_placeholder(s=path_str, config=config, epoch_num=None))


def _load_backbone_weights(
    backbone: nn.Module,
    head: nn.Module,
    config: dict,
    mode: str,
) -> None:
    """Load pretrained backbone from a baselines checkpoint.

    Tries keys in order: 'backbone', then raw state_dict.
    Never looks for 'teacher' (that's DINO-specific).
    """
    section = config.get("training" if mode.startswith("train") else "inference", {})

    backbone_weight_path = section.get("backbone_weight_path")
    load_epoch = section.get("load_epoch")

    if backbone_weight_path:
        ckpt_path = _resolve_path(backbone_weight_path, config)
        LOGGER.info(f"Loading backbone weights from: {ckpt_path}")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Try 'backbone' key first (JetCLR/baselines format)
        if "backbone" in state_dict:
            missing, unexpected = backbone.load_state_dict(
                state_dict["backbone"], strict=False
            )
            LOGGER.info(
                f"Loaded backbone weights from 'backbone' key "
                f"(missing={missing}, unexpected={unexpected})"
            )
        else:
            # Try loading the full dict directly (plain model save)
            try:
                missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                LOGGER.info(
                    f"Loaded backbone weights from raw state dict "
                    f"(missing={missing}, unexpected={unexpected})"
                )
            except RuntimeError:
                raise ValueError(
                    f"Checkpoint at {ckpt_path} has keys: {list(state_dict.keys())}. "
                    f"Expected 'backbone' key or a raw backbone state dict."
                )
        return

    if load_epoch is not None:
        # Resume from a finetuning checkpoint (has both backbone and head)
        from utils.ckpt import get_checkpoints_path

        ckpt_path = get_checkpoints_path(config, load_epoch)
        LOGGER.info(f"Resuming from finetune checkpoint: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "backbone" in state_dict:
            backbone.load_state_dict(state_dict["backbone"])
        if "head" in state_dict and state_dict["head"] is not None:
            head.load_state_dict(state_dict["head"])
        LOGGER.info(f"Resumed backbone + head from epoch {load_epoch}")
        return

    LOGGER.info("No pretrained weights specified — training from scratch")


def _load_assembled_model(
    model: AssembledModel,
    config: dict,
    mode: str,
) -> None:
    """Load a full assembled model (for inference of a finetuned model)."""
    section = config.get("training" if mode.startswith("train") else "inference", {})
    load_epoch = section.get("load_epoch")

    if load_epoch is None:
        raise ValueError("load_epoch must be set for inference mode")

    from utils.ckpt import get_checkpoints_path

    ckpt_path = get_checkpoints_path(config, load_epoch)
    LOGGER.info(f"Loading assembled model from: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # dino/classification_train.py saves a flat dict under 'model' key with
    # 'backbone.*' and 'heads.*' prefixed keys — not separate sub-dicts.
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
        LOGGER.info(f"Loaded assembled model from 'model' key (epoch {load_epoch})")
    elif "backbone" in state_dict:
        model.backbone.load_state_dict(state_dict["backbone"])
        if "head" in state_dict and state_dict["head"] is not None:
            model.heads.load_state_dict(state_dict["head"])
        LOGGER.info(f"Loaded backbone+head from separate keys (epoch {load_epoch})")
    else:
        raise ValueError(
            f"Checkpoint at {ckpt_path} has keys {list(state_dict.keys())}. "
            f"Expected 'model' or 'backbone'/'head' keys."
        )

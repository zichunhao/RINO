import torch.nn as nn
from torch import optim
import torch
from typing import Any
from ..ckpt import get_checkpoints_path
from ..logger import LOGGER

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


def get_param_groups(
    backbone: nn.Module,
    head: nn.Module,
    config: dict[str, Any],
    optimizer_params: dict[str, Any] | None = None,
) -> list[dict[str, list[nn.Parameter] | float | str]]:
    """
    Create parameter groups with different learning rates for backbone and head.

    Args:
        backbone: The backbone model (pretrained)
        head: The classification head
        config: The training config
        optimizer_params: Optional parameters to override config values

    Returns:
        List of parameter groups with different learning rates
    """
    optimizer_config = config["training"].get("optimizer", {})
    base_params = optimizer_config.get("params", {})

    # Merge with additional params if provided
    if optimizer_params is not None:
        base_params.update(optimizer_params)

    base_lr = base_params.get("lr", 1e-4)
    backbone_lr_factor = optimizer_config.get("backbone_lr_factor", 0.1)

    param_groups = [
        {
            "params": list(backbone.parameters()),
            "lr": base_lr * backbone_lr_factor,
            "name": "backbone",
        },
        {"params": list(head.parameters()), "lr": base_lr, "name": "head"},
    ]

    return param_groups


def get_optimizer_finetune(
    config: dict[str, Any],
    model_params: list[nn.Parameter] | list[dict[str, Any]],
    optimizer_params: dict[str, Any] | None = None,
) -> optim.Optimizer:
    """
    Get the optimizer from the training config with enhanced learning rate features.

    Args:
        config: The training config
        model_params: Either a list of parameters or parameter groups with learning rates
        optimizer_params: Optional additional optimizer parameters to override config

    Returns:
        Configured optimizer
    """
    optimizer_config = config["training"].get("optimizer", {})
    LOGGER.info(f"Optimizer config: {optimizer_config}")

    optimizer_type = optimizer_config.get("name", "adam").lower()
    base_optimizer_params = optimizer_config.get("params", {})

    # Merge with additional params if provided
    if optimizer_params is not None:
        base_optimizer_params.update(optimizer_params)

    # Set sensible defaults for different optimizers
    if optimizer_type == "adam":
        base_optimizer_params.setdefault("lr", 1e-4)
        base_optimizer_params.setdefault("weight_decay", 1e-4)
    elif optimizer_type == "adamw":
        base_optimizer_params.setdefault("lr", 1e-4)
        base_optimizer_params.setdefault("weight_decay", 0.01)
    elif optimizer_type == "sgd":
        base_optimizer_params.setdefault("lr", 1e-3)
        base_optimizer_params.setdefault("momentum", 0.9)
        base_optimizer_params.setdefault("weight_decay", 1e-4)

    optimizer_class = OPTIMIZER_MAP.get(optimizer_type)
    if optimizer_class is None:
        LOGGER.warning(f"Unsupported optimizer type: {optimizer_type}. Using Adam.")
        optimizer_class = optim.Adam

    optimizer = optimizer_class(model_params, **base_optimizer_params)
    LOGGER.info(f"Created optimizer: {optimizer}")

    # Load optimizer state if specified in config
    epoch = config["training"].get("load_epoch")
    if epoch is not None:
        ckpt_path = get_checkpoints_path(config=config, epoch_num=epoch)
        checkpoint_dict = torch.load(ckpt_path)
        if "optimizer" in checkpoint_dict:
            state_dict = checkpoint_dict["optimizer"]
        elif "optimizer_state_dict" in checkpoint_dict:
            state_dict = checkpoint_dict["optimizer_state_dict"]
        else:
            raise ValueError("Checkpoint file does not contain the optimizer state")
        optimizer.load_state_dict(state_dict)
        LOGGER.info(f"Loaded optimizer state from {ckpt_path}")

    return optimizer

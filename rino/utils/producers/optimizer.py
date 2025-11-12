import torch.nn as nn
from torch import optim
import torch
from typing import Any, Dict, List, Optional
import logging
from ..ckpt import get_checkpoints_path
from ..logger import LOGGER

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


def get_optimizer(
    config: Dict[str, Any],
    model_params: List[nn.Parameter],
) -> optim.Optimizer:
    """
    Get the optimizer from the training config. Supported optimizers: Adam, SGD, AdamW.

    Args:
        config: The training config.
        model_params: The model parameters.
        skip_load_if_error: If True, skip loading optimizer state on error.

    Returns:
        The configured optimizer.
    """
    optimizer_config = config["training"].get("optimizer", {})
    LOGGER.info(f"Optimizer config: {optimizer_config}")

    optimizer_type = optimizer_config.get("name", "adam").lower()
    optimizer_params = optimizer_config.get("params", {})

    optimizer_class = OPTIMIZER_MAP.get(optimizer_type)
    if optimizer_class is None:
        LOGGER.warning(f"Unsupported optimizer type: {optimizer_type}. Using Adam.")
        optimizer_class = optim.Adam

    optimizer = optimizer_class(model_params, **optimizer_params)

    # Load optimizer state if specified in config
    epoch = config["training"].get("load_epoch")
    if epoch:
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

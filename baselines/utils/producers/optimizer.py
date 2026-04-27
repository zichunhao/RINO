import torch.nn as nn
from torch import optim
import torch
from typing import Any, Dict, List, Optional, Tuple
import math
from ..ckpt import get_checkpoints_path
from ..logger import LOGGER

OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamw": optim.AdamW,
}


class WeightDecayScheduler:
    """Scheduler for weight decay parameter."""

    def __init__(self, optimizer: optim.Optimizer, schedule_config: Dict[str, Any]):
        """
        Args:
            optimizer: The optimizer to schedule weight_decay for.
            schedule_config: Configuration dict with keys:
                - type: 'linear', 'cosine', 'step', 'constant'
                - start_decay: initial weight_decay value
                - end_decay: final weight_decay value
                - total_epochs: total training epochs (for linear/cosine)
                - step_size: epochs per decay (for step scheduler)
                - gamma: decay factor (for step scheduler)
        """
        self.optimizer = optimizer
        self.config = schedule_config
        self.schedule_type = schedule_config.get("type", "constant")
        self.start_decay = schedule_config.get("start_decay", 0.01)
        self.end_decay = schedule_config.get("end_decay", 0.01)
        self.total_epochs = schedule_config.get("total_epochs", 100)
        self.step_size = schedule_config.get("step_size", 10)
        self.gamma = schedule_config.get("gamma", 0.1)
        self.current_epoch = 0

    def step(self):
        """Update weight_decay based on current epoch."""
        self.current_epoch += 1

        if self.schedule_type == "linear":
            decay = self._linear_schedule()
        elif self.schedule_type == "cosine":
            decay = self._cosine_schedule()
        elif self.schedule_type == "step":
            decay = self._step_schedule()
        else:  # constant
            decay = self.start_decay

        # Update weight_decay in optimizer
        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = decay

        return decay

    def _linear_schedule(self):
        """Linear decay from start_decay to end_decay."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        return self.start_decay + (self.end_decay - self.start_decay) * progress

    def _cosine_schedule(self):
        """Cosine annealing from start_decay to end_decay."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.end_decay + (self.start_decay - self.end_decay) * cosine_decay

    def _step_schedule(self):
        """Step decay: multiply by gamma every step_size epochs."""
        num_decays = self.current_epoch // self.step_size
        return self.start_decay * (self.gamma**num_decays)

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {"current_epoch": self.current_epoch, "config": self.config}

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.current_epoch = state_dict["current_epoch"]


def get_optimizer(
    config: Dict[str, Any],
    model_params: List[nn.Parameter],
) -> Tuple[optim.Optimizer, Optional[WeightDecayScheduler]]:
    """
    Get the optimizer and optional weight decay scheduler from the training config.

    Args:
        config: The training config.
        model_params: The model parameters.

    Returns:
        Tuple of (optimizer, weight_decay_scheduler or None).
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

    # Create weight decay scheduler if specified
    wd_scheduler = None
    if "weight_decay_schedule" in optimizer_config:
        wd_schedule_config = optimizer_config["weight_decay_schedule"]
        wd_scheduler = WeightDecayScheduler(optimizer, wd_schedule_config)
        LOGGER.info(f"Created weight decay scheduler: {wd_schedule_config}")

    return optimizer, wd_scheduler

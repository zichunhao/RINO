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


def _backbone_param_groups(
    backbone: nn.Module,
    base_lr: float,
    lr_decay: float,
) -> List[Dict[str, Any]]:
    """Build per-layer parameter groups for layer-wise LR decay (LLRD).

    Depth assignment for JetTransformerEncoder:
      - Stem (embeddings, CLS token, pos encoding, etc.): depth 0
      - ``transformer_encoder.layers.i``:  depth ``i + 1``
      - ``final_norm``, ``pooling_network``: depth ``num_layers`` (same as last layer)

    LR at depth *d*: ``base_lr * lr_decay^(num_layers - d + 1)``
      - stem (d=0):             ``base_lr * lr_decay^(num_layers + 1)``  [most decayed]
      - last encoder layer:     ``base_lr * lr_decay^1``

    Heads are kept in a separate group at ``base_lr`` by the caller.
    """
    # JetTransformerEncoder uses `transformer_encoder.layers`
    _encoder_attr = None
    for attr in ("transformer_encoder", "encoder"):
        candidate = getattr(backbone, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            _encoder_attr = attr
            break

    num_layers = len(getattr(backbone, _encoder_attr).layers) if _encoder_attr else 0

    if num_layers == 0:
        params = [p for p in backbone.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr * lr_decay}] if params else []

    _encoder_prefix = f"{_encoder_attr}.layers."
    _post_encoder_prefixes = ("final_norm", "pooling_network")
    groups: Dict[int, List] = {}

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(_encoder_prefix):
            depth = int(name.split(".")[2]) + 1
        elif any(name.startswith(p) for p in _post_encoder_prefixes):
            depth = num_layers
        else:
            depth = 0
        groups.setdefault(depth, []).append(param)

    param_groups = []
    for depth, params in sorted(groups.items()):
        lr = base_lr * (lr_decay ** (num_layers - depth + 1))
        param_groups.append({"params": params, "lr": lr})

    LOGGER.info(
        f"LLRD: {num_layers} encoder layers, decay={lr_decay:.4f}. "
        f"LR range: [{base_lr * lr_decay**(num_layers+1):.2e}, {base_lr * lr_decay:.2e}]"
    )
    return param_groups


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
    backbone: nn.Module,
    head_modules: List[nn.Module],
) -> Tuple[optim.Optimizer, Optional[WeightDecayScheduler]]:
    """
    Get the optimizer and optional weight decay scheduler from the training config.

    When ``optimizer.backbone_lr_decay`` is in (0, 1), applies layer-wise LR
    decay (LLRD) to the backbone: each encoder layer receives a lower LR than
    the one above it.  Head modules always use the base LR.

    Args:
        config: The training config.
        backbone: The backbone model (subject to optional LLRD).
        head_modules: Projection heads and auxiliary modules (always base LR).

    Returns:
        Tuple of (optimizer, weight_decay_scheduler or None).
    """
    optimizer_config = config["training"].get("optimizer", {})
    LOGGER.info(f"Optimizer config: {optimizer_config}")

    optimizer_type = optimizer_config.get("name", "adam").lower()
    optimizer_params = optimizer_config.get("params", {})
    base_lr = optimizer_params.get("lr", 1e-4)

    optimizer_class = OPTIMIZER_MAP.get(optimizer_type)
    if optimizer_class is None:
        LOGGER.warning(f"Unsupported optimizer type: {optimizer_type}. Using Adam.")
        optimizer_class = optim.Adam

    # Build backbone param groups (with optional LLRD)
    backbone_lr_decay = optimizer_config.get("backbone_lr_decay", 1.0)
    if 0 < backbone_lr_decay < 1:
        param_groups = _backbone_param_groups(backbone, base_lr, backbone_lr_decay)
    else:
        param_groups = [
            {"params": [p for p in backbone.parameters() if p.requires_grad]}
        ]

    # Head modules always use the base LR (taken from optimizer defaults)
    head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params})

    optimizer = optimizer_class(param_groups, **optimizer_params)

    # Create weight decay scheduler if specified
    wd_scheduler = None
    if "weight_decay_schedule" in optimizer_config:
        wd_schedule_config = optimizer_config["weight_decay_schedule"]
        wd_scheduler = WeightDecayScheduler(optimizer, wd_schedule_config)
        LOGGER.info(f"Created weight decay scheduler: {wd_schedule_config}")

    return optimizer, wd_scheduler

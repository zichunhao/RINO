"""Miscellaneous training utilities shared across DINO training scripts."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

from utils.ckpt import load_checkpoint
from utils.logger import LOGGER

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def should_save_and_log(accelerator: Accelerator | None) -> bool:
    return accelerator is None or accelerator.is_main_process


def _get_loss_module(loss: nn.Module) -> nn.Module:
    """Unwrap DDP-wrapped loss modules to access .step_epoch() etc."""
    return loss.module if hasattr(loss, "module") else loss


# ---------------------------------------------------------------------------
# Loss aggregation
# ---------------------------------------------------------------------------


def aggregate_losses(
    all_batch_losses: list[dict[str, float]],
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    """Average loss metrics across batches using sample-weighted means."""
    if not all_batch_losses:
        return {}

    all_keys = sorted(
        k
        for k in set().union(*[d.keys() for d in all_batch_losses])
        if k not in ["batch_size", "dom_unique", "gram_weight"] and "dom_top" not in k
    )

    aggregated = {}
    for key in all_keys:
        local_values = torch.tensor(
            [b.get(key, 0.0) for b in all_batch_losses], dtype=torch.float32
        )
        local_batch_sizes = torch.tensor(
            [b.get("batch_size", 1.0) for b in all_batch_losses], dtype=torch.float32
        )
        local_weighted_sum = (local_values * local_batch_sizes).sum()
        local_total_samples = local_batch_sizes.sum()

        if accelerator is not None and accelerator.num_processes > 1:
            local_weighted_sum = local_weighted_sum.to(accelerator.device)
            local_total_samples = local_total_samples.to(accelerator.device)
            global_weighted_sum = accelerator.reduce(
                local_weighted_sum, reduction="sum"
            )
            global_total_samples = accelerator.reduce(
                local_total_samples, reduction="sum"
            )
            weighted_mean = (global_weighted_sum / global_total_samples).item()
        else:
            weighted_mean = (local_weighted_sum / local_total_samples).item()

        aggregated[key] = weighted_mean

    return aggregated


# ---------------------------------------------------------------------------
# Momentum schedule
# ---------------------------------------------------------------------------


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# ---------------------------------------------------------------------------
# Loss warmup scaling
# ---------------------------------------------------------------------------


def _scale_loss_warmup_to_steps(loss_params: dict, steps_per_epoch: int) -> dict:
    """Return a deep copy of *loss_params* with epoch-based warmup keys converted to steps."""
    result = copy.deepcopy(loss_params)
    for key, val in result.items():
        if key.endswith("_warmup") and isinstance(val, dict):
            if "start_epoch" in val:
                val["start_step"] = val.pop("start_epoch") * steps_per_epoch
            if "end_epoch" in val:
                val["end_step"] = val.pop("end_epoch") * steps_per_epoch
            LOGGER.debug(
                f"{key}: start_step={val.get('start_step')}, end_step={val.get('end_step')}"
            )
    return result


def _rescale_warmup_steps(
    loss_module: nn.Module, old_steps_per_epoch: int, new_steps_per_epoch: int
) -> None:
    """Rescale WarmupSchedule step indices in-place after actual steps-per-epoch is known."""
    ratio = new_steps_per_epoch / old_steps_per_epoch
    for attr in ("_temp_schedule", "_weight_schedule"):
        schedule = getattr(loss_module, attr, None)
        if schedule is None:
            continue
        if schedule.mode == "step":
            if schedule.start_step is not None:
                schedule.start_step = round(schedule.start_step * ratio)
            if schedule.end_step is not None:
                schedule.end_step = round(schedule.end_step * ratio)
            LOGGER.debug(
                f"  Rescaled schedule end_step → {schedule.end_step} "
                f"(ratio={ratio:.4f})"
            )


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------


def get_training_start_params(
    config: dict[str, Any],
    device: torch.device,
    accelerator: Accelerator | None = None,
) -> tuple[int, float, int]:
    training_params = config["training"]
    load_epoch = training_params.get("load_epoch")

    if load_epoch is None:
        return 0, float("inf"), 0

    checkpoint = load_checkpoint(config=config, device=device, epoch=load_epoch)

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]["loss"]
    num_stale_epochs = checkpoint.get("num_stale_epochs", 0)

    if should_save_and_log(accelerator):
        LOGGER.info(f"Loaded checkpoint for epoch {checkpoint['epoch']}")
        LOGGER.info(
            f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.8f}, "
            f"num_stale_epochs {num_stale_epochs}"
        )

    return start_epoch, best_val_loss, num_stale_epochs


def restore_schedulers(
    checkpoint: dict[str, Any],
    wd_scheduler: Any | None,
    accelerator: Accelerator | None = None,
) -> None:
    """Restore weight decay scheduler state from a checkpoint in-place.

    NOTE: LR scheduler state is intentionally NOT restored here.
    It is handled inside get_scheduler() in producers/scheduler.py.
    """
    if wd_scheduler is not None and "weight_decay_scheduler" in checkpoint:
        wd_scheduler.load_state_dict(checkpoint["weight_decay_scheduler"])
        if should_save_and_log(accelerator):
            LOGGER.info("Loaded weight decay scheduler state from checkpoint")
    elif wd_scheduler is not None:
        if should_save_and_log(accelerator):
            LOGGER.warning(
                "Weight decay scheduler found but no 'weight_decay_scheduler' key "
                "in checkpoint — not restored"
            )

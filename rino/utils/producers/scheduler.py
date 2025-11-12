import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
    ConstantLR,
    LambdaLR,
)

from ..logger import LOGGER
from ..ckpt import get_checkpoints_path


def get_cosine_schedule(
    iter: int,
    total_iters: int,
    base_value: float = 1e-3,
    final_value: float = 0.0,
):
    return final_value + 0.5 * (base_value - final_value) * (
        1 + math.cos(math.pi * iter / total_iters)
    )


def get_scheduler(
    optimizer: Optimizer,
    config: dict,
) -> _LRScheduler | ReduceLROnPlateau | None:
    """
    Get the scheduler. If not specified, will give None.
    """

    scheduler_config = config["training"].get("scheduler")
    if not scheduler_config:
        LOGGER.info("No scheduler specified in config. Returning None.")
        return None

    scheduler_name = scheduler_config.get("name", "").lower()
    scheduler_params = scheduler_config.get("params", {})

    if scheduler_name == "steplr":
        scheduler = StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif scheduler_name == "linearlr":
        scheduler = LinearLR(optimizer, **scheduler_params)
    elif scheduler_name == "constantlr":
        scheduler = ConstantLR(optimizer, **scheduler_params)
    elif scheduler_name == "cosinescheduler":
        scheduler = LambdaLR(
            optimizer, 
            lr_lambda=lambda it: get_cosine_schedule(iter=it, **scheduler_params)
        )
    elif scheduler_name == "sequentiallr":
        scheduler = _create_sequential_scheduler(optimizer, scheduler_params)
        if not scheduler:
            return None
    else:
        LOGGER.warning(f"Unknown scheduler type: {scheduler_name}. Returning None.")
        return None

    LOGGER.info(f"Created scheduler: {scheduler.__class__.__name__}")
    LOGGER.debug(f"Scheduler parameters: {scheduler_params}")

    # Load scheduler state if specified in config
    epoch = config["training"].get("load_epoch")
    if epoch:
        ckpt_path = get_checkpoints_path(config=config, epoch_num=epoch)
        try:
            state_dict = torch.load(ckpt_path)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu")
        _load_scheduler_state(scheduler, state_dict)

    return scheduler


def _create_sequential_scheduler(optimizer: Optimizer, scheduler_params: dict):
    """Create a SequentialLR scheduler from config."""
    schedulers_config = scheduler_params.get("schedulers", [])
    milestones = scheduler_params.get("milestones", [])
    
    if not schedulers_config:
        LOGGER.error("SequentialLR requires 'schedulers' list in params")
        return None
    
    if not milestones:
        LOGGER.error("SequentialLR requires 'milestones' list in params")
        return None
    
    # Create individual schedulers
    schedulers = []
    for i, sched_config in enumerate(schedulers_config):
        sched_name = sched_config.get("name", "").lower()
        sched_params = sched_config.get("params", {})
        
        if sched_name == "steplr":
            scheduler = StepLR(optimizer, **sched_params)
        elif sched_name == "reducelronplateau":
            LOGGER.error("ReduceLROnPlateau cannot be used in SequentialLR")
            return None
        elif sched_name == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, **sched_params)
        elif sched_name == "cosineannealingwarmrestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, **sched_params)
        elif sched_name == "linearlr":
            scheduler = LinearLR(optimizer, **sched_params)
        elif sched_name == "constantlr":
            scheduler = ConstantLR(optimizer, **sched_params)
        elif sched_name == "cosinescheduler":
            scheduler = LambdaLR(
                optimizer, lr_lambda=lambda it: get_cosine_schedule(it, **sched_params)
            )
        else:
            LOGGER.error(f"Unknown scheduler type in SequentialLR: {sched_name}")
            return None
        
        schedulers.append(scheduler)
        LOGGER.info(f"Created scheduler {i} for SequentialLR: {scheduler.__class__.__name__}")
    
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def _load_scheduler_state(
    scheduler: _LRScheduler | ReduceLROnPlateau,
    state_dict: dict,
) -> None:
    if "scheduler" not in state_dict:
        LOGGER.warning(
            "Checkpoint file does not contain scheduler state. Starting with fresh scheduler."
        )
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.best = state_dict["scheduler"].get("best", scheduler.best)
        scheduler.num_bad_epochs = state_dict["scheduler"].get(
            "num_bad_epochs", scheduler.num_bad_epochs
        )
        LOGGER.info(
            f"Loaded ReduceLROnPlateau state: best={scheduler.best}, "
            f"num_bad_epochs={scheduler.num_bad_epochs}"
        )
    else:
        scheduler.load_state_dict(state_dict["scheduler"])
        LOGGER.info("Loaded scheduler state from checkpoint")
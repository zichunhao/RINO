"""General utilities for lightning modules."""

import logging
import math

from pytorch_lightning import Callback, LightningModule, Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .torch_utils import get_sched, get_submodules, gradient_norm

logger = logging.getLogger(__name__)


class LogGradNorm(Callback):
    """Logs the gradient norm."""

    def __init__(self, logging_interval: int = 1, depth: int = 0):
        self.logging_interval = logging_interval
        self.depth = depth

    def on_before_optimizer_step(
        self, _trainer: Trainer, pl_module: LightningModule, _optimizer: Optimizer
    ):
        if pl_module.global_step % self.logging_interval == 0:
            sub_modules = get_submodules(pl_module, self.depth)
            for subname, module in sub_modules:
                grad = gradient_norm(module)
                if grad > 0:
                    self.log("grad/" + subname, gradient_norm(module))


def get_max_steps(model: LightningModule) -> int:
    """Get the maximum number of steps from the model trainer."""
    try:
        logger.info("Attempting to get the max steps from the model trainer")
        max_steps = model.trainer.max_steps
        if max_steps < 1:
            steps_per_epoch = len(model.trainer.datamodule.train_dataloader())
            max_epochs = model.trainer.max_epochs
            max_steps = steps_per_epoch * max_epochs
        logger.info(f"Success:  max_steps = {max_steps}")
    except Exception as e:
        logger.info(f"Failed to get max steps from the model trainer: {e}")
        max_steps = 0
    return max_steps


def linear_warmup(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    init_factor: float = 1e-2,
) -> LambdaLR:
    """Return a scheduler with a linear warmup."""

    def fn(x: int) -> float:
        return min(1, init_factor + x * (1 - init_factor) / max(1, warmup_steps))

    return LambdaLR(optimizer, fn)


def linear_warmup_exp_decay(
    optimizer: Optimizer,
    model: LightningModule,  # noqa: ARG001
    warmup_steps: int = 1000,
    half_life: int = 1000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-1,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a sqrt decay."""

    def fn(x: int) -> float:
        if x < warmup_steps:
            return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
        decay = -math.log(2) / half_life
        return max(math.exp(decay * (x - warmup_steps)), final_factor)

    return LambdaLR(optimizer, fn)


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    model: LightningModule,
    warmup_steps: int = 100,
    total_steps: int = 1000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-1,
    warmup_ratio: float | None = None,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""
    # Replace the total_steps with the model trainer's actual max_steps
    total_steps = get_max_steps(model) or total_steps

    # Replace the wamup_steps with the ratio
    if warmup_ratio is not None:
        warmup_steps = int(warmup_ratio * total_steps)

    # Define the actual scheduler function
    def fn(x: int) -> float:
        if x < warmup_steps:
            return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
        progress = (x - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(final_factor, lr)

    # The lambda scheduler is the easiest way to define a custom scheduler
    return LambdaLR(optimizer, fn)


def one_cycle(
    model: LightningModule,
    optimizer: Optimizer,
    total_steps: int = 1000,
    **kwargs,
) -> OneCycleLR:
    """Get the learning rate scheduler."""
    total_steps = get_max_steps(model) or total_steps
    return OneCycleLR(
        optimizer,
        **kwargs,
        total_steps=total_steps,
        max_lr=optimizer.param_groups[0]["lr"],
    )


def standard_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler.

    In favour of deprecating this in the future, as it is overly verbose.
    """
    # Finish initialising the partialy created methods
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))

    # Use mltools to initialise the scheduler
    # as we can sync the cycle length with the number of steps per epoch
    sched = get_sched(
        model.hparams.sched_config.mltools,
        opt,
        steps_per_epoch=len(model.trainer.datamodule.train_dataloader()),
        max_epochs=model.trainer.max_epochs,
        max_steps=model.trainer.max_steps,
    )

    # Return the dict for the lightning trainer
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, **model.hparams.sched_config.lightning},
    }


def simple_optim_sched(model: LightningModule) -> dict:
    """Configure the optimizers and learning rate sheduler."""
    opt = model.hparams.optimizer(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = {
        "scheduler": model.hparams.scheduler(optimizer=opt, model=model),
        "interval": "step",
    }
    return [opt], [scheduler]


def multi_optim_sched(model: LightningModule, modules: list) -> dict:
    """Configure multiple optimizers and learning rate sheduler."""
    opt_params = model.hparams.optimizer
    sched_params = model.hparams.scheduler
    assert len(modules) == len(opt_params) == len(sched_params)
    opts = []
    scheds = []
    for module, opt_param, sched_param in zip(
        modules, opt_params, sched_params, strict=False
    ):
        opt = opt_param(filter(lambda p: p.requires_grad, module.parameters()))
        sched = {
            "scheduler": sched_param(optimizer=opt, model=model),
            "interval": "step",
        }
        opts.append(opt)
        scheds.append(sched)
    return opts, scheds

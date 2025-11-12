from pathlib import Path
from typing import Any
from tqdm import tqdm
import yaml
import argparse
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from utils.ckpt import save_checkpoint, get_checkpoints_path
from utils.producers import (
    get_optimizer,
    get_models,
    get_dataloader_and_config,
    get_scheduler,
)
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device
from augmentations import Augmenter
from losses import DINOLoss
from dino_train_batch import process_batch

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
# torch.set_float32_matmul_precision("high")


@torch.no_grad()
def update_momentum_encoder(student, teacher, m):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        # param_t.data = param_t.data * m + param_s.data * (1 - m)
        # This is equivalent to the above line but avoids creating a new tensor
        param_t.data.mul_(m).add_(param_s.data, alpha=(1 - m))

def aggregate_losses(all_batch_losses, accelerator: Accelerator | None = None):
    """
    Aggregate (average) loss metrics across batches and processes correctly
    using sample-weighted means.

    Args:
        all_batch_losses (list[dict]): List of per-batch loss dictionaries.
        accelerator (Accelerator, optional): HF Accelerator instance.

    Returns:
        dict[str, float]: Aggregated mean losses across all batches and processes.
    """
    if not all_batch_losses:
        return {}

    # Collect metric keys
    all_keys = set()
    for batch_loss in all_batch_losses:
        all_keys.update(
            k for k in batch_loss.keys()
            if k not in ["batch_size", "dom_unique"] and "dom_top" not in k
        )
    
    # CRITICAL: Sort keys to ensure consistent order across all processes
    all_keys = sorted(list(all_keys))

    aggregated = {}

    for key in all_keys:
        # Extract values and batch sizes for this key
        local_values = torch.tensor(
            [batch_loss.get(key, 0.0) for batch_loss in all_batch_losses],
            dtype=torch.float32,
        )
        local_batch_sizes = torch.tensor(
            [batch_loss.get("batch_size", 1.0) for batch_loss in all_batch_losses],
            dtype=torch.float32,
        )
        
        local_weighted_sum = (local_values * local_batch_sizes).sum()
        local_total_samples = local_batch_sizes.sum()
        
        # Gather across processes if using distributed training
        if accelerator is not None and accelerator.num_processes > 1:
            local_weighted_sum = local_weighted_sum.to(accelerator.device)
            local_total_samples = local_total_samples.to(accelerator.device)
            
            global_weighted_sum = accelerator.reduce(local_weighted_sum, reduction="sum")
            global_total_samples = accelerator.reduce(local_total_samples, reduction="sum")
            
            weighted_mean = (global_weighted_sum / global_total_samples).item()
        else:
            weighted_mean = (local_weighted_sum / local_total_samples).item()
        
        aggregated[key] = weighted_mean

    return aggregated

def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: int = 0,
) -> np.ndarray:
    """
    Cosine scheduler for learning rate
    Source: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L187C1-L198C20
    """
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

def should_save_and_log(accelerator):
    """Helper function to determine if this process should save checkpoints and log"""
    return accelerator is None or accelerator.is_main_process


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dino_loss: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    epoch: int,
    momentum_schedule: np.ndarray | None = None,
    part_batch_norm: nn.Module | None = None,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()  # Teacher always in eval mode
    dino_loss.train()

    # Use lists to collect all batch losses for proper averaging
    all_batch_losses = []

    with tqdm(
        dataloader, 
        desc="Training", 
        unit="batch", 
        mininterval=10, 
        disable=not (accelerator is None or accelerator.is_main_process)
    ) as pbar:
        for i, batch in enumerate(pbar):
            batch_losses = process_batch(
                student=student,
                teacher=teacher,
                part_batch_norm=part_batch_norm,
                dino_loss=dino_loss,
                batch=batch,
                device=device,
                config=config,
                optimizer=optimizer,
                augmenter=augmenter,
                accelerator=accelerator,
            )
            
            # Store batch losses for aggregation
            all_batch_losses.append(batch_losses)
            
            # Update progress bar with batch metrics (only on main process)
            if should_save_and_log(accelerator):
                pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_losses.items() if k != "batch_size"})

            # Update teacher
            if momentum_schedule is None:
                momentum = config["training"].get("teacher_momentum", 0.994)
            else:
                it = epoch * len(dataloader) + i
                momentum = momentum_schedule[it]
                
            # Update teacher
            with torch.no_grad():
                update_momentum_encoder(student=student, teacher=teacher, m=momentum)

    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict


@torch.no_grad()
def validate(
    student: nn.Module,
    teacher: nn.Module,
    dino_loss: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    part_batch_norm: nn.Module | None = None,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    dino_loss.eval()

    # Use lists to collect all batch losses for proper averaging
    all_batch_losses = []

    with tqdm(
        dataloader, 
        desc="Validation", 
        unit="batch", 
        mininterval=10, 
        disable=not (accelerator is None or accelerator.is_main_process)
    ) as pbar:
        for batch in pbar:
            batch_losses = process_batch(
                student=student,
                teacher=teacher,
                part_batch_norm=part_batch_norm,
                dino_loss=dino_loss,
                batch=batch,
                device=device,
                config=config,
                optimizer=None,
                augmenter=augmenter,
                accelerator=accelerator,
            )
            
            # Store batch losses for aggregation
            all_batch_losses.append(batch_losses)
            
            # Update progress bar with batch metrics (only on main process)
            if accelerator is None or accelerator.is_main_process:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_losses.items() if k != "batch_size"})
            
    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict


def get_checkpoint_dict(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: optim.Optimizer,
    part_batch_norm: nn.Module | None,
    scheduler: _LRScheduler | ReduceLROnPlateau | None,
    epoch: int,
    train_loss: dict[str, float],
    val_loss: dict[str, float],
) -> dict[str, Any]:
    checkpoint_dict = {
        "epoch": epoch,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            checkpoint_dict["scheduler"] = {
                "best": scheduler.best,
                "num_bad_epochs": scheduler.num_bad_epochs,
            }
        else:
            checkpoint_dict["scheduler"] = scheduler.state_dict()
    if part_batch_norm is not None:
        checkpoint_dict["part_batch_norm"] = part_batch_norm.state_dict()
    else:
        checkpoint_dict["part_batch_norm"] = None
    return checkpoint_dict


def get_training_start_params(
    config: dict[str, Any],
    device: torch.device,
    accelerator: Accelerator | None = None,
) -> tuple[int, float]:
    """
    Determine the true start_epoch and best_val_loss based on the configuration
    and any existing checkpoints.
    """
    training_params = config["training"]
    start_epoch = training_params.get("load_epoch")
    if start_epoch is None:
        return 0, float("inf")
    best_val_loss = float("inf")

    checkpoint_path = get_checkpoints_path(config, start_epoch)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]["loss"]
    
    # Only log on main process
    if should_save_and_log(accelerator):
        LOGGER.info(f"Loaded checkpoint info from {checkpoint_path}")
        LOGGER.info(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss}")

    return start_epoch, best_val_loss


def train(config: dict[str, Any]) -> None:

    use_accelerate = config.get("accelerate", False)

    # Initialize accelerator first
    if use_accelerate:
        # https://github.com/huggingface/transformers/issues/34699#issuecomment-2510417946
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,   # Main process loads batches
            split_batches=True       # Split fetched batches across processes
        )
        # cached datasets load a big dataset into memory, 
        # so we need to set a longer timeout
        kwargs = InitProcessGroupKwargs(timeout=timedelta(days=365))
        accelerator = Accelerator(
            dataloader_config=dataloader_config, kwargs_handlers=[kwargs]
        )
        device = accelerator.device

        # Debug information
        if accelerator.is_main_process:
            LOGGER.info("Using accelerate for distributed training.")
        LOGGER.info(f"Process index: {accelerator.process_index}")
        LOGGER.info(f"Local process index: {accelerator.local_process_index}")
        LOGGER.info(f"Number of processes: {accelerator.num_processes}")
        LOGGER.info(f"Is main process: {accelerator.is_main_process}")
    else:
        accelerator = None
        device = config.get("device", None)
        if device is None:
            device = get_available_device()
        else:
            device = torch.device(device)

        # Check device (only when not using accelerate)
        try:
            torch.empty(1).to(device)
        except AssertionError as e:
            LOGGER.error(f"Error: {e}")
            device = get_available_device()
            LOGGER.info(f"Using default device: {device}")
            
    if should_save_and_log(accelerator):
        LOGGER.info(f"PyTorch version: {torch.__version__}")
        LOGGER.info(f"CUDA version: {torch.version.cuda}")
        LOGGER.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        LOGGER.info(f"Training with config: {config}")

    # Get dataloaders
    train_dataloader, dataloader_config = get_dataloader_and_config(
        config=config,
        split="train",
        mode="training",
        accelerator=accelerator,
    )
    val_dataloader, _ = get_dataloader_and_config(
        config=config,
        split="val",
        mode="training",
        accelerator=accelerator,
    )

    # Initialize augmenter with feature information from dataloader config
    augmenter = Augmenter(
        labels_parts=dataloader_config.outputs.sequence,
        labels_jet=dataloader_config.outputs.class_,
        **config["augmentation_params"]["augmenter_kwargs"],
    )

    # Initialize student and teacher networks
    part_features = dataloader_config.outputs.sequence
    part_dim = len(part_features)

    # DON'T move to device before accelerator.prepare() if using accelerate
    student, teacher, part_batch_norm = get_models(
        part_dim=part_dim,
        config=config,
        mode="training",
        device=(
            device if not use_accelerate else None
        ),  # Pass None for device if using accelerate
    )

    # DINO loss - don't move to device if using accelerate
    num_global_views = sum(
        view["num"] for view in config["augmentation_params"]["global"]
    )
    num_local_views = sum(
        view["num"] for view in config["augmentation_params"]["local"]
    )
    dino_loss = DINOLoss(
        out_dim=config["model_params"]["proj_dim"],
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        **config["loss_params"]["dino"],
    )

    # Only move to device if not using accelerate
    if not use_accelerate:
        dino_loss = dino_loss.to(device)

    if should_save_and_log(accelerator):
        LOGGER.info(f"Loss params: {config['loss_params']}")

    # Optimizer (only for student)
    optimizer = get_optimizer(
        config=config,
        model_params=student.parameters(),
    )
    if should_save_and_log(accelerator):
        LOGGER.info(f"Optimizer: {optimizer}")

    # Scheduler
    scheduler = get_scheduler(optimizer=optimizer, config=config)
    if should_save_and_log(accelerator):
        LOGGER.info(f"Scheduler: {scheduler}")

    # Prepare everything with accelerator if using it
    if use_accelerate:
        # Prepare all components that need gradient synchronization or device placement
        student, teacher, optimizer, train_dataloader, val_dataloader, dino_loss = (
            accelerator.prepare(
                student, teacher, optimizer, train_dataloader, val_dataloader, dino_loss
            )
        )

        # Part batch norm should also be prepared if it exists
        if part_batch_norm is not None:
            part_batch_norm = accelerator.prepare(part_batch_norm)

        # Scheduler might need to be prepared depending on the type
        if scheduler is not None and hasattr(scheduler, "step"):
            # Some schedulers might need preparation, but be careful here
            # Most don't need it, but if you're having issues, you can try:
            # scheduler = accelerator.prepare(scheduler)
            pass
        LOGGER.info("Accelerator prepared all components.")

    # Training loop
    training_params = config["training"]
    start_epoch, best_val_loss = get_training_start_params(
        config=config, device=device, accelerator=accelerator
    )

    patience = training_params.get("patience", float("inf"))
    num_stale_epochs = 0
    num_epochs: int = training_params["num_epochs"]
    if start_epoch > 0:
        dino_loss.resume_epoch(start_epoch)

    # momentum schedule for updating teacher
    momentum_schedule = cosine_scheduler(
        base_value=config["training"].get("teacher_momentum", 0.996),
        final_value=1.0,
        epochs=num_epochs,
        niter_per_ep=len(train_dataloader),
        warmup_epochs=config["training"].get("teacher_momentum_warmup_epochs", 0),
        start_warmup_value=config["training"].get("teacher_momentum_warmup_start", 0.0),
    )

    should_stop = False
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if epoch == 0 and part_batch_norm is not None:
            part_batch_norm.train()
        elif part_batch_norm is not None:
            part_batch_norm.eval()

        detect_anomaly = config["training"].get("autograd_detect_anomaly", False)
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            train_metrics = train_epoch(
                student=student,
                teacher=teacher,
                part_batch_norm=part_batch_norm,
                dino_loss=dino_loss,
                optimizer=optimizer,
                dataloader=train_dataloader,
                device=device,
                config=config,
                augmenter=augmenter,
                epoch=epoch,
                momentum_schedule=momentum_schedule,
                accelerator=accelerator,
            )
        
        if should_save_and_log(accelerator):
            # Save checkpoint before validation
            checkpoint_dict = get_checkpoint_dict(
                student=student,
                teacher=teacher,
                part_batch_norm=part_batch_norm,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_metrics,
                val_loss=float("inf"),
            )
            saved_path = save_checkpoint(
                checkpoint_dict=checkpoint_dict,
                config=config,
                epoch_num=epoch,
            )

        val_metrics = validate(
            student=student,
            teacher=teacher,
            part_batch_norm=part_batch_norm,
            dino_loss=dino_loss,
            dataloader=val_dataloader,
            device=device,
            config=config,
            augmenter=augmenter,
            accelerator=accelerator,
        )
        try:
            dino_loss.step_epoch()
        except AttributeError:
            # DDP
            dino_loss.module.step_epoch()

        val_loss = val_metrics["loss"]
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        if should_save_and_log(accelerator):
            log_message = f"Epoch [{epoch}/{num_epochs}]\n"
            log_message += f"Train loss: {train_metrics['loss']:.8f}\n"
            for k, v in train_metrics.items():
                if k != "loss":
                    log_message += f"Train {k}: {v:.8f}\n"
            
            log_message += f"Valid loss: {val_metrics['loss']:.8f}\n"
            for k, v in val_metrics.items():
                if k != "loss":
                    log_message += f"Valid {k}: {v:.8f}\n"
            
            LOGGER.info(log_message)

        # Save checkpoint
        if should_save_and_log(accelerator):
            checkpoint_dict = get_checkpoint_dict(
                student=student,
                teacher=teacher,
                part_batch_norm=part_batch_norm,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_metrics,
                val_loss=val_metrics,
            )
            saved_path = save_checkpoint(
                checkpoint_dict=checkpoint_dict,
                config=config,
                epoch_num=epoch,
            )
            LOGGER.info(
                f"Checkpoint (keys: {list(checkpoint_dict.keys())}) saved to {saved_path}"
            )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if should_save_and_log(accelerator):
                checkpoint_dict = get_checkpoint_dict(
                    student=student,
                    teacher=teacher,
                    part_batch_norm=part_batch_norm,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_loss=train_metrics,
                    val_loss=val_metrics,
                )
                saved_path = save_checkpoint(
                    checkpoint_dict=checkpoint_dict,
                    config=config,
                    epoch_num="best",
                )
                LOGGER.info(f"Best model saved to {saved_path}")
            num_stale_epochs = 0
        else:
            num_stale_epochs += 1
            if num_stale_epochs >= patience:
                if should_save_and_log(accelerator):
                    LOGGER.info(
                        f"Validation loss has not improved for {patience} epochs. "
                        f"Stopping training at epoch {epoch}."
                    )
                should_stop = True
        if should_save_and_log(accelerator):
            LOGGER.info(f"num_stale_epochs/patience: {num_stale_epochs}/{patience}")
            
        if should_stop:
            if (accelerator is not None) and (accelerator.num_processes > 1) and (accelerator.is_main_process):
                accelerator.set_trigger()
            else:
                break
        
        if accelerator is not None:
            if accelerator.check_trigger():
                break

    if should_save_and_log(accelerator):
        # Save the final model
        checkpoint_dict = get_checkpoint_dict(
            student=student,
            teacher=teacher,
            part_batch_norm=part_batch_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_metrics,
            val_loss=val_metrics,
        )
        saved_path = save_checkpoint(
            checkpoint_dict=checkpoint_dict,
            config=config,
            epoch_num="final",
        )
        LOGGER.info(f"Final model saved to {saved_path}")
    
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINO model")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lf",
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file. If not specified, logs will be written to stdout.",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    configure_logger(
        logger=LOGGER,
        name="DINO Training",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    train(config=config)

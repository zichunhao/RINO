from pathlib import Path
from typing import Any
from tqdm import tqdm
import yaml
import argparse
import torch
import torch.nn as nn
from torch import optim
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from losses import BinaryFocalLoss, FocalLoss
from utils.ckpt import save_checkpoint, get_checkpoints_path
from utils.producers import (
    get_optimizer_finetune,
    get_param_groups,
    get_models_finetune,
    get_dataloader_and_config,
    get_scheduler,
)
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device
from models import ModelWithNewHead

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent


def aggregate_classification_metrics(
    batch_metrics,
    accelerator=None
):
    """
    Aggregate classification metrics across batches and optionally across distributed processes.
    
    Args:
        batch_metrics: List of batch metric dictionaries
        accelerator: Accelerator instance for distributed training
    
    Returns:
        Dictionary with aggregated classification metrics including accuracy
    """
    if not batch_metrics:
        return {}
    
    if accelerator is not None:
        # Distributed training: gather across processes
        
        # Collect all scalar metrics for gathering
        scalar_metrics = {}
        list_metrics = {}
        
        # Process each metric type
        loss_keys = ["loss", "lwf_loss", "l2sp_loss", "classification_loss"]
        count_keys = ["true_count", "batch_size"]
        list_keys = ["predictions", "true_labels", "probabilities"]
        
        # Aggregate scalar metrics locally first
        local_aggregated = {}
        
        # Weighted averages for losses
        total_samples = sum(batch["batch_size"] for batch in batch_metrics)
        for key in loss_keys:
            if total_samples > 0:
                weighted_sum = sum(
                    batch.get(key, 0.0) * batch["batch_size"] 
                    for batch in batch_metrics
                )
                local_aggregated[key] = weighted_sum / total_samples
            else:
                local_aggregated[key] = 0.0
        
        # Sums for counts
        for key in count_keys:
            local_aggregated[key] = sum(batch.get(key, 0) for batch in batch_metrics)
        
        # Convert local aggregates to tensors for gathering
        for key in loss_keys + count_keys:
            tensor_val = torch.tensor(local_aggregated[key], device=accelerator.device, dtype=torch.float32)
            scalar_metrics[key] = tensor_val
        
        # Gather scalar metrics from all processes
        gathered_metrics = {}
        for key, tensor in scalar_metrics.items():
            gathered_tensor = accelerator.gather(tensor)
            if key in loss_keys:
                # For losses, we need to re-weight by total samples across all processes
                if key == "loss" or "loss" in key:
                    # We need to gather the weighted sums and total samples separately
                    pass  # Will handle below
                else:
                    gathered_metrics[key] = gathered_tensor.mean().item()
            else:
                # For counts, sum across processes
                gathered_metrics[key] = gathered_tensor.sum().item()
        
        # Handle weighted average for losses properly across processes
        total_samples_tensor = scalar_metrics["batch_size"]
        gathered_total_samples = accelerator.gather(total_samples_tensor).sum().item()
        
        for key in loss_keys:
            # Gather weighted sums
            local_weighted_sum = local_aggregated[key] * local_aggregated["batch_size"]
            weighted_sum_tensor = torch.tensor(local_weighted_sum, device=accelerator.device, dtype=torch.float32)
            gathered_weighted_sum = accelerator.gather(weighted_sum_tensor).sum().item()
            
            if gathered_total_samples > 0:
                gathered_metrics[key] = gathered_weighted_sum / gathered_total_samples
            else:
                gathered_metrics[key] = 0.0
        
        # Handle list metrics
        for key in list_keys:
            all_items = []
            for batch in batch_metrics:
                batch_items = batch.get(key, [])
                if isinstance(batch_items, list):
                    all_items.extend(batch_items)
                else:
                    all_items.append(batch_items)
            
            if all_items and isinstance(all_items[0], (int, float)):
                items_tensor = torch.tensor(all_items, device=accelerator.device)
                gathered_items = accelerator.gather_for_metrics(items_tensor)
                gathered_metrics[key] = gathered_items.cpu().tolist()
            else:
                gathered_metrics[key] = all_items
        
        # Calculate accuracy
        if "true_count" in gathered_metrics and "batch_size" in gathered_metrics:
            total_true = gathered_metrics["true_count"]
            total_samples = gathered_metrics["batch_size"]
            gathered_metrics["acc"] = total_true / total_samples if total_samples > 0 else 0.0
        
        return gathered_metrics
    
    else:
        # Single process: compute aggregations normally
        total_samples = sum(batch["batch_size"] for batch in batch_metrics)
        
        aggregated_metrics = {}
        
        # Weighted averages for losses
        loss_keys = ["loss", "lwf_loss", "l2sp_loss", "classification_loss"]
        for key in loss_keys:
            if total_samples > 0:
                weighted_sum = sum(
                    batch.get(key, 0.0) * batch["batch_size"]
                    for batch in batch_metrics
                )
                aggregated_metrics[key] = weighted_sum / total_samples
            else:
                aggregated_metrics[key] = 0.0
        
        # Sums for counts
        count_keys = ["true_count", "batch_size"]
        for key in count_keys:
            aggregated_metrics[key] = sum(batch.get(key, 0) for batch in batch_metrics)
        
        # Concatenate lists
        list_keys = ["predictions", "true_labels", "probabilities"]
        for key in list_keys:
            all_items = []
            for batch in batch_metrics:
                batch_items = batch.get(key, [])
                if isinstance(batch_items, list):
                    all_items.extend(batch_items)
                else:
                    all_items.append(batch_items)
            aggregated_metrics[key] = all_items
        
        # Calculate accuracy
        if "true_count" in aggregated_metrics and "batch_size" in aggregated_metrics:
            total_true = aggregated_metrics["true_count"]
            total_samples = aggregated_metrics["batch_size"]
            aggregated_metrics["acc"] = total_true / total_samples if total_samples > 0 else 0.0
        
        return aggregated_metrics


def should_save_and_log(accelerator):
    """Helper function to determine if this process should save checkpoints and log"""
    return accelerator is None or accelerator.is_main_process

def train_epoch(
    model: ModelWithNewHead,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    part_batch_norm: nn.Module | None = None,
    accelerator: Accelerator | None = None,
    initial_model: ModelWithNewHead | None = None,
    lwf_alpha: float = 0,
    l2sp_alpha: float = 0,
) -> dict[str, float]:
    model.train()

    # Use lists to collect all batch metrics for proper averaging
    all_batch_metrics = []

    with tqdm(
        dataloader, 
        desc="Training", 
        unit="batch", 
        mininterval=10, 
        disable=not (accelerator is None or accelerator.is_main_process)
    ) as pbar:
        for batch in pbar:
            batch_metrics = process_batch(
                model=model,
                criterion=criterion,
                batch=batch,
                device=device,
                part_batch_norm=part_batch_norm,
                optimizer=optimizer,
                accelerator=accelerator,
                config=config,
                initial_model=initial_model,
                lwf_alpha=lwf_alpha,
                l2sp_alpha=l2sp_alpha,
            )

            # Store batch metrics for aggregation
            all_batch_metrics.append(batch_metrics)

            # Calculate batch accuracy for progress bar
            batch_acc = batch_metrics["true_count"] / batch_metrics["batch_size"]
            
            # Update progress bar with batch metrics (only on main process)
            if should_save_and_log(accelerator):
                pbar.set_postfix(
                    {
                        "loss": f"{batch_metrics['loss']:.4f}",
                        "lwf_loss": f"{batch_metrics['lwf_loss']:.4f}",
                        "l2sp_loss": f"{batch_metrics['l2sp_loss']:.4f}",
                        "classification_loss": f"{batch_metrics['classification_loss']:.4f}",
                        "acc": f"{batch_acc:.4f}",
                    }
                )

    # Aggregate metrics across all batches and processes
    epoch_metrics = aggregate_classification_metrics(all_batch_metrics, accelerator)
    return epoch_metrics


@torch.no_grad()
def validate(
    model: ModelWithNewHead,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    part_batch_norm: nn.Module | None = None,
    accelerator: Accelerator | None = None,
    initial_model: ModelWithNewHead | None = None,
    lwf_alpha: float = 0,
    l2sp_alpha: float = 0,
) -> dict[str, float]:
    model.eval()

    # Use lists to collect all batch metrics for proper averaging
    all_batch_metrics = []

    with tqdm(
        dataloader, 
        desc="Validation", 
        unit="batch", 
        mininterval=10, 
        disable=not (accelerator is None or accelerator.is_main_process)
    ) as pbar:
        for batch in pbar:
            batch_metrics = process_batch(
                model=model,
                criterion=criterion,
                batch=batch,
                device=device,
                part_batch_norm=part_batch_norm,
                optimizer=None,
                accelerator=accelerator,
                config=config,
                initial_model=initial_model,
                lwf_alpha=lwf_alpha,
                l2sp_alpha=l2sp_alpha,
            )

            # Store batch metrics for aggregation
            all_batch_metrics.append(batch_metrics)

            # Calculate batch accuracy for progress bar
            batch_acc = batch_metrics["true_count"] / batch_metrics["batch_size"]
            
            # Update progress bar with batch metrics (only on main process)
            if should_save_and_log(accelerator):
                pbar.set_postfix(
                    {
                        "loss": f"{batch_metrics['loss']:.4f}",
                        "lwf_loss": f"{batch_metrics['lwf_loss']:.4f}",
                        "l2sp_loss": f"{batch_metrics['l2sp_loss']:.4f}",
                        "acc": f"{batch_acc:.4f}",
                    }
                )

    # Aggregate metrics across all batches and processes
    epoch_metrics = aggregate_classification_metrics(all_batch_metrics, accelerator)
    return epoch_metrics


def process_batch(
    model: ModelWithNewHead,
    criterion: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    part_batch_norm: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    accelerator: Accelerator | None = None,
    config: dict[str, Any] | None = None,
    initial_model: ModelWithNewHead | None = None,
    lwf_alpha: float = 0,
    l2sp_alpha: float = 0,
) -> dict[str, float]:
    particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(device)
    jets = torch.tensor(batch["class_"], dtype=torch.float32).to(device)
    mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)
    labels = torch.tensor(batch["aux"]["label"], dtype=torch.long).to(device)

    if part_batch_norm is not None:
        particles = part_batch_norm(particles, mask=mask)

    # Get logits and representations from the unified model
    logits, rep = model(particles=particles, jets=jets, mask=mask, include_dino_head=False)
    lwf_loss = 0
    l2sp_loss = 0
    loss = 0
    if initial_model is not None:
        if lwf_alpha > 0:
            _, initial_rep = initial_model(particles=particles, jets=jets, mask=mask, include_dino_head=False)
            lwf_loss = nn.functional.mse_loss(rep, initial_rep) / rep.shape[-1]  # normalize by feature dimension
            loss += lwf_alpha * lwf_loss
            lwf_loss = lwf_loss.detach().cpu().item()
        if l2sp_alpha > 0:
            l2sp_loss = 0
            n_params = 0
            try:
                backbone = model.backbone
            except AttributeError:
                backbone = model.module.backbone
            try: 
                initial_backbone = initial_model.backbone
            except AttributeError:
                initial_backbone = initial_model.module.backbone
            # compute MSE
            for param, init_param in zip(backbone.parameters(), initial_backbone.parameters()):
                l2sp_loss += torch.sum((param - init_param) ** 2)
                n_params += param.numel()
            l2sp_loss = l2sp_loss / n_params
            loss += l2sp_alpha * l2sp_loss
            l2sp_loss = l2sp_loss.detach().cpu().item()

    # Check if we're doing binary classification based on logits shape
    if logits.shape[-1] == 1:
        # For binary classification with BCEWithLogitsLoss
        # logits should be shape (batch_size, 1) or (batch_size,)
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # Remove the last dimension if it's 1
        
        # Convert labels to float for BCEWithLogitsLoss
        labels_float = labels.float()
        
        # Calculate loss
        classification_loss = criterion(logits, labels_float)
        loss += classification_loss
        
        # Get predictions (sigmoid + threshold at 0.5)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        
        # For consistency with multi-class case, expand probabilities to 2D
        # [prob_class_0, prob_class_1]
        probs_expanded = torch.stack([1 - probs, probs], dim=1)
        
    else:
        # For multi-class classification with CrossEntropyLoss
        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Calculate loss
        classification_loss = criterion(logits, labels)
        loss += classification_loss
        probs_expanded = probs

    classification_loss = classification_loss.detach().cpu().item()

    # Calculate metrics
    true_count = (preds == labels).sum().item()  # Count of correct predictions
    batch_size = labels.size(0)  # Batch size for proper averaging

    # Create detailed metrics dictionary
    metrics = {
        "loss": loss.item(),
        "lwf_loss": lwf_loss,
        "l2sp_loss": l2sp_loss,
        "classification_loss": classification_loss,
        "true_count": true_count,
        "batch_size": batch_size,
        "predictions": preds.cpu().tolist(),
        "true_labels": labels.cpu().tolist(),
        "probabilities": probs_expanded.detach().cpu().tolist(),
    }

    if optimizer is not None:
        optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Handle gradient clipping
        grad_clip = config.get("training", {}).get("grad_clip", None) if config else None
        if grad_clip is not None and grad_clip > 0:
            if accelerator is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

        optimizer.step()

    return metrics


def get_state_dict(model: nn.Module):
    if hasattr(model, 'module'):
        # a feature of nn.DataParallel
        return model.module.state_dict()
    else:
        return model.state_dict()


def get_checkpoint_dict(
    model: ModelWithNewHead,
    optimizer: optim.Optimizer,
    part_batch_norm: nn.Module | None,
    scheduler: _LRScheduler | ReduceLROnPlateau | None,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    initial_model: ModelWithNewHead | None = None,
) -> dict[str, Any]:
    checkpoint_dict = {
        "epoch": epoch,
        "model": get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
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
        checkpoint_dict["part_batch_norm"] = get_state_dict(part_batch_norm)
    else:
        checkpoint_dict["part_batch_norm"] = None
        
    if initial_model is not None:
        checkpoint_dict["initial_model"] = get_state_dict(initial_model)
    else:
        checkpoint_dict["initial_model"] = None

    return checkpoint_dict


def set_backbone_trainable(model: ModelWithNewHead, trainable: bool = True) -> None:
    """Set the backbone to be trainable or frozen."""
    if trainable:
        try:
            model.train_backbone()  # Use the method from ModelWithNewHead
        except AttributeError:
            # Model is wrapped (e.g., by DistributedDataParallel)
            model.module.train_backbone()
    else:
        try:
            model.freeze_backbone()  # Use the method from ModelWithNewHead
        except AttributeError:
            # Model is wrapped (e.g., by DistributedDataParallel)
            model.module.freeze_backbone()

def get_training_start_params(
    config: dict[str, Any],
    device: torch.device,
    accelerator: Accelerator | None
) -> tuple[int, float]:
    """
    Determine the true start_epoch and best_val_loss based on the configuration
    and any existing checkpoints.

    Args:
        config (dict[str, Any]): The configuration dictionary.
        device (torch.device): The device to load the checkpoint to.
        logger (logging.Logger): Logger for outputting information.

    Returns:
        Tuple[int, float]: The true start_epoch and best_val_loss.
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
    best_val_loss = checkpoint["val_metrics"]["loss"]
    
    # Only log on main process
    if should_save_and_log(accelerator):
        LOGGER.info(f"Loaded checkpoint info from {checkpoint_path}")
        LOGGER.info(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss}")

    return start_epoch, best_val_loss

def load_best_checkpoint(
    config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Load the best checkpoint based on validation loss."""
    checkpoint_path = get_checkpoints_path(config, "best")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


# Modified sections of the main train function:
def train(config: dict[str, Any]) -> None:
    
    device = config.get("device", None)
    if device is None:
        device = get_available_device()
    else:
        device = torch.device(device)

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
        LOGGER.info(f"Using Accelerate with device: {device}")
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

    # Initialize model
    part_features = dataloader_config.outputs.sequence
    part_dim = len(part_features)
    model, part_batch_norm = get_models_finetune(
        part_dim=part_dim,
        config=config,
        mode="training",
        device=(
            device if not use_accelerate else None
        ),  # Pass None for device if using accelerate
        train_head=True,  # need new head
    )

    save_each_epoch = config["training"].get("save_each_epoch", False)
    if should_save_and_log(accelerator):
        if save_each_epoch:
            LOGGER.info("Saving model at each epoch")
        else:
            LOGGER.info("Only saving model at best epochs")

    lwf_alpha = config["training"].get("lwf_alpha", 0)
    l2sp_alpha = config["training"].get("l2sp_alpha", 0)

    if lwf_alpha > 0 or l2sp_alpha > 0:
        if should_save_and_log(accelerator):
            LOGGER.info(f"Using alpha_lwf={lwf_alpha} and alpha_l2sp={l2sp_alpha}")
            LOGGER.info("Initializing initial model for LwF/L2SP")
        
        # Create a new model instance with the same architecture
        # Cannot deepcopy because of weight norm
        initial_model, _ = get_models_finetune(
            part_dim=part_dim,
            config=config,
            mode="training",
            device=(device if not use_accelerate else None),
            train_head=True,
        )
        
        # Copy the state dict instead of deepcopy
        try:
            initial_model.load_state_dict(model.state_dict())
        except RuntimeError:
            # If model is wrapped (e.g., by DistributedDataParallel), access the module
            initial_model.module.load_state_dict(model.state_dict())
        initial_model.eval()  # Set to eval mode
        for param in initial_model.parameters():
            param.requires_grad = False
    else:
        LOGGER.info("LwF is not enabled; initial model will not be used")
        initial_model = None

    # Loss function
    output_dim = config["head_params"]["output_dim"]
    bce_pos_weight = config["training"].get("bce_pos_weight", 1.0)
    if output_dim == 1:
        # if should_save_and_log(accelerator):
        #     LOGGER.info("Using BCEWithLogitsLoss for binary classification")
        # if bce_pos_weight > 0 and bce_pos_weight != 1.0:
        #     LOGGER.info(f"Using BCE loss positive class weight: {bce_pos_weight}")
        #     pos_weight = torch.tensor(bce_pos_weight, device=device)
        # else:
        #     pos_weight = None
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        if bce_pos_weight > 0 and bce_pos_weight != 1.0:
            use_focal_loss = config["training"].get("use_focal_loss", False)
            if use_focal_loss:
                if should_save_and_log(accelerator):
                    LOGGER.info("Using BinaryFocalLoss for binary classification")
                criterion = BinaryFocalLoss()
            else:
                if should_save_and_log(accelerator):
                    LOGGER.info(f"Using BCEWithLogitsLoss for binary classification with pos_weight={bce_pos_weight}")
                pos_weight = torch.tensor(bce_pos_weight, device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if should_save_and_log(accelerator):
                LOGGER.info("Using BCEWithLogitsLoss for binary classification")
            criterion = nn.BCEWithLogitsLoss()

    else:
        if should_save_and_log(accelerator):
            LOGGER.info("Using CrossEntropyLoss for multi-class classification")
        criterion = nn.CrossEntropyLoss()

    # Get starting epoch and best validation loss
    start_epoch, best_val_loss = get_training_start_params(config=config, device=device, accelerator=accelerator)

    # Training parameters
    training_params = config["training"]
    num_epochs: int = training_params["num_epochs"]
    patience = training_params.get("patience", float("inf"))
    freeze_backbone = training_params.get("freeze_backbone", None)
    freeze_embedding = training_params.get("freeze_embedding", False)
    if freeze_embedding:
        LOGGER.info("Freezing embedding layers")
        model.freeze_embedding()

    # Initialize freezing-related variables (for finetuning)
    is_backbone_frozen = False
    freeze_epoch = None

    if isinstance(freeze_backbone, bool):
        if freeze_backbone:  # True means auto
            is_backbone_frozen = True
            if should_save_and_log(accelerator):  # Add this conditional
                LOGGER.info("Starting training with frozen backbone (auto unfreeze)")
            set_backbone_trainable(model, trainable=False)
        else:  # False means no freezing
            if should_save_and_log(accelerator):  # Add this conditional
                LOGGER.info("Starting training with trainable backbone")
            set_backbone_trainable(model, trainable=True)
    elif isinstance(freeze_backbone, int):  # Int means freeze until specific epoch
        is_backbone_frozen = True
        freeze_epoch = freeze_backbone
        if should_save_and_log(accelerator):  # Add this conditional
            LOGGER.info(f"Starting training with frozen backbone (unfreeze at epoch {freeze_epoch})")
        set_backbone_trainable(model, trainable=False)
    else:  # None or any other value means no freezing
        if should_save_and_log(accelerator):  # Add this conditional
            LOGGER.info("Starting training with trainable backbone")
        set_backbone_trainable(model, trainable=True)

    def setup_optimizer_scheduler():
        try:
            param_groups = get_param_groups(model.backbone, model.head, config)  # Access components
        except AttributeError:
            # If model is wrapped (e.g., by DistributedDataParallel), access the module
            param_groups = get_param_groups(model.module.backbone, model.module.head, config)
        optimizer = get_optimizer_finetune(config, param_groups)
        scheduler = get_scheduler(optimizer=optimizer, config=config)
        return optimizer, scheduler

    # Initial optimizer and scheduler setup
    optimizer, scheduler = setup_optimizer_scheduler()
    
    if use_accelerate:
        model, optimizer, train_dataloader, val_dataloader = (
            accelerator.prepare(
                model,
                optimizer, 
                train_dataloader,
                val_dataloader,
            )
        )
        if initial_model is not None:
            initial_model = accelerator.prepare(initial_model)

        # Also prepare part_batch_norm if it exists
        if part_batch_norm is not None:
            part_batch_norm = accelerator.prepare(part_batch_norm)

    # If resuming training, load states
    if start_epoch > 0:
        checkpoint_path = get_checkpoints_path(config, start_epoch - 1)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            # If model is wrapped (e.g., by DistributedDataParallel), access the module
            model.module.load_state_dict(checkpoint["model"])
        if initial_model is not None:
            if "initial_model" in checkpoint:
                initial_model.load_state_dict(checkpoint["initial_model"])
            else:
                # If initial_model was not saved, we can skip loading it
                LOGGER.warning("No initial model state found in checkpoint, skipping load and disable LWF&L2SP")
                initial_model = None
                lwf_alpha = 0  # Disable LwF if no initial model is available
                l2sp_alpha = 0  # Disable L2SP if no initial model is available
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.best = checkpoint["scheduler"]["best"]
                scheduler.num_bad_epochs = checkpoint["scheduler"]["num_bad_epochs"]
            else:
                scheduler.load_state_dict(checkpoint["scheduler"])
                
        if part_batch_norm is not None and checkpoint.get("part_batch_norm") is not None:
            try:
                part_batch_norm.load_state_dict(checkpoint["part_batch_norm"])
            except RuntimeError:
                # If part_batch_norm is wrapped, access the module
                part_batch_norm.module.load_state_dict(checkpoint["part_batch_norm"])

        if should_save_and_log(accelerator):
            LOGGER.info(f"Resumed training from checkpoint {checkpoint_path}")

    # Training loop
    num_stale_epochs = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Check if we should unfreeze based on epoch number
        if freeze_epoch is not None and epoch >= freeze_epoch and is_backbone_frozen:
            if should_save_and_log(accelerator):
                LOGGER.info(f"Unfreezing backbone at epoch {epoch} (scheduled)")
            is_backbone_frozen = False
            set_backbone_trainable(model, trainable=True)
            optimizer, scheduler = setup_optimizer_scheduler()
            if use_accelerate:
                optimizer = accelerator.prepare(optimizer)
            best_val_loss = float("inf")
            num_stale_epochs = 0

        train_metrics = train_epoch(
            model=model,  # Pass unified model
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            config=config,
            part_batch_norm=part_batch_norm,
            accelerator=accelerator,
            initial_model=initial_model,
            lwf_alpha=lwf_alpha,
            l2sp_alpha=l2sp_alpha,
        )

        val_metrics = validate(
            model=model,  # Pass unified model
            criterion=criterion,
            dataloader=val_dataloader,
            device=device,
            config=config,
            part_batch_norm=part_batch_norm,
            accelerator=accelerator,
            initial_model=initial_model,
            lwf_alpha=lwf_alpha,
            l2sp_alpha=l2sp_alpha,
        )

        val_loss = val_metrics["loss"]
        val_loss_improved = val_loss < best_val_loss
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if should_save_and_log(accelerator):
            log_message = f"Epoch [{epoch}/{start_epoch + num_epochs}]\n"
            log_message += f"Train loss: {train_metrics['loss']:.6f}, "
            log_message += f"LwF loss: {train_metrics['lwf_loss']:.6f}, "
            log_message += f"L2SP loss: {train_metrics['l2sp_loss']:.6f}, "
            log_message += f"Classification loss: {train_metrics['classification_loss']:.6f}, "
            log_message += f"Acc: {train_metrics['acc']:.4f}\n"
            log_message += f"Valid loss: {val_metrics['loss']:.6f}, "
            log_message += f"LwF loss: {val_metrics['lwf_loss']:.6f}, "
            log_message += f"L2SP loss: {val_metrics['l2sp_loss']:.6f}, "
            log_message += f"Classification loss: {val_metrics['classification_loss']:.6f}, "
            log_message += f"Acc: {val_metrics['acc']:.4f}\n"
            log_message += f"Best valid loss: {best_val_loss:.6f} [patience: {num_stale_epochs}/{patience}]"
            LOGGER.info(log_message)

            # Save checkpoint
            if save_each_epoch or val_loss_improved:
                checkpoint_dict = get_checkpoint_dict(
                    model=model,
                    optimizer=optimizer,
                    part_batch_norm=part_batch_norm,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    initial_model=initial_model, 
                )

                saved_path = save_checkpoint(
                    checkpoint_dict=checkpoint_dict,
                    config=config,
                    epoch_num=epoch,
                )
                LOGGER.info(f"Checkpoint saved to {saved_path}")

        # Save best model and handle patience
        if val_loss_improved:
            best_val_loss = val_loss
            if should_save_and_log(accelerator):
                checkpoint_dict = get_checkpoint_dict(
                    model=model,
                    optimizer=optimizer,
                    part_batch_norm=part_batch_norm,
                    scheduler=scheduler,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    initial_model=initial_model,
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
                # Only unfreeze if freeze_backbone is an integer and we haven't reached the unfreeze epoch yet
                if is_backbone_frozen and isinstance(freeze_backbone, int) and epoch < freeze_backbone:
                    # Load the best checkpoint before unfreezing
                    best_checkpoint = load_best_checkpoint(config, device)
                    try:
                        model.load_state_dict(best_checkpoint["model"])
                    except RuntimeError:
                        # If model is wrapped (e.g., by DistributedDataParallel), access the module
                        model.module.load_state_dict(best_checkpoint["model"])
                    if (
                        part_batch_norm is not None
                        and best_checkpoint["part_batch_norm"] is not None
                    ):
                        try:
                            part_batch_norm.load_state_dict(
                                best_checkpoint["part_batch_norm"]
                            )
                        except RuntimeError:
                            # If part_batch_norm is wrapped, access the module
                            part_batch_norm.module.load_state_dict(
                                best_checkpoint["part_batch_norm"]
                            )

                    # Unfreeze and reset training
                    is_backbone_frozen = False
                    set_backbone_trainable(model, trainable=True)
                    if should_save_and_log(accelerator):
                        LOGGER.info("Unfreezing backbone")
                    optimizer, scheduler = setup_optimizer_scheduler()
                    if use_accelerate:
                        optimizer = accelerator.prepare(optimizer)
                    best_val_loss = float("inf")
                    num_stale_epochs = 0
                else:
                    # Stop training in all other cases:
                    # 1. freeze_backbone is True (always frozen)
                    # 2. freeze_backbone is False (never frozen, normal early stopping)
                    # 3. freeze_backbone is int but we've already unfrozen
                    if should_save_and_log(accelerator):
                        LOGGER.info(
                            f"Validation loss has not improved for {patience} epochs. "
                            f"Stopping training at epoch {epoch}."
                        )
                    break

    if should_save_and_log(accelerator):
        # Save final model
        final_checkpoint_dict = get_checkpoint_dict(
            model=model,
            optimizer=optimizer,
            part_batch_norm=part_batch_norm,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            initial_model=initial_model, 
        )
        saved_path = save_checkpoint(
            checkpoint_dict=final_checkpoint_dict,  # Use the newly created dict
            config=config,
            epoch_num="final",
        )
        LOGGER.info(f"Final model saved to {saved_path}")
    
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune classification model")
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
        name="Classification Finetuning",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    train(config=config)
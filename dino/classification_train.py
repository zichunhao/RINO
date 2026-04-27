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
from utils.ckpt import save_checkpoint, get_checkpoints_path, find_latest_checkpoint_epoch
from utils.producers import (
    get_optimizer_finetune,
    get_param_groups,
    get_models_finetune,
    get_dataloader_and_config,
    get_scheduler,
)
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device
from models import AssembledModel

from itertools import cycle

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent


# ------------------------------------------------------------------ #
# Gradient Reversal Layer (for domain adversarial training)           #
# ------------------------------------------------------------------ #
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal: identity in forward, negated gradients in backward."""
    return _GradReverse.apply(x, alpha)


def aggregate_classification_metrics(batch_metrics, accelerator=None):
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
        loss_keys = ["loss", "lwf_loss", "l2sp_loss", "l2sp_rmse", "l2sp_relative", "adv_loss", "classification_loss"]
        count_keys = ["true_count", "batch_size"]
        list_keys = ["predictions", "true_labels", "probabilities"]

        # Aggregate scalar metrics locally first
        local_aggregated = {}

        # Weighted averages for losses
        total_samples = sum(batch["batch_size"] for batch in batch_metrics)
        for key in loss_keys:
            if total_samples > 0:
                weighted_sum = sum(
                    batch.get(key, 0.0) * batch["batch_size"] for batch in batch_metrics
                )
                local_aggregated[key] = weighted_sum / total_samples
            else:
                local_aggregated[key] = 0.0

        # Sums for counts
        for key in count_keys:
            local_aggregated[key] = sum(batch.get(key, 0) for batch in batch_metrics)

        # Convert local aggregates to tensors for gathering
        for key in loss_keys + count_keys:
            tensor_val = torch.tensor(
                local_aggregated[key], device=accelerator.device, dtype=torch.float32
            )
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
            weighted_sum_tensor = torch.tensor(
                local_weighted_sum, device=accelerator.device, dtype=torch.float32
            )
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
            gathered_metrics["acc"] = (
                total_true / total_samples if total_samples > 0 else 0.0
            )

        return gathered_metrics

    else:
        # Single process: compute aggregations normally
        total_samples = sum(batch["batch_size"] for batch in batch_metrics)

        aggregated_metrics = {}

        # Weighted averages for losses
        loss_keys = ["loss", "lwf_loss", "l2sp_loss", "l2sp_rmse", "l2sp_relative", "adv_loss", "classification_loss"]
        for key in loss_keys:
            if total_samples > 0:
                weighted_sum = sum(
                    batch.get(key, 0.0) * batch["batch_size"] for batch in batch_metrics
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
            aggregated_metrics["acc"] = (
                total_true / total_samples if total_samples > 0 else 0.0
            )

        return aggregated_metrics


def should_save_and_log(accelerator):
    """Helper function to determine if this process should save checkpoints and log"""
    return accelerator is None or accelerator.is_main_process


def train_epoch(
    model: AssembledModel,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    accelerator: Accelerator | None = None,
    initial_model: AssembledModel | None = None,
    lwf_alpha: float = 0,
    l2sp_alpha: float = 0,
    label_smoothing: float = 0,
    wandb_run: Any = None,
    global_step: int = 0,
    # Domain adversarial components (all None when disabled)
    domain_disc: nn.Module | None = None,
    domain_dataloader_iter: Any = None,
    adv_alpha: float = 0,
    steps_per_epoch: int = -1,
    accumulation_steps: int = 1,
    scheduler=None,
) -> tuple[dict[str, float], int]:
    model.train()
    if domain_disc is not None:
        domain_disc.train()

    # Use lists to collect all batch metrics for proper averaging
    all_batch_metrics = []

    # For gradient accumulation: zero gradients at the start
    if accumulation_steps > 1:
        optimizer.zero_grad()

    with tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        mininterval=10,
        disable=not (accelerator is None or accelerator.is_main_process),
    ) as pbar:
        for step_idx, batch in enumerate(pbar):
            if steps_per_epoch > 0 and step_idx >= steps_per_epoch:
                break

            # With gradient accumulation, process_batch handles backward but
            # we control zero_grad and optimizer.step externally
            use_optimizer = optimizer if accumulation_steps <= 1 else None
            batch_metrics = process_batch(
                model=model,
                criterion=criterion,
                batch=batch,
                device=device,
                optimizer=use_optimizer,
                accelerator=accelerator,
                config=config,
                initial_model=initial_model,
                lwf_alpha=lwf_alpha,
                l2sp_alpha=l2sp_alpha,
                label_smoothing=label_smoothing,
                domain_disc=domain_disc,
                domain_dataloader_iter=domain_dataloader_iter,
                adv_alpha=adv_alpha,
            )

            # Batch-wise scheduler step (no-accumulation path)
            if accumulation_steps <= 1 and scheduler is not None:
                scheduler.step()

            # Gradient accumulation: manually backward + step every N batches
            if accumulation_steps > 1:
                loss = batch_metrics["_loss_tensor"]
                scaled_loss = loss / accumulation_steps
                if accelerator is not None:
                    accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()

                if (step_idx + 1) % accumulation_steps == 0:
                    grad_clip = config.get("training", {}).get("grad_clip", None) if config else None
                    if grad_clip is not None and grad_clip > 0:
                        if accelerator is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    # Step scheduler per optimizer step (batch-wise warmup)
                    if scheduler is not None:
                        scheduler.step()

            # Store batch metrics for aggregation
            all_batch_metrics.append(batch_metrics)

            # Calculate batch accuracy for progress bar
            batch_acc = batch_metrics["true_count"] / batch_metrics["batch_size"]

            # Update progress bar with batch metrics (only on main process)
            if should_save_and_log(accelerator):
                postfix = {
                    "loss": f"{batch_metrics['loss']:.4f}",
                    "cls": f"{batch_metrics['classification_loss']:.4f}",
                    "acc": f"{batch_acc:.4f}",
                }
                if batch_metrics.get("adv_loss", 0) > 0:
                    postfix["adv"] = f"{batch_metrics['adv_loss']:.4f}"
                pbar.set_postfix(postfix)

                if wandb_run is not None:
                    log_dict = {
                        "batch/loss": batch_metrics["loss"],
                        "batch/lwf_loss": batch_metrics["lwf_loss"],
                        "batch/l2sp_loss": batch_metrics["l2sp_loss"],
                        "batch/l2sp_rmse": batch_metrics["l2sp_rmse"],
                        "batch/l2sp_relative": batch_metrics["l2sp_relative"],
                        "batch/classification_loss": batch_metrics[
                            "classification_loss"
                        ],
                        "batch/acc": batch_acc,
                        "batch/lr": optimizer.param_groups[0]["lr"],
                    }
                    if batch_metrics.get("adv_loss", 0) > 0:
                        log_dict["batch/adv_loss"] = batch_metrics["adv_loss"]
                    wandb_run.log(log_dict, step=global_step)

            global_step += 1

    # Aggregate metrics across all batches and processes
    epoch_metrics = aggregate_classification_metrics(all_batch_metrics, accelerator)
    return epoch_metrics, global_step


@torch.no_grad()
def validate(
    model: AssembledModel,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    accelerator: Accelerator | None = None,
    initial_model: AssembledModel | None = None,
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
        disable=not (accelerator is None or accelerator.is_main_process),
    ) as pbar:
        for batch in pbar:
            batch_metrics = process_batch(
                model=model,
                criterion=criterion,
                batch=batch,
                device=device,
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
    model: AssembledModel,
    criterion: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    accelerator: Accelerator | None = None,
    config: dict[str, Any] | None = None,
    initial_model: AssembledModel | None = None,
    lwf_alpha: float = 0,
    l2sp_alpha: float = 0,
    label_smoothing: float = 0,
    # Domain adversarial components
    domain_disc: nn.Module | None = None,
    domain_dataloader_iter: Any = None,
    adv_alpha: float = 0,
) -> dict[str, float]:
    particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(device)
    jets = torch.tensor(batch["class_"], dtype=torch.float32).to(device)
    mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)
    labels = torch.tensor(batch["aux"]["label"], dtype=torch.long).to(device)

    # Get representations and logits from the assembled model
    rep, logits = model(particles=particles, jets=jets, mask=mask)
    lwf_loss = 0
    l2sp_loss = 0
    loss = 0
    if initial_model is not None:
        if lwf_alpha > 0:
            initial_rep = initial_model(
                particles=particles, jets=jets, mask=mask, rep_only=True
            )
            lwf_loss = (
                nn.functional.mse_loss(rep, initial_rep) / rep.shape[-1]
            )  # normalize by feature dimension
            loss += lwf_alpha * lwf_loss
            lwf_loss = lwf_loss.detach().cpu().item()
        # Always compute L2-SP for monitoring; only add to loss if alpha > 0
        with torch.no_grad() if l2sp_alpha == 0 else torch.enable_grad():
            l2sp_loss_val = 0
            init_param_norm_sq = 0
            n_params = 0
            try:
                backbone = model.backbone
            except AttributeError:
                backbone = model.module.backbone
            try:
                initial_backbone = initial_model.backbone
            except AttributeError:
                initial_backbone = initial_model.module.backbone
            # compute MSE of backbone params vs pretrained
            for param, init_param in zip(
                backbone.parameters(), initial_backbone.parameters()
            ):
                l2sp_loss_val += torch.sum((param - init_param) ** 2)
                init_param_norm_sq += torch.sum(init_param ** 2)
                n_params += param.numel()
            l2sp_loss_val = l2sp_loss_val / n_params
        if l2sp_alpha > 0:
            loss += l2sp_alpha * l2sp_loss_val
        l2sp_loss = l2sp_loss_val.detach().cpu().item()
        # Diagnostic: RMSE and relative shift (not used in training)
        l2sp_rmse = l2sp_loss_val.detach().cpu().item() ** 0.5
        init_rms = (init_param_norm_sq.detach().cpu().item() / n_params) ** 0.5
        l2sp_relative = l2sp_rmse / init_rms if init_rms > 0 else 0.0

    # Check if we're doing binary classification based on logits shape
    if logits.shape[-1] == 1:
        # For binary classification with BCEWithLogitsLoss
        # logits should be shape (batch_size, 1) or (batch_size,)
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # Remove the last dimension if it's 1

        # Convert labels to float for BCEWithLogitsLoss
        labels_float = labels.float()

        # Label smoothing: 0 → ε/2, 1 → 1-ε/2
        if label_smoothing > 0:
            labels_float = labels_float * (1 - label_smoothing) + 0.5 * label_smoothing

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

    # Domain adversarial loss
    adv_loss = 0
    if domain_disc is not None and domain_dataloader_iter is not None and adv_alpha > 0:
        # Get a domain batch from JetClass (cycles when exhausted)
        domain_batch = next(domain_dataloader_iter)
        d_particles = torch.tensor(domain_batch["sequence"], dtype=torch.float32).to(device)
        d_jets = torch.tensor(domain_batch["class_"], dtype=torch.float32).to(device)
        d_mask = torch.tensor(domain_batch["mask"], dtype=torch.bool).to(device)

        # Get representations for both domains
        # JetNet rep (from the main task forward pass) — domain label 1
        # JetClass rep — domain label 0
        with torch.no_grad() if not model.training else torch.enable_grad():
            d_rep = model(particles=d_particles, jets=d_jets, mask=d_mask, rep_only=True)

        # Concatenate and create domain labels
        all_rep = torch.cat([rep, d_rep], dim=0)
        domain_labels = torch.cat([
            torch.ones(rep.size(0), device=device),   # JetNet = 1
            torch.zeros(d_rep.size(0), device=device), # JetClass = 0
        ])

        # Gradient reversal: backbone learns to confuse the discriminator
        reversed_rep = grad_reverse(all_rep, alpha=adv_alpha)
        domain_logits = domain_disc(reversed_rep).squeeze(-1)
        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            domain_logits, domain_labels
        )
        loss = loss + adv_loss
        adv_loss = adv_loss.detach().cpu().item()

    # Calculate metrics
    true_count = (preds == labels).sum().item()  # Count of correct predictions
    batch_size = labels.size(0)  # Batch size for proper averaging

    # Create detailed metrics dictionary
    metrics = {
        "loss": loss.item(),
        "lwf_loss": lwf_loss,
        "l2sp_loss": l2sp_loss,
        "l2sp_rmse": l2sp_rmse if initial_model is not None else 0.0,
        "l2sp_relative": l2sp_relative if initial_model is not None else 0.0,
        "adv_loss": adv_loss,
        "classification_loss": classification_loss,
        "true_count": true_count,
        "batch_size": batch_size,
        "predictions": preds.cpu().tolist(),
        "true_labels": labels.cpu().tolist(),
        "probabilities": probs_expanded.detach().cpu().tolist(),
    }

    if optimizer is not None:
        from utils.producers.optimizer_finetune import SAM

        optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Handle gradient clipping
        grad_clip = (
            config.get("training", {}).get("grad_clip", None) if config else None
        )
        if grad_clip is not None and grad_clip > 0:
            if accelerator is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

        if isinstance(optimizer, SAM):
            # SAM two-step: ascend with current gradient, recompute at perturbed point
            optimizer.first_step()
            # Second forward-backward at perturbed weights
            optimizer.zero_grad()
            rep2, logits2 = model(particles=particles, jets=jets, mask=mask)
            loss2 = criterion(logits2.squeeze(), labels.float())
            if label_smoothing > 0:
                loss2 = loss2 * (1 - label_smoothing) + 0.5 * label_smoothing
            if accelerator is not None:
                accelerator.backward(loss2)
            else:
                loss2.backward()
            if grad_clip is not None and grad_clip > 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.second_step()
        else:
            optimizer.step()
    else:
        # When optimizer is None (gradient accumulation mode),
        # keep loss tensor for external backward
        metrics["_loss_tensor"] = loss

    return metrics


def get_state_dict(model: nn.Module):
    if hasattr(model, "module"):
        # a feature of nn.DataParallel
        return model.module.state_dict()
    else:
        return model.state_dict()


def get_checkpoint_dict(
    model: AssembledModel,
    optimizer: optim.Optimizer,
    scheduler: _LRScheduler | ReduceLROnPlateau | None,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    initial_model: AssembledModel | None = None,
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

    if initial_model is not None:
        checkpoint_dict["initial_model"] = get_state_dict(initial_model)
    else:
        checkpoint_dict["initial_model"] = None

    return checkpoint_dict


def set_backbone_trainable(model: AssembledModel, trainable: bool = True) -> None:
    """Set the backbone to be trainable or frozen."""
    if trainable:
        try:
            model.train_backbone()  # Use the method from AssembledModel
        except AttributeError:
            # Model is wrapped (e.g., by DistributedDataParallel)
            model.module.train_backbone()
    else:
        try:
            model.freeze_backbone()  # Use the method from AssembledModel
        except AttributeError:
            # Model is wrapped (e.g., by DistributedDataParallel)
            model.module.freeze_backbone()


def get_training_start_params(
    config: dict[str, Any], device: torch.device, accelerator: Accelerator | None
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
        LOGGER.info(
            f"Resuming from epoch {start_epoch} with best val loss {best_val_loss}"
        )

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
def train(config: dict[str, Any], use_wandb: bool = False) -> None:

    # Auto-resume logic:
    # - Skip if final checkpoint exists (training fully completed)
    # - Resume from last epoch checkpoint if one exists but final doesn't
    auto_resume = config["training"].get("auto_resume", True)
    if auto_resume:
        final_ckpt_path = get_checkpoints_path(config, "final")
        if final_ckpt_path.exists():
            LOGGER.info(
                f"Skipping training: final checkpoint already exists at {final_ckpt_path} "
                f"(auto_resume=True)"
            )
            return
        # Resume from last epoch checkpoint if one exists
        if config["training"].get("load_epoch") is None:
            latest = find_latest_checkpoint_epoch(config)
            if latest is not None:
                LOGGER.info(
                    f"Auto-resuming from epoch {latest} (auto_resume=True)"
                )
                config["training"]["load_epoch"] = "LAST"

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
            dispatch_batches=True,  # Main process loads batches
            split_batches=True,  # Split fetched batches across processes
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

    # W&B
    wandb_run = None
    if use_wandb and should_save_and_log(accelerator):
        try:
            import wandb  # noqa: PLC0415

            wandb_id = None
            wandb_id_file = get_checkpoints_path(config, 0).parent / "wandb_run_id.txt"
            if (
                config["training"].get("load_epoch") is not None
                and wandb_id_file.exists()
            ):
                wandb_id = wandb_id_file.read_text().strip()
                LOGGER.info(f"Resuming W&B run {wandb_id}")

            wandb_run = wandb.init(
                project="PARCEL-Classification",
                name=config.get("name", "classification"),
                config=config,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
            wandb_id_file.write_text(wandb_run.id)
            LOGGER.info("W&B run initialized")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    # Inject train_fraction_seed for label efficiency runs so each run_index
    # gets a different random subset of the training data.
    run_index = config.get("training", {}).get("run_index")
    if run_index is not None:
        train_kwargs = config.get("training", {}).get("dataloader", {}).get("train", {}).get("kwargs", {})
        if train_kwargs.get("train_fraction", 1.0) < 1.0:
            train_kwargs["train_fraction_seed"] = 42 + run_index
            LOGGER.info(f"Label efficiency: train_fraction_seed={42 + run_index} (run_index={run_index})")

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
    model = get_models_finetune(
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
    label_smoothing = config["training"].get("label_smoothing", 0)

    # Always initialize reference model for L2-SP/LwF monitoring
    # (even when alphas=0, we log the metric for diagnostic purposes)
    if should_save_and_log(accelerator):
        LOGGER.info(f"Using alpha_lwf={lwf_alpha} and alpha_l2sp={l2sp_alpha}")
        LOGGER.info("Initializing initial model for L2SP/LwF monitoring")

    # Create a new model instance with the same architecture
    # Cannot deepcopy because of weight norm
    initial_model = get_models_finetune(
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

    # Loss function
    output_dim = config["models"]["head"]["params"]["output_dim"]
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
                    LOGGER.info(
                        f"Using BCEWithLogitsLoss for binary classification with pos_weight={bce_pos_weight}"
                    )
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
    start_epoch, best_val_loss = get_training_start_params(
        config=config, device=device, accelerator=accelerator
    )

    # Training parameters
    training_params = config["training"]
    num_epochs: int = training_params["num_epochs"]
    steps_per_epoch: int = training_params.get("steps_per_epoch", -1)
    patience = training_params.get("patience", float("inf"))
    freeze_backbone = training_params.get("freeze_backbone", None)
    freeze_embedding = training_params.get("freeze_embedding", False)

    # Zero-init head output layer (LP-FT companion: σ(0)=0.5 → symmetric first gradient)
    zero_init_head = training_params.get("zero_init_head", False)
    if zero_init_head:
        head = model.heads if hasattr(model, 'heads') else (model.module.heads if hasattr(model, 'module') else None)
        if head is not None and hasattr(head, 'last_layer'):
            LOGGER.info("Zero-initializing head output layer only")
            for p in head.last_layer.parameters():
                torch.nn.init.zeros_(p.data)
        elif head is not None:
            # Fallback: zero the last Linear layer found
            last_linear = None
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    last_linear = m
            if last_linear is not None:
                LOGGER.info("Zero-initializing last Linear in head")
                torch.nn.init.zeros_(last_linear.weight.data)
                if last_linear.bias is not None:
                    torch.nn.init.zeros_(last_linear.bias.data)

    # Small-init head: scale output layer weights by a small factor
    small_init_head = training_params.get("small_init_head", None)
    if small_init_head is not None and small_init_head > 0:
        head = model.heads if hasattr(model, 'heads') else (model.module.heads if hasattr(model, 'module') else None)
        if head is not None:
            last_linear = None
            for m in head.modules():
                if isinstance(m, torch.nn.Linear):
                    last_linear = m
            if last_linear is not None:
                LOGGER.info(f"Small-init head: scaling last Linear weights by {small_init_head}")
                last_linear.weight.data *= small_init_head
                if last_linear.bias is not None:
                    last_linear.bias.data *= small_init_head

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
            LOGGER.info(
                f"Starting training with frozen backbone (unfreeze at epoch {freeze_epoch})"
            )
        set_backbone_trainable(model, trainable=False)
    else:  # None or any other value means no freezing
        if should_save_and_log(accelerator):  # Add this conditional
            LOGGER.info("Starting training with trainable backbone")
        set_backbone_trainable(model, trainable=True)

    # Batch-wise warmup: step scheduler per optimizer step instead of per epoch
    batch_wise_scheduler = training_params.get("batch_wise_warmup", False)
    accum_steps = training_params.get("accumulation_steps", 1)
    _steps_per_epoch = len(train_dataloader) // accum_steps  # optimizer steps per epoch

    def setup_optimizer_scheduler():
        try:
            param_groups = get_param_groups(model.backbone, model.heads, config)
        except AttributeError:
            # If model is wrapped (e.g., by DistributedDataParallel), access the module
            param_groups = get_param_groups(
                model.module.backbone, model.module.heads, config
            )
        optimizer = get_optimizer_finetune(config, param_groups)
        if batch_wise_scheduler:
            LOGGER.info(f"Batch-wise warmup: {_steps_per_epoch} optimizer steps/epoch (accum={accum_steps})")
            scheduler = get_scheduler(
                optimizer=optimizer, config=config, steps_per_epoch=_steps_per_epoch
            )
        else:
            scheduler = get_scheduler(optimizer=optimizer, config=config)
        return optimizer, scheduler

    # Initial optimizer and scheduler setup
    optimizer, scheduler = setup_optimizer_scheduler()

    # ------------------------------------------------------------------ #
    # Domain adversarial training setup                                  #
    # ------------------------------------------------------------------ #
    adv_config = config["training"].get("adversarial", {})
    adv_alpha = adv_config.get("alpha", 0)
    domain_disc = None
    domain_dataloader_iter = None

    if adv_alpha > 0:
        # Build domain discriminator: MLP on backbone representations
        d_model = config["models"]["backbone"]["params"]["d_model"]
        disc_hidden = adv_config.get("hidden_dims", [256])
        disc_layers = []
        in_dim = d_model
        for h in disc_hidden:
            disc_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = h
        disc_layers.append(nn.Linear(in_dim, 1))
        domain_disc = nn.Sequential(*disc_layers)
        if not use_accelerate:
            domain_disc = domain_disc.to(device)

        # Add domain_disc params to optimizer
        base_lr = config["training"]["optimizer"]["params"].get("lr", 1e-4)
        optimizer.add_param_group({
            "params": list(domain_disc.parameters()),
            "lr": base_lr,
            "name": "domain_disc",
        })

        # Load JetClass domain dataloader (cycles when exhausted)
        domain_dataloader, _ = get_dataloader_and_config(
            config=config,
            split="domain",
            mode="training",
            accelerator=accelerator,
        )

        if should_save_and_log(accelerator):
            LOGGER.info(
                f"Domain adversarial training enabled: alpha={adv_alpha}, "
                f"disc_hidden={disc_hidden}, domain batches={len(domain_dataloader)}"
            )

    if use_accelerate:
        prepare_args = [model, optimizer, train_dataloader, val_dataloader]
        if domain_disc is not None:
            prepare_args.append(domain_disc)
        prepared = accelerator.prepare(*prepare_args)
        if domain_disc is not None:
            model, optimizer, train_dataloader, val_dataloader, domain_disc = prepared
        else:
            model, optimizer, train_dataloader, val_dataloader = prepared
        if initial_model is not None:
            initial_model = accelerator.prepare(initial_model)

    # Create cycling iterator for domain dataloader (after accelerator.prepare)
    if adv_alpha > 0:
        domain_dataloader_iter = iter(cycle(domain_dataloader))

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
            # L2-SP monitoring: always use pretrained weights as reference,
            # NOT a checkpoint from a previous finetuning epoch.
            # Only load initial_model from checkpoint for LwF (which needs
            # the exact training-start state).
            if lwf_alpha > 0:
                if "initial_model" in checkpoint:
                    initial_model.load_state_dict(checkpoint["initial_model"])
                else:
                    LOGGER.warning("Disabling LwF — no initial model state saved")
                    lwf_alpha = 0
            # For L2-SP monitoring, keep the freshly-initialized pretrained weights
            LOGGER.info(
                "L2-SP reference: using pretrained backbone weights (not checkpoint state)"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.best = checkpoint["scheduler"]["best"]
                scheduler.num_bad_epochs = checkpoint["scheduler"]["num_bad_epochs"]
            else:
                scheduler.load_state_dict(checkpoint["scheduler"])

        if should_save_and_log(accelerator):
            LOGGER.info(f"Resumed training from checkpoint {checkpoint_path}")

    # Training loop
    num_stale_epochs = 0
    global_step = start_epoch * len(train_dataloader)
    if wandb_run is not None and config["training"].get("load_epoch") is not None:
        wandb_step = getattr(wandb_run, "step", 0) or 0
        if wandb_step > global_step:
            LOGGER.info(
                f"Adjusting global_step from {global_step} to {wandb_step} "
                "to match resumed W&B run (avoids non-monotonic step warnings)"
            )
            global_step = wandb_step
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

        train_metrics, global_step = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            config=config,
            accelerator=accelerator,
            initial_model=initial_model,
            lwf_alpha=lwf_alpha,
            l2sp_alpha=l2sp_alpha,
            label_smoothing=label_smoothing,
            wandb_run=wandb_run,
            global_step=global_step,
            domain_disc=domain_disc,
            domain_dataloader_iter=domain_dataloader_iter,
            adv_alpha=adv_alpha,
            steps_per_epoch=steps_per_epoch,
            accumulation_steps=training_params.get("accumulation_steps", 1),
            scheduler=scheduler if batch_wise_scheduler else None,
        )

        val_metrics = validate(
            model=model,  # Pass unified model
            criterion=criterion,
            dataloader=val_dataloader,
            device=device,
            config=config,
            accelerator=accelerator,
            initial_model=initial_model,
            lwf_alpha=lwf_alpha,
            l2sp_alpha=l2sp_alpha,
        )

        val_loss = val_metrics["loss"]
        val_loss_improved = val_loss < best_val_loss
        if scheduler is not None and not batch_wise_scheduler:
            # Epoch-level scheduler step (skip if batch-wise — already stepped in train_epoch)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if should_save_and_log(accelerator):
            log_message = f"Epoch [{epoch}/{start_epoch + num_epochs}]\n"
            log_message += f"Train loss: {train_metrics['loss']:.6f}, "
            log_message += f"LwF loss: {train_metrics['lwf_loss']:.6f}, "
            log_message += f"L2SP loss: {train_metrics['l2sp_loss']:.6f} (RMSE: {train_metrics['l2sp_rmse']:.6f}, rel: {train_metrics['l2sp_relative']:.4f}), "
            log_message += (
                f"Classification loss: {train_metrics['classification_loss']:.6f}, "
            )
            log_message += f"Acc: {train_metrics['acc']:.4f}\n"
            log_message += f"Valid loss: {val_metrics['loss']:.6f}, "
            log_message += f"LwF loss: {val_metrics['lwf_loss']:.6f}, "
            log_message += f"L2SP loss: {val_metrics['l2sp_loss']:.6f} (RMSE: {val_metrics['l2sp_rmse']:.6f}, rel: {val_metrics['l2sp_relative']:.4f}), "
            log_message += (
                f"Classification loss: {val_metrics['classification_loss']:.6f}, "
            )
            log_message += f"Acc: {val_metrics['acc']:.4f}\n"
            log_message += f"Best valid loss: {best_val_loss:.6f} [patience: {num_stale_epochs}/{patience}]"
            LOGGER.info(log_message)

        if wandb_run is not None:
            wandb_metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            wandb_metrics.update(
                {
                    f"train/{k}": v
                    for k, v in train_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
            wandb_metrics.update(
                {
                    f"val/{k}": v
                    for k, v in val_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
            wandb_run.log(wandb_metrics, step=global_step)

        if should_save_and_log(accelerator):
            # Save checkpoint
            if save_each_epoch or val_loss_improved:
                checkpoint_dict = get_checkpoint_dict(
                    model=model,
                    optimizer=optimizer,
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
        # Determine warmup milestone from scheduler config
        warmup_milestone = 0
        sched_params = training_params.get("scheduler", {}).get("params", {})
        milestones = sched_params.get("milestones", [])
        if milestones:
            warmup_milestone = milestones[0]
        # Also account for LP-FT freeze epochs
        if isinstance(freeze_backbone, int):
            warmup_milestone = max(warmup_milestone, freeze_backbone)

        if val_loss_improved:
            best_val_loss = val_loss
            if should_save_and_log(accelerator):
                checkpoint_dict = get_checkpoint_dict(
                    model=model,
                    optimizer=optimizer,
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
        elif epoch < warmup_milestone:
            # Don't count patience during warmup / frozen epochs
            pass
        else:
            num_stale_epochs += 1
            if num_stale_epochs >= patience:
                # Only unfreeze if freeze_backbone is an integer and we haven't reached the unfreeze epoch yet
                if (
                    is_backbone_frozen
                    and isinstance(freeze_backbone, int)
                    and epoch < freeze_backbone
                ):
                    # Load the best checkpoint before unfreezing
                    best_checkpoint = load_best_checkpoint(config, device)
                    try:
                        model.load_state_dict(best_checkpoint["model"])
                    except RuntimeError:
                        # If model is wrapped (e.g., by DistributedDataParallel), access the module
                        model.module.load_state_dict(best_checkpoint["model"])

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

    if wandb_run is not None:
        wandb_run.finish()


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
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Run index for parallel runs. When set, checkpoints are saved "
        "under a run-{N}/ subdirectory (e.g. experiments/.../run-1/checkpoints/).",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Insert run subdirectory into checkpoint paths for parallel indexed jobs
    if args.run_index is not None:
        config.setdefault("training", {})["run_index"] = args.run_index
        run_subdir = f"run-{args.run_index}"
        ckpt_dir = config["training"]["checkpoints_dir"]
        # Insert run-N/ before the last path component (typically "checkpoints")
        ckpt_path = Path(ckpt_dir)
        config["training"]["checkpoints_dir"] = str(
            ckpt_path.parent / run_subdir / ckpt_path.name
        )
        if "inference" in config and "output_dir" in config["inference"]:
            out_dir = config["inference"]["output_dir"]
            out_path = Path(out_dir)
            config["inference"]["output_dir"] = str(
                out_path.parent / run_subdir / out_path.name
            )

        # Deterministic seeding per run_index so parallel runs are reproducible
        # and can be debugged / pruned. seed = 42 + run_index.
        import random as _random
        import numpy as _np
        seed = 42 + args.run_index
        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        LOGGER.info(f"Deterministic seed set to {seed} (42 + run_index={args.run_index})")

    configure_logger(
        logger=LOGGER,
        name="Classification Finetuning",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    train(config=config, use_wandb=args.use_wandb)

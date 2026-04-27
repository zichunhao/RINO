import math
import sys
from pathlib import Path
from typing import Any
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from modules.jet_augs import (
    rotate_jets,
    distort_jets,
    rescale_pts,
    crop_jets,
    translate_jets,
    collinear_fill_jets,
    rino_post_normalize,
)
from modules.losses import contrastive_loss, align_loss, uniform_loss
from modules.perf_eval import get_perf_stats, linear_classifier_test

baseline_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
project_dir = baseline_dir.parent
# add baseline_dir to sys.path for imports
sys.path.append(str(baseline_dir))

from utils.logger import LOGGER, configure_logger
from utils.producers import (
    get_dataloader_and_config,
    get_optimizer,
    get_scheduler,
    get_model,
)

torch.set_num_threads(2)


def should_save_and_log(accelerator: Accelerator | None) -> bool:
    """Helper function to determine if this process should save checkpoints and log"""
    return accelerator is None or accelerator.is_main_process


def augment_batch(
    x: torch.Tensor,
    config: dict[str, Any],
    mask: torch.Tensor | None = None,
    jet_pt: torch.Tensor | None = None,
    jet_energy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentations to a batch of jets.

    Input: [B, n_constituents, D] - already on correct device.
    When feature_map is absent (legacy mode), D=3 with (pT, deta, dphi).
    When feature_map is present (RINO mode), D=4 with (pT, energy, deta, dphi).

    Returns two augmented views. In RINO mode the output is (B, N, 7) normalized
    features; in legacy mode it is (B, N, 3) rescaled features.
    """
    feature_map = config.get("feature_map", None)

    if feature_map is not None:
        pt_idx = feature_map["pt"]
        eta_idx = feature_map["eta"]
        phi_idx = feature_map["phi"]
        energy_idx = feature_map.get("energy")
        split_idxs = [energy_idx] if energy_idx is not None else []
    else:
        pt_idx, eta_idx, phi_idx = 0, 1, 2
        split_idxs = []

    # First view - always rotated
    x_i = rotate_jets(x, pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

    # Second view - copy and apply augmentations
    x_j = x_i.clone()

    if config.get("rot", False):
        x_j = rotate_jets(x_j, pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

    if config.get("cf", False):
        x_j = collinear_fill_jets(x_j, pt_idx=pt_idx, split_idxs=split_idxs)
        x_j = collinear_fill_jets(x_j, pt_idx=pt_idx, split_idxs=split_idxs)

    if config.get("ptd", False):
        x_j = distort_jets(
            x_j,
            strength=config.get("ptst", 0.1),
            pT_clip_min=config.get("ptcm", 0.1),
            pt_idx=pt_idx,
            eta_idx=eta_idx,
            phi_idx=phi_idx,
        )

    if config.get("trs", False):
        width = config.get("trsw", 1.0)
        x_j = translate_jets(x_j, width=width, pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)
        x_i = translate_jets(x_i, width=width, pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

    # Post-augmentation normalization
    post_normalize = config.get("post_normalize", None)
    if post_normalize == "rino":
        assert mask is not None, "mask required for RINO post-normalization"
        assert jet_pt is not None, "jet_pt required for RINO post-normalization"
        assert jet_energy is not None, "jet_energy required for RINO post-normalization"
        x_i = rino_post_normalize(
            x_i, jet_pt, jet_energy, mask,
            pt_idx=pt_idx, energy_idx=energy_idx, eta_idx=eta_idx, phi_idx=phi_idx,
        )
        x_j = rino_post_normalize(
            x_j, jet_pt, jet_energy, mask,
            pt_idx=pt_idx, energy_idx=energy_idx, eta_idx=eta_idx, phi_idx=phi_idx,
        )
    else:
        # Legacy: rescale pT
        x_i = rescale_pts(x_i, pt_idx=pt_idx)
        x_j = rescale_pts(x_j, pt_idx=pt_idx)

    return x_i, x_j


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    scheduler: Any = None,
    accelerator: Accelerator | None = None,
    wandb_run: Any = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    """Train for one epoch"""
    model.train()

    # Accumulate losses as GPU tensors; only sync to CPU once at the end.
    loss_accum = torch.zeros(1, device=device)
    align_accum = torch.zeros(1, device=device)
    uniform_accum = torch.zeros(1, device=device)
    n_batches = 0
    n_nan = 0

    compute_align_uniform = epoch % 10 == 0
    use_sgdca = config["training"]["optimizer"] == "sgdca"
    detect_anomaly = config["training"].get("autograd_detect_anomaly", False)

    aug_config = config["augmentation"]
    temp = config["training"]["temperature"]
    use_rino = aug_config.get("post_normalize") == "rino"

    with tqdm(
        dataloader,
        desc=f"Training Epoch {epoch}",
        unit="batch",
        disable=not should_save_and_log(accelerator),
    ) as pbar:
        for i, batch in enumerate(pbar):

            with torch.autograd.set_detect_anomaly(detect_anomaly):
                optimizer.zero_grad()

                x = batch["sequence"].to(device, dtype=torch.float32, non_blocking=True)
                mask = batch["mask"].to(device, dtype=torch.bool, non_blocking=True)

                max_len = int(mask.sum(dim=1).max().item())
                x = x[:, :max_len]
                mask = mask[:, :max_len]

                # Extract jet-level quantities for RINO post-normalization
                jet_pt = jet_energy = None
                if use_rino and "class_" in batch:
                    cls = batch["class_"].to(device, dtype=torch.float32, non_blocking=True)
                    jet_pt = cls[:, 0]
                    jet_energy = cls[:, 1]

                if epoch == 0 and i == 0 and should_save_and_log(accelerator):
                    LOGGER.debug(f"Input shape: {x.shape}")
                    LOGGER.debug(
                        f"Input stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}"
                    )

                x_i, x_j = augment_batch(
                    x, aug_config, mask=mask, jet_pt=jet_pt, jet_energy=jet_energy,
                )

                if epoch == 0 and i == 0 and should_save_and_log(accelerator):
                    LOGGER.debug(
                        f"Aug x_i stats - min: {x_i.min():.4f}, max: {x_i.max():.4f}, mean: {x_i.mean():.4f}"
                    )
                    LOGGER.debug(
                        f"Aug x_j stats - min: {x_j.min():.4f}, max: {x_j.max():.4f}, mean: {x_j.mean():.4f}"
                    )

                _, z_i = model(x_i, mask=mask)
                _, z_j = model(x_j, mask=mask)

                if epoch == 0 and i == 0 and should_save_and_log(accelerator):
                    LOGGER.debug(f"z_i shape: {z_i.shape}, z_j shape: {z_j.shape}")
                    LOGGER.debug(f"z_i norm: {z_i.norm(dim=1).mean():.4f}")

                loss = contrastive_loss(z_i, z_j, temp)

                # NaN check — only forces a sync when there actually is a NaN,
                # which should be rare; avoids stalling every step.
                if torch.isnan(loss) or torch.isinf(loss):
                    LOGGER.error(f"NaN/Inf loss at epoch {epoch}, batch {i}")
                    LOGGER.error(
                        f"z_i stats: min={z_i.min():.4f}, max={z_i.max():.4f}, has_nan={torch.isnan(z_i).any()}"
                    )
                    LOGGER.error(
                        f"z_j stats: min={z_j.min():.4f}, max={z_j.max():.4f}, has_nan={torch.isnan(z_j).any()}"
                    )
                    n_nan += 1
                    continue

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                optimizer.step()

                if use_sgdca and scheduler is not None:
                    iters = len(dataloader)
                    scheduler.step(epoch + i / iters)
                elif scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()

            # Accumulate on GPU — no CPU sync here.
            loss_accum += loss.detach()
            n_batches += 1

            if compute_align_uniform:
                with torch.no_grad():
                    align = align_loss(z_i, z_j)
                    uniform = (uniform_loss(z_i) + uniform_loss(z_j)) / 2
                    align_accum += align
                    uniform_accum += uniform

            # Progress bar update: sync every 20 steps to amortise cost.
            if should_save_and_log(accelerator) and i % 20 == 0:
                pbar.set_postfix(
                    {"loss": f"{(loss_accum / max(n_batches, 1)).item():.4f}"}
                )

            if wandb_run is not None and should_save_and_log(accelerator):
                batch_wandb = {
                    "batch/loss": loss.detach().item(),
                    "batch/lr": optimizer.param_groups[0]["lr"],
                }
                if compute_align_uniform:
                    batch_wandb["batch/align_loss"] = align.item()
                    batch_wandb["batch/uniform_loss"] = uniform.item()
                wandb_run.log(batch_wandb, step=global_step)

            global_step += 1

    if n_batches == 0:
        return {"loss": float("nan")}, global_step

    # Single GPU→CPU sync per epoch.
    metrics = {"loss": (loss_accum / n_batches).item()}
    if compute_align_uniform:
        metrics["align_loss"] = (align_accum / n_batches).item()
        metrics["uniform_loss"] = (uniform_accum / n_batches).item()

    return metrics, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    """Validate the model"""
    model.eval()

    loss_accum = torch.zeros(1, device=device)
    n_batches = 0
    temp = config["training"]["temperature"]
    aug_config = config["augmentation"]
    use_rino = aug_config.get("post_normalize") == "rino"

    with tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        disable=not should_save_and_log(accelerator),
    ) as pbar:
        for i, batch in enumerate(pbar):
            x = batch["sequence"].to(device, dtype=torch.float32, non_blocking=True)
            mask = batch["mask"].to(device, dtype=torch.bool, non_blocking=True)

            max_len = int(mask.sum(dim=1).max().item())
            x = x[:, :max_len]
            mask = mask[:, :max_len]

            jet_pt = jet_energy = None
            if use_rino and "class_" in batch:
                cls = batch["class_"].to(device, dtype=torch.float32, non_blocking=True)
                jet_pt = cls[:, 0]
                jet_energy = cls[:, 1]

            x_i, x_j = augment_batch(
                x, aug_config, mask=mask, jet_pt=jet_pt, jet_energy=jet_energy,
            )

            _, z_i = model(x_i, mask=mask)
            _, z_j = model(x_j, mask=mask)

            loss = contrastive_loss(z_i, z_j, temp)

            if torch.isnan(loss) or torch.isinf(loss):
                if should_save_and_log(accelerator) and i < 5:
                    LOGGER.error(f"NaN loss at val batch {i}")
                continue

            loss_accum += loss
            n_batches += 1

    if n_batches == 0:
        LOGGER.error("All validation batches had NaN!")
        return {"loss": float("nan")}

    LOGGER.info(f"Valid batches: {n_batches}/{len(dataloader)}")
    return {"loss": (loss_accum / n_batches).item()}


@torch.no_grad()
def run_linear_classifier_test(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    val_dataloader_2: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
    accelerator: Accelerator | None = None,
) -> tuple[float, float]:
    """Run linear classifier test to evaluate representations"""
    model.eval()

    aug_config = config["augmentation"]
    use_rino = aug_config.get("post_normalize") == "rino"
    feature_map = aug_config.get("feature_map", None)

    if feature_map is not None:
        pt_idx = feature_map["pt"]
        eta_idx = feature_map["eta"]
        phi_idx = feature_map["phi"]
        energy_idx = feature_map.get("energy")
    else:
        pt_idx, eta_idx, phi_idx = 0, 1, 2
        energy_idx = None

    # Get representations for both validation sets
    reps_1 = []
    labels_1 = []

    for batch in tqdm(
        val_dataloader, desc="LCT Set 1", disable=not should_save_and_log(accelerator)
    ):
        x = batch["sequence"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        label = batch.get("label")
        if label is None:
            label = batch["aux"]["label"]
        mask = batch["mask"].to(device, dtype=torch.bool, non_blocking=True)

        if aug_config.get("trs", False):
            x = translate_jets(x, width=aug_config.get("trsw", 1.0), pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

        if use_rino:
            cls = batch["class_"].to(device, dtype=torch.float32, non_blocking=True)
            x = rino_post_normalize(
                x, cls[:, 0], cls[:, 1], mask,
                pt_idx=pt_idx, energy_idx=energy_idx, eta_idx=eta_idx, phi_idx=phi_idx,
            )
        else:
            x = rescale_pts(x, pt_idx=pt_idx)

        rep, _ = model(x, mask=mask)
        reps_1.append(rep.cpu())
        labels_1.append(
            torch.tensor(label) if not isinstance(label, torch.Tensor) else label.cpu()
        )

    reps_1 = torch.cat(reps_1, dim=0).numpy()
    labels_1 = torch.cat(labels_1, dim=0).numpy()

    # Repeat for second validation set
    reps_2 = []
    labels_2 = []

    for batch in tqdm(
        val_dataloader_2, desc="LCT Set 2", disable=not should_save_and_log(accelerator)
    ):
        x = batch["sequence"].to(device, dtype=torch.float32, non_blocking=True)
        label = batch.get("label")
        if label is None:
            label = batch["aux"]["label"]
        mask = batch["mask"].to(device, dtype=torch.bool, non_blocking=True)

        if aug_config.get("trs", False):
            x = translate_jets(x, width=aug_config.get("trsw", 1.0), pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

        if use_rino:
            cls = batch["class_"].to(device, dtype=torch.float32, non_blocking=True)
            x = rino_post_normalize(
                x, cls[:, 0], cls[:, 1], mask,
                pt_idx=pt_idx, energy_idx=energy_idx, eta_idx=eta_idx, phi_idx=phi_idx,
            )
        else:
            x = rescale_pts(x, pt_idx=pt_idx)

        # Forward pass - no transpose!
        rep, _ = model(x, mask=mask)
        reps_2.append(rep.cpu())
        labels_2.append(
            torch.tensor(label) if not isinstance(label, torch.Tensor) else label.cpu()
        )

    reps_2 = torch.cat(reps_2, dim=0).numpy()
    labels_2 = torch.cat(labels_2, dim=0).numpy()

    # Train linear classifier
    lct_config = config["linear_classifier_test"]

    out_dat, out_lbs, losses = linear_classifier_test(
        reps_1.shape[1],
        lct_config["batch_size"],
        lct_config["n_epochs"],
        lct_config["optimizer"],
        lct_config["learning_rate"],
        reps_1,
        labels_1,
        reps_2,
        labels_2,
    )

    auc, imtafe = get_perf_stats(out_lbs, out_dat)
    return auc, imtafe


def get_state_dict(model: nn.Module) -> dict[str, Any]:
    """Get state dict, handling DataParallel/DistributedDataParallel and AssembledModel"""
    # First unwrap DDP/DataParallel if needed
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    # Check if it's an AssembledModel
    from models import AssembledModel

    if isinstance(model, AssembledModel):
        # Save backbone and heads separately
        state_dict = {
            "backbone": model.backbone.state_dict(),
            "head": (
                model.heads.state_dict() if hasattr(model.heads, "state_dict") else None
            ),
        }
        # Also save embedding if it's not Identity
        if not isinstance(model.embedding, nn.Identity):
            state_dict["embedding"] = model.embedding.state_dict()
        return state_dict
    else:
        # Regular model
        return model.state_dict()


def train(config: dict[str, Any], use_wandb: bool = False) -> None:
    """Main training function"""

    use_accelerate = config.get("accelerate", False)

    # Initialize accelerator
    if use_accelerate:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,
            split_batches=True,
        )
        kwargs = InitProcessGroupKwargs(timeout=timedelta(days=365))
        # bf16-mixed precision: ~2× speedup on Ampere (A10/A5000/etc) and
        # halves activation memory.
        accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[kwargs],
            mixed_precision="bf16",
        )
        device = accelerator.device
        LOGGER.info(f"Device: {device}")

        if accelerator.is_main_process:
            LOGGER.info("Using accelerate for distributed training")

        LOGGER.addFilter(lambda record: accelerator.is_main_process)
    else:
        accelerator = None
        device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        LOGGER.info(f"Device: {device}")

    if should_save_and_log(accelerator):
        LOGGER.info(f"Training with config: {config}")

    # W&B
    wandb_run = None
    if use_wandb and should_save_and_log(accelerator):
        try:
            import wandb  # noqa: PLC0415

            _job_name = config.get("name", "jetclr")
            _expt_dir = Path(
                config["experiment_dir"]
                .replace("PROJECT_ROOT", str(project_dir))
                .replace("JOBNAME", _job_name)
            )
            wandb_id = None
            wandb_id_file = _expt_dir / "wandb_run_id.txt"
            if config["training"].get("load_epoch", 0) > 0 and wandb_id_file.exists():
                wandb_id = wandb_id_file.read_text().strip()
                LOGGER.info(f"Resuming W&B run {wandb_id}")

            wandb_run = wandb.init(
                project="RINO-JetCLR-train",
                name=_job_name,
                config=config,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            _expt_dir.mkdir(parents=True, exist_ok=True)
            wandb_id_file.write_text(wandb_run.id)
            LOGGER.info("W&B run initialized")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    # Get dataloaders
    train_dataloader, dataloader_config_train = get_dataloader_and_config(
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
    # RINO post-normalization produces 7 features; legacy produces 3
    _aug_cfg = config.get("augmentation", {})
    if _aug_cfg.get("post_normalize") == "rino":
        _part_dim = 7
    else:
        _part_dim = 3
    model = get_model(
        part_dim=_part_dim,
        config=config,
        device=device,
        mode="training",
        assemble=True,
    )
    LOGGER.info(f"Model: {model}")

    if not use_accelerate:
        model = model.to(device)

    optimizer, _ = get_optimizer(
        config=config,
        model_params=model.parameters(),
    )

    steps_per_epoch = len(train_dataloader)
    LOGGER.info(f"steps_per_epoch (pre-prepare) = {steps_per_epoch}")

    scheduler = get_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch,
    )

    # Prepare with accelerator
    if use_accelerate:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )

    # Correct steps_per_epoch from the prepared dataloader.
    # With dispatch_batches=True the length is unchanged (every process
    # sees all batches).  With standard DDP it shrinks by num_processes.
    actual_steps_per_epoch = len(train_dataloader)
    if actual_steps_per_epoch != steps_per_epoch:
        LOGGER.info(
            f"steps_per_epoch corrected after prepare: "
            f"{steps_per_epoch} → {actual_steps_per_epoch} (standard DDP sharding)"
        )
        scheduler = get_scheduler(
            optimizer=optimizer, config=config, steps_per_epoch=actual_steps_per_epoch
        )
        steps_per_epoch = actual_steps_per_epoch
    LOGGER.info(f"steps_per_epoch = {steps_per_epoch}")

    # Training loop
    training_params = config["training"]
    num_epochs = training_params["num_epochs"]
    start_epoch = training_params.get("load_epoch", 0)

    best_val_loss = float("inf")

    # Create experiment directory
    expt_dir = config["experiment_dir"]
    job_name = config.get("name", "jetclr")
    expt_dir = expt_dir.replace("PROJECT_ROOT", str(project_dir))
    expt_dir = expt_dir.replace("JOBNAME", job_name)
    expt_dir = Path(expt_dir)
    if should_save_and_log(accelerator):
        expt_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Experiment directory: {expt_dir}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "auc": [],
        "imtafe": [],
        "align_loss": [],
        "uniform_loss": [],
    }

    global_step = start_epoch * len(train_dataloader)
    if wandb_run is not None and config["training"].get("load_epoch", 0) > 0:
        wandb_step = getattr(wandb_run, "step", 0) or 0
        if wandb_step > global_step:
            LOGGER.info(
                f"Adjusting global_step from {global_step} to {wandb_step} "
                "to match resumed W&B run (avoids non-monotonic step warnings)"
            )
            global_step = wandb_step
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Train
        train_metrics, global_step = train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            config=config,
            epoch=epoch,
            scheduler=scheduler,
            accelerator=accelerator,
            wandb_run=wandb_run,
            global_step=global_step,
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            config=config,
            accelerator=accelerator,
        )

        # Step scheduler (ReduceLROnPlateau only — all others step per batch)
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_metrics["loss"])

        t1 = time.time()

        if should_save_and_log(accelerator):
            log_msg = f"Epoch {epoch}/{num_epochs}\n"
            log_msg += f"Train loss: {train_metrics['loss']:.6f}\n"
            log_msg += f"Val loss: {val_metrics['loss']:.6f}\n"
            if "align_loss" in train_metrics:
                log_msg += f"Align loss: {train_metrics['align_loss']:.6f}\n"
                log_msg += f"Uniform loss: {train_metrics['uniform_loss']:.6f}\n"
            log_msg += f"Time: {t1-t0:.1f}s"
            LOGGER.info(log_msg)

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            if "align_loss" in train_metrics:
                history["align_loss"].append(train_metrics["align_loss"])
                history["uniform_loss"].append(train_metrics["uniform_loss"])

        if wandb_run is not None:
            wandb_metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            wandb_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            wandb_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb_run.log(wandb_metrics, step=global_step)

        # Note: will not run linear classifier test
        # # Run LCT every 10 epochs
        # if epoch % 10 == 0 and should_save_and_log(accelerator):
        #     LOGGER.info("Running Linear Classifier Test...")
        #     auc, imtafe = run_linear_classifier_test(
        #         model=model,
        #         val_dataloader=val_dataloader,
        #         val_dataloader_2=val_dataloader_2,
        #         device=device,
        #         config=config,
        #         accelerator=accelerator,
        #     )
        #     LOGGER.info(f"LCT - AUC: {auc:.4f}, IMTAFE: {imtafe:.1f}")
        #     history["auc"].append(auc)
        #     history["imtafe"].append(imtafe)

        # Save checkpoint
        if epoch % 5 == 0 and should_save_and_log(accelerator):
            checkpoint = {
                "epoch": epoch,
                "train_loss": train_metrics,
                "val_loss": val_metrics,
            }

            # Save model state (handles AssembledModel properly)
            model_state = get_state_dict(model)
            checkpoint.update(model_state)

            # Save optimizer and scheduler
            checkpoint["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint["scheduler"] = scheduler.state_dict()

            save_path = expt_dir / f"model_ep{epoch}.pt"
            torch.save(checkpoint, save_path)
            LOGGER.info(f"Checkpoint saved to {save_path}")

            # Save history
            for key, values in history.items():
                if len(values) > 0:
                    np.save(expt_dir / f"{key}.npy", np.array(values))

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if should_save_and_log(accelerator):
                checkpoint = {
                    "epoch": epoch,
                    "train_loss": train_metrics,
                    "val_loss": val_metrics,
                }

                # Save model state
                model_state = get_state_dict(model)
                checkpoint.update(model_state)

                save_path = expt_dir / "model_best.pt"
                torch.save(checkpoint, save_path)
                LOGGER.info(f"Best model saved to {save_path}")

    # Save final model
    if should_save_and_log(accelerator):
        checkpoint = {
            "epoch": num_epochs - 1,
        }

        # Save model state
        model_state = get_state_dict(model)
        checkpoint.update(model_state)

        checkpoint["optimizer"] = optimizer.state_dict()

        save_path = expt_dir / "model_final.pt"
        torch.save(checkpoint, save_path)
        LOGGER.info(f"Final model saved to {save_path}")

    if accelerator is not None:
        accelerator.end_training()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jetCLR model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lf", "--log-file", type=str, default=None, help="Path to log file"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    configure_logger(
        logger=LOGGER,
        name="jetCLR Training",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    LOGGER.info(f"Config file: {args.config}")
    train(config=config, use_wandb=args.use_wandb)

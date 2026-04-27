"""JetCLR+iBOT+KoLeo training script.

Single-model contrastive pre-training using kT clustering views:
- NT-Xent contrastive loss on CLS tokens (invariance across RG scales)
- iBOT masked particle prediction with stop-gradient targets
- KoLeo entropy maximization regularization

Usage:
    python dino/jetclr_train.py -c configs/jetclr/<config>.yaml
    accelerate launch dino/jetclr_train.py -c configs/jetclr/<config>.yaml
"""

import gc
import math
from pathlib import Path
from typing import Any

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from utils.ckpt import (
    save_checkpoint as _save_checkpoint_raw,
    get_checkpoints_path,
    find_latest_checkpoint_epoch,
    load_checkpoint,
    get_state_dict,
)
from utils.producers import (
    get_optimizer,
    get_models_single,
    get_dataloader_and_config,
    get_scheduler,
)
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device, check_bf16_support
from utils.training import (
    aggregate_losses,
    should_save_and_log,
    _scale_loss_warmup_to_steps,
    _rescale_warmup_steps,
    _get_loss_module,
    get_training_start_params,
    restore_schedulers,
)
from augmentations import Augmenter
from losses import NTXentLoss, iBOTLoss, KoLeoLoss, ReconLoss
from jetclr_train_batch import process_batch

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent


# ---------------------------------------------------------------------- #
# Checkpoint helpers for single-model setup                               #
# ---------------------------------------------------------------------- #


def get_checkpoint_dict(
    backbone: nn.Module,
    projection_head: nn.Module,
    ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    recon_head: nn.Module | None,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    wd_scheduler: Any | None,
    epoch: int,
    train_loss: dict[str, float],
    val_loss: dict[str, float],
    num_stale_epochs: int = 0,
) -> dict[str, Any]:
    checkpoint_dict = {
        "epoch": epoch,
        "backbone": get_state_dict(backbone),
        "projection_head": get_state_dict(projection_head),
        "ibot_head": get_state_dict(ibot_head) if ibot_head is not None else None,
        "ibot_pos_embedding": (
            get_state_dict(ibot_pos_embedding) if ibot_pos_embedding is not None else None
        ),
        "recon_head": get_state_dict(recon_head) if recon_head is not None else None,
        # Also save under DINO-compatible keys for cross-loading
        "student": get_state_dict(backbone),
        "student_dino_head": get_state_dict(projection_head),
        "student_ibot_head": (
            get_state_dict(ibot_head) if ibot_head is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "num_stale_epochs": num_stale_epochs,
    }
    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            checkpoint_dict["scheduler"] = {
                "best": scheduler.best,
                "num_bad_epochs": scheduler.num_bad_epochs,
            }
        else:
            checkpoint_dict["scheduler"] = scheduler.state_dict()
    if wd_scheduler is not None:
        checkpoint_dict["weight_decay_scheduler"] = wd_scheduler.state_dict()
    return checkpoint_dict


# ---------------------------------------------------------------------- #
# Training epoch                                                          #
# ---------------------------------------------------------------------- #


def train_epoch(
    backbone: nn.Module,
    projection_head: nn.Module,
    ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    recon_head: nn.Module | None,
    ntxent_loss: nn.Module,
    ibot_loss: nn.Module | None,
    recon_loss: nn.Module | None,
    koleo_loss: nn.Module | None,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    use_bf16: bool = False,
    accelerator: Accelerator | None = None,
    wandb_run: Any = None,
    global_step: int = 0,
    scheduler=None,
) -> tuple[dict[str, float], int]:
    backbone.train()
    projection_head.train()
    if ibot_head is not None:
        ibot_head.train()
    if ibot_pos_embedding is not None:
        ibot_pos_embedding.train()
    if recon_head is not None:
        recon_head.train()
    ntxent_loss.train()
    if ibot_loss is not None:
        ibot_loss.train()
    if recon_loss is not None:
        recon_loss.train()

    all_batch_losses = []

    with tqdm(
        dataloader,
        desc="Training",
        unit="batch",
        mininterval=10,
        disable=not (accelerator is None or accelerator.is_main_process),
    ) as pbar:
        for i, batch in enumerate(pbar):
            batch_losses = process_batch(
                backbone=backbone,
                projection_head=projection_head,
                ibot_head=ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                recon_head=recon_head,
                ntxent_loss=ntxent_loss,
                ibot_loss=ibot_loss,
                recon_loss=recon_loss,
                koleo_loss=koleo_loss,
                batch=batch,
                device=device,
                config=config,
                augmenter=augmenter,
                optimizer=optimizer,
                accelerator=accelerator,
                use_bf16=use_bf16,
            )

            all_batch_losses.append(batch_losses)

            if should_save_and_log(accelerator):
                formatted_metrics = {}
                for k, v in batch_losses.items():
                    if k not in ["batch_size"]:
                        if k == "masked_ratio":
                            formatted_metrics[k] = f"{v*100:.1f}%"
                        elif k == "num_masked":
                            formatted_metrics[k] = f"{int(v)}"
                        else:
                            formatted_metrics[k] = f"{v:.4f}"
                pbar.set_postfix(formatted_metrics)

                if wandb_run is not None:
                    batch_wandb = {
                        f"batch/{k}": v
                        for k, v in batch_losses.items()
                        if isinstance(v, (int, float)) and k != "batch_size"
                    }
                    batch_wandb["batch/lr"] = optimizer.param_groups[0]["lr"]
                    wandb_run.log(batch_wandb, step=global_step)

            global_step += 1

            _get_loss_module(ntxent_loss).step_step()
            if ibot_loss is not None:
                _get_loss_module(ibot_loss).step_step()
            if recon_loss is not None:
                _get_loss_module(recon_loss).step_step()
            if koleo_loss is not None:
                _get_loss_module(koleo_loss).step_step()

            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict, global_step


# ---------------------------------------------------------------------- #
# Validation                                                              #
# ---------------------------------------------------------------------- #


@torch.no_grad()
def validate(
    backbone: nn.Module,
    projection_head: nn.Module,
    ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    recon_head: nn.Module | None,
    ntxent_loss: nn.Module,
    ibot_loss: nn.Module | None,
    recon_loss: nn.Module | None,
    koleo_loss: nn.Module | None,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    use_bf16: bool = False,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    backbone.eval()
    projection_head.eval()
    if ibot_head is not None:
        ibot_head.eval()
    if ibot_pos_embedding is not None:
        ibot_pos_embedding.eval()
    if recon_head is not None:
        recon_head.eval()
    ntxent_loss.eval()
    if ibot_loss is not None:
        ibot_loss.eval()
    if recon_loss is not None:
        recon_loss.eval()

    all_batch_losses = []

    with tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        mininterval=10,
        disable=not (accelerator is None or accelerator.is_main_process),
    ) as pbar:
        for batch in pbar:
            batch_losses = process_batch(
                backbone=backbone,
                projection_head=projection_head,
                ibot_head=ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                recon_head=recon_head,
                ntxent_loss=ntxent_loss,
                ibot_loss=ibot_loss,
                recon_loss=recon_loss,
                koleo_loss=koleo_loss,
                batch=batch,
                device=device,
                config=config,
                augmenter=augmenter,
                optimizer=None,
                accelerator=accelerator,
                use_bf16=use_bf16,
            )

            all_batch_losses.append(batch_losses)

            if accelerator is None or accelerator.is_main_process:
                formatted_metrics = {}
                for k, v in batch_losses.items():
                    if k not in ["batch_size"]:
                        if k == "masked_ratio":
                            formatted_metrics[k] = f"{v*100:.1f}%"
                        elif k == "num_masked":
                            formatted_metrics[k] = f"{int(v)}"
                        else:
                            formatted_metrics[k] = f"{v:.4f}"
                pbar.set_postfix(formatted_metrics)

    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict


# ---------------------------------------------------------------------- #
# Main training function                                                  #
# ---------------------------------------------------------------------- #


def train(config: dict[str, Any], use_wandb: bool = False) -> None:

    use_accelerate = config.get("accelerate", False)

    # ------------------------------------------------------------------ #
    # float32 matmul precision                                             #
    # ------------------------------------------------------------------ #
    float32_matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(float32_matmul_precision)

    # ------------------------------------------------------------------ #
    # Accelerator / device setup                                           #
    # ------------------------------------------------------------------ #
    if use_accelerate:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,
            split_batches=True,
        )
        kwargs = InitProcessGroupKwargs(timeout=timedelta(days=365))
        accelerator = Accelerator(
            dataloader_config=dataloader_config, kwargs_handlers=[kwargs]
        )
        device = accelerator.device

        if accelerator.is_main_process:
            LOGGER.info("Using accelerate for distributed training.")
        LOGGER.info(
            f"Local process index: {accelerator.local_process_index}/{accelerator.num_processes}, "
            f"Process index: {accelerator.process_index}/{accelerator.num_processes}, "
            f"Is main process: {accelerator.is_main_process}"
        )
        LOGGER.addFilter(lambda record: accelerator.is_main_process)
    else:
        accelerator = None
        device = config.get("device", None)
        if device is None:
            device = get_available_device()
        else:
            device = torch.device(device)

        try:
            torch.empty(1).to(device)
        except AssertionError as e:
            LOGGER.error(f"Error: {e}")
            device = get_available_device()
            LOGGER.info(f"Using default device: {device}")

    # ------------------------------------------------------------------ #
    # bf16 support check                                                   #
    # ------------------------------------------------------------------ #
    want_bf16 = config.get("use_bf16", False)
    if want_bf16:
        bf16_supported = check_bf16_support(device)
        if not bf16_supported:
            LOGGER.warning(
                f"use_bf16=True requested but bfloat16 is NOT supported on device "
                f"'{device}'. Falling back to float32."
            )
        use_bf16 = bf16_supported
    else:
        use_bf16 = False

    seed = config.get("seed", None)
    if seed is not None:
        import random

        rank = accelerator.process_index if accelerator is not None else 0
        rank_seed = seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)
        LOGGER.info(f"Random seed set to {rank_seed} (base={seed}, rank={rank})")

    if should_save_and_log(accelerator):
        LOGGER.info(f"PyTorch version: {torch.__version__}")
        LOGGER.info(f"CUDA version: {torch.version.cuda}")
        LOGGER.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        LOGGER.info(f"float32_matmul_precision: {float32_matmul_precision}")
        LOGGER.info(f"use_bf16: {use_bf16} (requested: {want_bf16})")
        LOGGER.info(f"Training with config: {config}")

    # ------------------------------------------------------------------ #
    # auto_resume                                                         #
    # ------------------------------------------------------------------ #
    _training_params_early = config["training"]
    if (
        _training_params_early.get("auto_resume", False)
        and _training_params_early.get("load_epoch") is None
    ):
        _latest = find_latest_checkpoint_epoch(config)
        if _latest is not None:
            _training_params_early["load_epoch"] = _latest
            if should_save_and_log(accelerator):
                LOGGER.info(f"auto_resume: found latest checkpoint at epoch {_latest}")
        elif should_save_and_log(accelerator):
            LOGGER.info("auto_resume: no existing checkpoints found, starting from scratch")

    # ------------------------------------------------------------------ #
    # W&B                                                                  #
    # ------------------------------------------------------------------ #
    wandb_run = None
    if use_wandb and should_save_and_log(accelerator):
        try:
            import wandb

            wandb_id = None
            wandb_id_file = get_checkpoints_path(config, 0).parent / "wandb_run_id.txt"
            if (
                config["training"].get("load_epoch") is not None
                and wandb_id_file.exists()
            ):
                wandb_id = wandb_id_file.read_text().strip()
                LOGGER.info(f"Resuming W&B run {wandb_id}")

            wandb_run = wandb.init(
                project="JetCLR-training",
                name=config.get("name", "jetclr"),
                config=config,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
            wandb_id_file.write_text(wandb_run.id)
            LOGGER.info("W&B run initialized")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    train_dataloader, dataloader_config = get_dataloader_and_config(
        config=config, split="train", mode="training", accelerator=accelerator,
    )
    val_dataloader, _ = get_dataloader_and_config(
        config=config, split="val", mode="training", accelerator=accelerator,
    )

    augmenter = Augmenter(
        labels_parts=dataloader_config.outputs.sequence,
        labels_jet=dataloader_config.outputs.class_,
        **config["augmentation_params"]["augmenter_kwargs"],
    )

    part_features = dataloader_config.outputs.sequence
    part_dim = len(part_features)

    # ------------------------------------------------------------------ #
    # Model (single — no teacher)                                          #
    # ------------------------------------------------------------------ #
    backbone, projection_head, ibot_head, ibot_pos_embedding = get_models_single(
        part_dim=part_dim,
        config=config,
        mode="training",
        device=(device if not use_accelerate else None),
    )

    # Reconstruction head (optional): maps d_model -> part_dim for MAE-style loss
    recon_head = None
    if "recon_head" in config.get("models", {}):
        from models import MLPHead
        recon_head_config = config["models"]["recon_head"]
        recon_head_params = recon_head_config.get("params", {})
        recon_in_dim = backbone.d_model
        recon_head = MLPHead(input_dim=recon_in_dim, **recon_head_params)
        if not use_accelerate:
            recon_head = recon_head.to(device)
        LOGGER.info(f"Reconstruction head: {recon_head}")

    # ------------------------------------------------------------------ #
    # Steps per epoch                                                      #
    # ------------------------------------------------------------------ #
    if use_accelerate and not accelerator.dispatch_batches:
        num_processes = accelerator.num_processes
        steps_per_epoch = math.ceil(len(train_dataloader) / num_processes)
    else:
        steps_per_epoch = len(train_dataloader)
    LOGGER.info(f"steps_per_epoch (pre-prepare) = {steps_per_epoch}")

    # ------------------------------------------------------------------ #
    # Losses                                                               #
    # ------------------------------------------------------------------ #
    ntxent_loss = NTXentLoss(
        **_scale_loss_warmup_to_steps(config["loss_params"]["ntxent"], steps_per_epoch),
    )

    ibot_loss = None
    if ibot_head is not None and "ibot" in config["loss_params"]:
        ibot_loss = iBOTLoss(
            out_dim=config["models"]["ibot_head"]["params"]["output_dim"],
            **_scale_loss_warmup_to_steps(
                config["loss_params"]["ibot"], steps_per_epoch
            ),
        )
        LOGGER.info("iBOT loss enabled")

    recon_loss = None
    if recon_head is not None and "recon" in config["loss_params"]:
        recon_loss = ReconLoss(
            **_scale_loss_warmup_to_steps(
                config["loss_params"]["recon"], steps_per_epoch
            ),
        )
        LOGGER.info("Reconstruction loss enabled")

    koleo_loss = None
    if "koleo" in config["loss_params"]:
        koleo_loss = KoLeoLoss(
            **_scale_loss_warmup_to_steps(
                config["loss_params"]["koleo"], steps_per_epoch
            )
        )
        LOGGER.info(f"KoLeo loss enabled with params: {config['loss_params']['koleo']}")

    if not use_accelerate:
        ntxent_loss = ntxent_loss.to(device)
        if ibot_loss is not None:
            ibot_loss = ibot_loss.to(device)
        if recon_loss is not None:
            recon_loss = recon_loss.to(device)
        if koleo_loss is not None:
            koleo_loss = koleo_loss.to(device)

    if should_save_and_log(accelerator):
        LOGGER.info(f"Loss params: {config['loss_params']}")

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler                                                #
    # ------------------------------------------------------------------ #
    head_modules = [projection_head]
    if ibot_head is not None:
        head_modules.append(ibot_head)
    if ibot_pos_embedding is not None:
        head_modules.append(ibot_pos_embedding)
    if recon_head is not None:
        head_modules.append(recon_head)

    optimizer, wd_scheduler = get_optimizer(
        config=config, backbone=backbone, head_modules=head_modules,
    )
    if should_save_and_log(accelerator):
        LOGGER.info(f"Optimizer: {optimizer}")

    scheduler = get_scheduler(
        optimizer=optimizer, config=config, steps_per_epoch=steps_per_epoch
    )

    # ------------------------------------------------------------------ #
    # Accelerator prepare                                                  #
    # ------------------------------------------------------------------ #
    if use_accelerate:
        prepare_dict = {
            "backbone": backbone,
            "projection_head": projection_head,
            "optimizer": optimizer,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "ntxent_loss": ntxent_loss,
        }
        if ibot_head is not None:
            prepare_dict["ibot_head"] = ibot_head
        if ibot_pos_embedding is not None:
            prepare_dict["ibot_pos_embedding"] = ibot_pos_embedding
        if ibot_loss is not None:
            prepare_dict["ibot_loss"] = ibot_loss
        if recon_head is not None:
            prepare_dict["recon_head"] = recon_head
        if recon_loss is not None:
            prepare_dict["recon_loss"] = recon_loss
        if koleo_loss is not None:
            prepare_dict["koleo_loss"] = koleo_loss

        prepared = accelerator.prepare(*prepare_dict.values())
        prepared_dict = dict(zip(prepare_dict.keys(), prepared))

        backbone = prepared_dict["backbone"]
        projection_head = prepared_dict["projection_head"]
        optimizer = prepared_dict["optimizer"]
        train_dataloader = prepared_dict["train_dataloader"]
        val_dataloader = prepared_dict["val_dataloader"]
        ntxent_loss = prepared_dict["ntxent_loss"]

        if ibot_head is not None:
            ibot_head = prepared_dict["ibot_head"]
        if ibot_pos_embedding is not None:
            ibot_pos_embedding = prepared_dict["ibot_pos_embedding"]
        if ibot_loss is not None:
            ibot_loss = prepared_dict["ibot_loss"]
        if recon_head is not None:
            recon_head = prepared_dict["recon_head"]
        if recon_loss is not None:
            recon_loss = prepared_dict["recon_loss"]
        if koleo_loss is not None:
            koleo_loss = prepared_dict["koleo_loss"]

        LOGGER.info("Accelerator prepared all components.")

    # ------------------------------------------------------------------ #
    # Correct steps_per_epoch from the prepared dataloader                 #
    # ------------------------------------------------------------------ #
    actual_steps_per_epoch = len(train_dataloader)
    if actual_steps_per_epoch != steps_per_epoch:
        if should_save_and_log(accelerator):
            LOGGER.info(
                f"steps_per_epoch corrected after prepare: "
                f"{steps_per_epoch} -> {actual_steps_per_epoch}"
            )
        _rescale_warmup_steps(
            _get_loss_module(ntxent_loss), steps_per_epoch, actual_steps_per_epoch
        )
        if ibot_loss is not None:
            _rescale_warmup_steps(
                _get_loss_module(ibot_loss), steps_per_epoch, actual_steps_per_epoch
            )
        if recon_loss is not None:
            _rescale_warmup_steps(
                _get_loss_module(recon_loss), steps_per_epoch, actual_steps_per_epoch
            )
        if koleo_loss is not None:
            _rescale_warmup_steps(
                _get_loss_module(koleo_loss), steps_per_epoch, actual_steps_per_epoch
            )
        scheduler = get_scheduler(
            optimizer=optimizer, config=config, steps_per_epoch=actual_steps_per_epoch
        )
        steps_per_epoch = actual_steps_per_epoch
    if should_save_and_log(accelerator):
        LOGGER.info(f"steps_per_epoch = {steps_per_epoch}")

    # ------------------------------------------------------------------ #
    # Training state                                                       #
    # ------------------------------------------------------------------ #
    training_params = config["training"]

    start_epoch, best_val_loss, num_stale_epochs = get_training_start_params(
        config=config, device=device, accelerator=accelerator
    )

    if start_epoch > 0:
        load_epoch = training_params.get("load_epoch")
        checkpoint = load_checkpoint(config=config, device="cpu", epoch=load_epoch)

        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(
                        accelerator.device if accelerator is not None else device
                    )

        restore_schedulers(
            checkpoint=checkpoint, wd_scheduler=wd_scheduler, accelerator=accelerator,
        )

    patience = training_params.get("patience", float("inf"))
    num_epochs: int = training_params["num_epochs"]

    reset_best_val_loss_epochs: set[int] = set(
        training_params.get("reset_best_val_loss_epochs", [])
    )

    if start_epoch > 0:
        _get_loss_module(ntxent_loss).resume_epoch(start_epoch)
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).resume_epoch(start_epoch)
        if recon_loss is not None:
            _get_loss_module(recon_loss).resume_epoch(start_epoch)
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).resume_epoch(start_epoch)

    should_stop = False

    # ------------------------------------------------------------------ #
    # Save untrained checkpoint                                            #
    # ------------------------------------------------------------------ #
    if should_save_and_log(accelerator) and start_epoch == 0:
        untrained_ckpt = get_checkpoint_dict(
            backbone=backbone, projection_head=projection_head,
            ibot_head=ibot_head, ibot_pos_embedding=ibot_pos_embedding, recon_head=recon_head,
            optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
            epoch=-1,
            train_loss={"loss": float("inf")},
            val_loss={"loss": float("inf")},
            num_stale_epochs=0,
        )
        saved_path = _save_checkpoint_raw(
            checkpoint_dict=untrained_ckpt, config=config, epoch_num="untrained",
        )
        LOGGER.info(f"Untrained model saved to {saved_path}")

    if num_epochs == 0:
        LOGGER.info("Configured for 0 epochs. Exiting after saving untrained checkpoint.")
        return

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #
    global_step = start_epoch * len(train_dataloader)
    if wandb_run is not None and config["training"].get("load_epoch") is not None:
        wandb_step = getattr(wandb_run, "step", 0) or 0
        if wandb_step > global_step:
            LOGGER.info(
                f"Adjusting global_step from {global_step} to {wandb_step} "
                "to match resumed W&B run"
            )
            global_step = wandb_step
    if start_epoch > 0:
        _get_loss_module(ntxent_loss).resume_step(global_step)
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).resume_step(global_step)
        if recon_loss is not None:
            _get_loss_module(recon_loss).resume_step(global_step)
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).resume_step(global_step)

    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        if should_save_and_log(accelerator):
            LOGGER.info(f"Starting epoch {epoch} with learning rate: {current_lr:.8f}")

        detect_anomaly = config["training"].get("autograd_detect_anomaly", False)
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            train_metrics, global_step = train_epoch(
                backbone=backbone,
                projection_head=projection_head,
                ibot_head=ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                recon_head=recon_head,
                ntxent_loss=ntxent_loss,
                ibot_loss=ibot_loss,
                recon_loss=recon_loss,
                koleo_loss=koleo_loss,
                optimizer=optimizer,
                dataloader=train_dataloader,
                device=device,
                config=config,
                augmenter=augmenter,
                use_bf16=use_bf16,
                accelerator=accelerator,
                wandb_run=wandb_run,
                global_step=global_step,
                scheduler=scheduler,
            )

        # Save intermediate checkpoint (before validation)
        if should_save_and_log(accelerator):
            ckpt = get_checkpoint_dict(
                backbone=backbone, projection_head=projection_head,
                ibot_head=ibot_head, ibot_pos_embedding=ibot_pos_embedding, recon_head=recon_head,
                optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
                epoch=epoch, train_loss=train_metrics,
                val_loss={"loss": float("inf")},
                num_stale_epochs=num_stale_epochs,
            )
            _save_checkpoint_raw(checkpoint_dict=ckpt, config=config, epoch_num=epoch)

        gc.collect()

        val_metrics = validate(
            backbone=backbone,
            projection_head=projection_head,
            ibot_head=ibot_head,
            ibot_pos_embedding=ibot_pos_embedding,
            recon_head=recon_head,
            ntxent_loss=ntxent_loss,
            ibot_loss=ibot_loss,
            recon_loss=recon_loss,
            koleo_loss=koleo_loss,
            dataloader=val_dataloader,
            device=device,
            config=config,
            augmenter=augmenter,
            use_bf16=use_bf16,
            accelerator=accelerator,
        )

        # Step epoch-level loss schedulers
        _get_loss_module(ntxent_loss).step_epoch()
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).step_epoch()
        if recon_loss is not None:
            _get_loss_module(recon_loss).step_epoch()
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).step_epoch()

        if wd_scheduler is not None:
            current_wd = wd_scheduler.step()
            if should_save_and_log(accelerator):
                LOGGER.info(f"Weight decay after epoch {epoch}: {current_wd:.6f}")

        val_loss = val_metrics["loss"]
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Reset best_val_loss if designated
        if epoch in reset_best_val_loss_epochs:
            if should_save_and_log(accelerator):
                LOGGER.info(
                    f"[reset_best_val_loss] Epoch {epoch}: resetting "
                    f"best_val_loss ({best_val_loss:.8f} -> inf) and "
                    f"num_stale_epochs ({num_stale_epochs} -> 0)."
                )
            best_val_loss = float("inf")
            num_stale_epochs = 0

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

        if wandb_run is not None:
            wandb_metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            wandb_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            wandb_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb_metrics["loss/ntxent_weight"] = _get_loss_module(
                ntxent_loss
            ).get_current_weight()
            if ibot_loss is not None:
                ibot_mod = _get_loss_module(ibot_loss)
                wandb_metrics["loss/ibot_weight"] = ibot_mod.get_current_weight()
                wandb_metrics["loss/ibot_teacher_temp"] = (
                    ibot_mod.get_current_teacher_temp()
                )
            if recon_loss is not None:
                wandb_metrics["loss/recon_weight"] = _get_loss_module(
                    recon_loss
                ).get_current_weight()
            if koleo_loss is not None:
                wandb_metrics["loss/koleo_weight"] = _get_loss_module(
                    koleo_loss
                ).get_current_weight()
            wandb_run.log(wandb_metrics, step=global_step)

        # Patience tracking & best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_stale_epochs = 0

            if should_save_and_log(accelerator):
                ckpt = get_checkpoint_dict(
                    backbone=backbone, projection_head=projection_head,
                    ibot_head=ibot_head, ibot_pos_embedding=ibot_pos_embedding, recon_head=recon_head,
                    optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
                    epoch=epoch, train_loss=train_metrics, val_loss=val_metrics,
                    num_stale_epochs=num_stale_epochs,
                )
                saved_path = _save_checkpoint_raw(
                    checkpoint_dict=ckpt, config=config, epoch_num="best",
                )
                LOGGER.info(f"Best model saved to {saved_path}")
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

        # Save epoch checkpoint
        if should_save_and_log(accelerator):
            ckpt = get_checkpoint_dict(
                backbone=backbone, projection_head=projection_head,
                ibot_head=ibot_head, ibot_pos_embedding=ibot_pos_embedding, recon_head=recon_head,
                optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
                epoch=epoch, train_loss=train_metrics, val_loss=val_metrics,
                num_stale_epochs=num_stale_epochs,
            )
            saved_path = _save_checkpoint_raw(
                checkpoint_dict=ckpt, config=config, epoch_num=epoch,
            )
            LOGGER.info(f"Checkpoint saved to {saved_path}")

        if should_stop:
            if (
                accelerator is not None
                and accelerator.num_processes > 1
                and accelerator.is_main_process
            ):
                accelerator.set_trigger()
            else:
                break

        if accelerator is not None:
            if accelerator.check_trigger():
                break

    # Save final checkpoint
    if should_save_and_log(accelerator):
        ckpt = get_checkpoint_dict(
            backbone=backbone, projection_head=projection_head,
            ibot_head=ibot_head, ibot_pos_embedding=ibot_pos_embedding, recon_head=recon_head,
            optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler,
            epoch=epoch, train_loss=train_metrics, val_loss=val_metrics,
            num_stale_epochs=num_stale_epochs,
        )
        saved_path = _save_checkpoint_raw(
            checkpoint_dict=ckpt, config=config, epoch_num="final",
        )
        LOGGER.info(f"Final model saved to {saved_path}")

    if accelerator is not None:
        accelerator.end_training()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JetCLR+iBOT+KoLeo model")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-lv", "--log-level", type=str, default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("-lf", "--log-file", type=str, default=None)
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    configure_logger(
        logger=LOGGER,
        name="JetCLR Training",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    try:
        train(config=config, use_wandb=args.use_wandb)
    except Exception as e:
        LOGGER.error(f"An error occurred during training: {e}")
        raise e

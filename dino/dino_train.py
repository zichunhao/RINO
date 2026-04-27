import gc
import math
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
from utils.ckpt import (
    save_checkpoint,
    get_checkpoints_path,
    find_latest_checkpoint_epoch,
    get_checkpoint_dict,
    load_checkpoint,
)
from utils.producers import (
    get_optimizer,
    get_models,
    get_dataloader_and_config,
    get_scheduler,
)
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device, check_bf16_support
from utils.ema import update_momentum_encoder
from utils.training import (
    aggregate_losses,
    cosine_scheduler,
    should_save_and_log,
    _rescale_warmup_steps,
    _scale_loss_warmup_to_steps,
    get_training_start_params,
    restore_schedulers,
    _get_loss_module,
)
from augmentations import Augmenter
from losses import DINOLoss, iBOTLoss, GramLoss, KoLeoLoss
from dino_train_batch import process_batch

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent


def train_epoch(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    student_ibot_head: nn.Module,
    teacher_backbone: nn.Module,
    teacher_dino_head: nn.Module,
    teacher_ibot_head: nn.Module,
    ibot_pos_embedding: nn.Module | None,
    dino_loss: nn.Module,
    ibot_loss: nn.Module | None,
    gram_loss: nn.Module | None,
    koleo_loss: nn.Module | None,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    momentum_schedule: np.ndarray,
    use_bf16: bool = False,
    accelerator: Accelerator | None = None,
    wandb_run: Any = None,
    global_step: int = 0,
    scheduler=None,
    dino_scale_embedding: nn.Module | None = None,
) -> tuple[dict[str, float], int]:
    student_backbone.train()
    student_dino_head.train()
    if student_ibot_head is not None:
        student_ibot_head.train()
    teacher_backbone.eval()
    teacher_dino_head.eval()
    if teacher_ibot_head is not None:
        teacher_ibot_head.eval()
    if ibot_pos_embedding is not None:
        ibot_pos_embedding.eval()
    dino_loss.train()
    if ibot_loss is not None:
        ibot_loss.train()
    if gram_loss is not None:
        gram_loss.train()

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
                student_backbone=student_backbone,
                student_dino_head=student_dino_head,
                student_ibot_head=student_ibot_head,
                teacher_backbone=teacher_backbone,
                teacher_dino_head=teacher_dino_head,
                teacher_ibot_head=teacher_ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                dino_loss=dino_loss,
                ibot_loss=ibot_loss,
                gram_loss=gram_loss,
                koleo_loss=koleo_loss,
                batch=batch,
                device=device,
                config=config,
                augmenter=augmenter,
                optimizer=optimizer,
                accelerator=accelerator,
                use_bf16=use_bf16,
                dino_scale_embedding=dino_scale_embedding,
                step_index=i,
            )

            all_batch_losses.append(batch_losses)

            if should_save_and_log(accelerator):
                formatted_metrics = {}
                for k, v in batch_losses.items():
                    if k not in ["batch_size", "ibot_status", "gram_weight"]:
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
                        if isinstance(v, (int, float)) and k not in ("batch_size",)
                    }
                    # NOTE: param_groups[0] is the stem (deepest backbone layer).
                    # With LLRD (backbone_lr_decay < 1), this is the most decayed LR.
                    # The head LR (max across groups) equals the true base LR.
                    batch_wandb["batch/lr"] = optimizer.param_groups[0]["lr"]
                    batch_wandb["batch/dino_teacher_temp"] = _get_loss_module(
                        dino_loss
                    ).get_current_teacher_temp()
                    if ibot_loss is not None:
                        batch_wandb["batch/ibot_teacher_temp"] = _get_loss_module(
                            ibot_loss
                        ).get_current_teacher_temp()
                    batch_wandb["batch/teacher_momentum"] = momentum_schedule[
                        min(global_step + 1, len(momentum_schedule) - 1)
                    ]
                    wandb_run.log(batch_wandb, step=global_step)

            global_step += 1

            _get_loss_module(dino_loss).step_step()
            if ibot_loss is not None:
                _get_loss_module(ibot_loss).step_step()
            if gram_loss is not None:
                _get_loss_module(gram_loss).step_step()
            if koleo_loss is not None:
                _get_loss_module(koleo_loss).step_step()

            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            momentum = momentum_schedule[min(global_step, len(momentum_schedule) - 1)]

            with torch.no_grad():
                update_momentum_encoder(
                    student=student_backbone, teacher=teacher_backbone, m=momentum
                )
                update_momentum_encoder(
                    student=student_dino_head, teacher=teacher_dino_head, m=momentum
                )
                if student_ibot_head is not None and teacher_ibot_head is not None:
                    update_momentum_encoder(
                        student=student_ibot_head, teacher=teacher_ibot_head, m=momentum
                    )

    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict, global_step


@torch.no_grad()
def validate(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    student_ibot_head: nn.Module,
    teacher_backbone: nn.Module,
    teacher_dino_head: nn.Module,
    teacher_ibot_head: nn.Module,
    ibot_pos_embedding: nn.Module | None,
    dino_loss: nn.Module,
    ibot_loss: nn.Module | None,
    gram_loss: nn.Module | None,
    koleo_loss: nn.Module | None,
    dataloader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    augmenter: Augmenter,
    use_bf16: bool = False,
    accelerator: Accelerator | None = None,
    dino_scale_embedding: nn.Module | None = None,
) -> dict[str, float]:
    student_backbone.eval()
    student_dino_head.eval()
    if student_ibot_head is not None:
        student_ibot_head.eval()
    teacher_backbone.eval()
    teacher_dino_head.eval()
    if teacher_ibot_head is not None:
        teacher_ibot_head.eval()
    if ibot_pos_embedding is not None:
        ibot_pos_embedding.eval()
    dino_loss.eval()
    if ibot_loss is not None:
        ibot_loss.eval()
    if gram_loss is not None:
        gram_loss.eval()

    all_batch_losses = []

    # Optional held-out validation views. If `augmentation_params_val` is
    # present in the config, the validation loop uses it in place of the
    # training `augmentation_params`. This lets the config hold out certain
    # subjet scales from training and measure the DINO / iBOT loss on them
    # during validation — a direct probe of cross-RG-scale interpolation.
    # Back-compatible: absent → val uses the same views as training.
    val_config = config
    _saved_num_global = _saved_num_local = None
    if config.get("augmentation_params_val") is not None:
        val_config = {**config, "augmentation_params": config["augmentation_params_val"]}
        LOGGER.info(
            f"Validation using held-out augmentation_params_val "
            f"(global={val_config['augmentation_params'].get('global')}, "
            f"local={val_config['augmentation_params'].get('local')})"
        )
        # Override DINOLoss view counts to match val views so .chunk() aligns.
        val_num_global = sum(
            v["num"] for v in val_config["augmentation_params"]["global"]
        )
        val_num_local = sum(
            v["num"] for v in val_config["augmentation_params"]["local"]
        )
        _saved_num_global = dino_loss.num_global_views
        _saved_num_local = dino_loss.num_local_views
        dino_loss.num_global_views = val_num_global
        dino_loss.num_local_views = val_num_local

    with tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        mininterval=10,
        disable=not (accelerator is None or accelerator.is_main_process),
    ) as pbar:
        for batch in pbar:
            batch_losses = process_batch(
                student_backbone=student_backbone,
                student_dino_head=student_dino_head,
                student_ibot_head=student_ibot_head,
                teacher_backbone=teacher_backbone,
                teacher_dino_head=teacher_dino_head,
                teacher_ibot_head=teacher_ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                dino_scale_embedding=dino_scale_embedding,
                dino_loss=dino_loss,
                ibot_loss=ibot_loss,
                gram_loss=gram_loss,
                koleo_loss=koleo_loss,
                batch=batch,
                device=device,
                config=val_config,
                augmenter=augmenter,
                optimizer=None,
                accelerator=accelerator,
                use_bf16=use_bf16,
            )

            all_batch_losses.append(batch_losses)

            if accelerator is None or accelerator.is_main_process:
                formatted_metrics = {}
                for k, v in batch_losses.items():
                    if k not in ["batch_size", "ibot_status", "gram_weight"]:
                        if k == "masked_ratio":
                            formatted_metrics[k] = f"{v*100:.1f}%"
                        elif k == "num_masked":
                            formatted_metrics[k] = f"{int(v)}"
                        else:
                            formatted_metrics[k] = f"{v:.4f}"
                pbar.set_postfix(formatted_metrics)

    # Restore training view counts so subsequent training epochs are unaffected.
    if _saved_num_global is not None:
        dino_loss.num_global_views = _saved_num_global
        dino_loss.num_local_views = _saved_num_local

    loss_dict = aggregate_losses(all_batch_losses, accelerator)
    return loss_dict


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
        rank = accelerator.process_index if accelerator is not None else 0
        rank_seed = seed + rank
        import random  # noqa: PLC0415

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
    # auto_resume: resolve load_epoch before anything that depends on it  #
    # (must run before W&B init so the correct run ID is looked up)       #
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
            LOGGER.info(
                "auto_resume: no existing checkpoints found, starting from scratch"
            )

    # ------------------------------------------------------------------ #
    # W&B                                                                #
    # ------------------------------------------------------------------ #
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
                project="RINO-training",
                name=config.get("name", "rino"),
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

    augmenter = Augmenter(
        labels_parts=dataloader_config.outputs.sequence,
        labels_jet=dataloader_config.outputs.class_,
        **config["augmentation_params"]["augmenter_kwargs"],
    )

    part_features = dataloader_config.outputs.sequence
    part_dim = len(part_features)

    (
        (student_backbone, student_dino_head, student_ibot_head),
        (teacher_backbone, teacher_dino_head, teacher_ibot_head),
        ibot_pos_embedding,
        dino_scale_embedding,
    ) = get_models(
        part_dim=part_dim,
        config=config,
        mode="training",
        device=(device if not use_accelerate else None),
    )

    # ------------------------------------------------------------------ #
    # Steps-per-epoch: used to convert epoch-based warmup configs to     #
    # step-based ones so temperature and LR ramps are smooth.            #
    # Use the raw dataloader length here (before accelerator.prepare).   #
    # For dispatch_batches=True every process sees all batches, so no    #
    # division is needed.  For standard DDP the prepared dataloader will #
    # have a shorter length; we correct that after prepare() below.      #
    # ------------------------------------------------------------------ #
    if use_accelerate and not accelerator.dispatch_batches:
        # Standard DDP: each process gets a shard of the data
        num_processes = accelerator.num_processes
        steps_per_epoch = math.ceil(len(train_dataloader) / num_processes)
        LOGGER.info(
            f"steps_per_epoch (pre-prepare) = {steps_per_epoch} (num_processes={num_processes})"
        )
    else:
        # dispatch_batches=True: every process iterates all batches
        steps_per_epoch = len(train_dataloader)
        LOGGER.info(f"steps_per_epoch (pre-prepare) = {steps_per_epoch}")

    # ------------------------------------------------------------------ #
    # Losses                                                             #
    # ------------------------------------------------------------------ #
    num_global_views = sum(
        view["num"] for view in config["augmentation_params"]["global"]
    )
    num_local_views = sum(
        view["num"] for view in config["augmentation_params"]["local"]
    )
    if config["augmentation_params"].get("symmetric", False):
        assert num_global_views == num_local_views, (
            f"Symmetric DINO requires equal view counts, "
            f"got {num_global_views} global vs {num_local_views} local"
        )
        LOGGER.info(
            "Symmetric DINO enabled: global/local views will alternate per batch"
        )
    dino_loss = DINOLoss(
        out_dim=config["models"]["dino_head"]["params"]["output_dim"],
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        **_scale_loss_warmup_to_steps(config["loss_params"]["dino"], steps_per_epoch),
    )

    ibot_loss = None
    if student_ibot_head is not None and "ibot" in config["loss_params"]:
        if "ibot_head" in config["models"]:
            ibot_loss = iBOTLoss(
                out_dim=config["models"]["ibot_head"]["params"]["output_dim"],
                **_scale_loss_warmup_to_steps(
                    config["loss_params"]["ibot"], steps_per_epoch
                ),
            )
            LOGGER.info("iBOT loss enabled")
        else:
            LOGGER.warning(
                "iBOT loss params found but ibot_head not in config. Skipping iBOT loss."
            )
    elif "ibot" in config["loss_params"]:
        LOGGER.warning(
            "iBOT loss params found but no ibot_head model provided. Skipping iBOT loss."
        )

    gram_loss = None
    if "gram" in config["loss_params"]:
        gram_loss = GramLoss(
            **_scale_loss_warmup_to_steps(
                config["loss_params"]["gram"], steps_per_epoch
            )
        )
        LOGGER.info(f"Gram loss enabled with params: {config['loss_params']['gram']}")

    koleo_loss = None
    if "koleo" in config["loss_params"]:
        koleo_loss = KoLeoLoss(
            **_scale_loss_warmup_to_steps(
                config["loss_params"]["koleo"], steps_per_epoch
            )
        )
        LOGGER.info(f"KoLeo loss enabled with params: {config['loss_params']['koleo']}")

    if not use_accelerate:
        dino_loss = dino_loss.to(device)
        if ibot_loss is not None:
            ibot_loss = ibot_loss.to(device)
        if gram_loss is not None:
            gram_loss = gram_loss.to(device)
        if koleo_loss is not None:
            koleo_loss = koleo_loss.to(device)

    if should_save_and_log(accelerator):
        LOGGER.info(f"Loss params: {config['loss_params']}")

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler                                              #
    # ------------------------------------------------------------------ #
    head_modules = [student_dino_head]
    if student_ibot_head is not None:
        head_modules.append(student_ibot_head)
    if ibot_pos_embedding is not None:
        head_modules.append(ibot_pos_embedding)
    if dino_scale_embedding is not None:
        head_modules.append(dino_scale_embedding)

    optimizer, wd_scheduler = get_optimizer(
        config=config,
        backbone=student_backbone,
        head_modules=head_modules,
    )
    if should_save_and_log(accelerator):
        LOGGER.info(f"Optimizer: {optimizer}")
        if wd_scheduler:
            LOGGER.info(f"Weight decay scheduler: {wd_scheduler}")

    scheduler = get_scheduler(
        optimizer=optimizer, config=config, steps_per_epoch=steps_per_epoch
    )
    if should_save_and_log(accelerator):
        LOGGER.info(f"Scheduler: {scheduler}")

    # ------------------------------------------------------------------ #
    # Accelerator prepare                                                #
    # ------------------------------------------------------------------ #
    if use_accelerate:
        prepare_dict = {
            "student_backbone": student_backbone,
            "student_dino_head": student_dino_head,
            "teacher_backbone": teacher_backbone,
            "teacher_dino_head": teacher_dino_head,
            "optimizer": optimizer,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "dino_loss": dino_loss,
        }
        if student_ibot_head is not None:
            prepare_dict["student_ibot_head"] = student_ibot_head
        if teacher_ibot_head is not None:
            prepare_dict["teacher_ibot_head"] = teacher_ibot_head
        if ibot_pos_embedding is not None:
            prepare_dict["ibot_pos_embedding"] = ibot_pos_embedding
        if dino_scale_embedding is not None:
            prepare_dict["dino_scale_embedding"] = dino_scale_embedding
        if ibot_loss is not None:
            prepare_dict["ibot_loss"] = ibot_loss
        if gram_loss is not None:
            prepare_dict["gram_loss"] = gram_loss
        if koleo_loss is not None:
            prepare_dict["koleo_loss"] = koleo_loss

        prepared = accelerator.prepare(*prepare_dict.values())
        prepared_dict = dict(zip(prepare_dict.keys(), prepared))

        # Unpack by name — completely immune to ordering bugs
        student_backbone = prepared_dict["student_backbone"]
        student_dino_head = prepared_dict["student_dino_head"]
        teacher_backbone = prepared_dict["teacher_backbone"]
        teacher_dino_head = prepared_dict["teacher_dino_head"]
        optimizer = prepared_dict["optimizer"]
        train_dataloader = prepared_dict["train_dataloader"]
        val_dataloader = prepared_dict["val_dataloader"]
        dino_loss = prepared_dict["dino_loss"]

        if student_ibot_head is not None:
            student_ibot_head = prepared_dict["student_ibot_head"]
        if teacher_ibot_head is not None:
            teacher_ibot_head = prepared_dict["teacher_ibot_head"]
        if ibot_pos_embedding is not None:
            ibot_pos_embedding = prepared_dict["ibot_pos_embedding"]
        if dino_scale_embedding is not None:
            dino_scale_embedding = prepared_dict["dino_scale_embedding"]
        if ibot_loss is not None:
            ibot_loss = prepared_dict["ibot_loss"]
        if gram_loss is not None:
            gram_loss = prepared_dict["gram_loss"]
        if koleo_loss is not None:
            koleo_loss = prepared_dict["koleo_loss"]

        LOGGER.info("Accelerator prepared all components.")

    # ------------------------------------------------------------------ #
    # Correct steps_per_epoch from the prepared dataloader.              #
    # With dispatch_batches=True the length is unchanged (every process  #
    # sees all batches).  With standard DDP it shrinks by num_processes. #
    # ------------------------------------------------------------------ #
    actual_steps_per_epoch = len(train_dataloader)
    if actual_steps_per_epoch != steps_per_epoch:
        if should_save_and_log(accelerator):
            LOGGER.info(
                f"steps_per_epoch corrected after prepare: "
                f"{steps_per_epoch} → {actual_steps_per_epoch} "
                f"(standard DDP sharding)"
            )
        _rescale_warmup_steps(
            _get_loss_module(dino_loss), steps_per_epoch, actual_steps_per_epoch
        )
        if ibot_loss is not None:
            _rescale_warmup_steps(
                _get_loss_module(ibot_loss), steps_per_epoch, actual_steps_per_epoch
            )
        if gram_loss is not None:
            _rescale_warmup_steps(
                _get_loss_module(gram_loss), steps_per_epoch, actual_steps_per_epoch
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
    # Training state                                                     #
    # ------------------------------------------------------------------ #
    training_params = config["training"]

    start_epoch, best_val_loss, num_stale_epochs = get_training_start_params(
        config=config, device=device, accelerator=accelerator
    )

    if start_epoch > 0:
        load_epoch = training_params.get("load_epoch")
        # Do it on CPU first
        checkpoint = load_checkpoint(config=config, device="cpu", epoch=load_epoch)

        # Restore optimizer state AFTER accelerator.prepare() so params are already on the right device
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Move optimizer state tensors to the correct device (moment buffers stay on CPU otherwise)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(
                        accelerator.device if accelerator is not None else device
                    )

        restore_schedulers(
            checkpoint=checkpoint,
            wd_scheduler=wd_scheduler,
            accelerator=accelerator,
        )

    patience = training_params.get("patience", float("inf"))
    num_epochs: int = training_params["num_epochs"]

    # Epochs after which best_val_loss and num_stale_epochs are reset to
    # float('inf') / 0 respectively.  The reset fires *after* validation
    # completes on the listed epoch, so epoch+1 is the first one tracked
    # under the new loss landscape.
    reset_best_val_loss_epochs: set[int] = set(
        training_params.get("reset_best_val_loss_epochs", [])
    )
    if reset_best_val_loss_epochs and should_save_and_log(accelerator):
        LOGGER.info(
            f"best_val_loss / patience will reset after validation at epochs: "
            f"{sorted(reset_best_val_loss_epochs)}"
        )

    if start_epoch > 0:
        _get_loss_module(dino_loss).resume_epoch(start_epoch)
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).resume_epoch(start_epoch)
        if gram_loss is not None:
            _get_loss_module(gram_loss).resume_epoch(start_epoch)
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).resume_epoch(start_epoch)

    # ------------------------------------------------------------------ #
    # Teacher momentum schedule                                          #
    # ------------------------------------------------------------------ #
    teacher_momentum_dict = config["training"].get("teacher_momentum")
    if teacher_momentum_dict is None:
        teacher_momentum_dict = {"initial_value": 0.996, "final_value": 1.0}

    momentum_epochs = teacher_momentum_dict.get("epochs", num_epochs)
    momentum_schedule = cosine_scheduler(
        base_value=teacher_momentum_dict["initial_value"],
        final_value=teacher_momentum_dict["final_value"],
        epochs=momentum_epochs,
        niter_per_ep=steps_per_epoch,
        warmup_epochs=0,
        start_warmup_value=teacher_momentum_dict["initial_value"],
    )

    should_stop = False

    # ------------------------------------------------------------------ #
    # Save untrained checkpoint                                          #
    # ------------------------------------------------------------------ #
    if should_save_and_log(accelerator) and start_epoch == 0:
        untrained_checkpoint = get_checkpoint_dict(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            student_ibot_head=student_ibot_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            teacher_ibot_head=teacher_ibot_head,
            ibot_pos_embedding=ibot_pos_embedding,
            dino_scale_embedding=dino_scale_embedding,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            epoch=-1,
            train_loss={"loss": float("inf")},
            val_loss={"loss": float("inf")},
            num_stale_epochs=0,
        )
        saved_path = save_checkpoint(
            checkpoint_dict=untrained_checkpoint,
            config=config,
            epoch_num="untrained",
        )
        LOGGER.info(f"Untrained model saved to {saved_path}")

    if num_epochs == 0:
        LOGGER.info(
            "Configured for 0 epochs of training. Exiting after saving untrained checkpoint."
        )
        return

    # ------------------------------------------------------------------ #
    # Main training loop                                                 #
    # ------------------------------------------------------------------ #
    global_step = start_epoch * len(train_dataloader)
    if wandb_run is not None and config["training"].get("load_epoch") is not None:
        # W&B's step from the resumed run is the ground truth; our epoch-based
        # estimate can be slightly off if the dataloader length changed between runs.
        wandb_step = getattr(wandb_run, "step", 0) or 0
        if wandb_step > global_step:
            LOGGER.info(
                f"Adjusting global_step from {global_step} to {wandb_step} "
                "to match resumed W&B run (avoids non-monotonic step warnings)"
            )
            global_step = wandb_step
    if start_epoch > 0:
        _get_loss_module(dino_loss).resume_step(global_step)
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).resume_step(global_step)
        if gram_loss is not None:
            _get_loss_module(gram_loss).resume_step(global_step)
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).resume_step(global_step)

    for epoch in range(start_epoch, num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]  # stem LR (most decayed with LLRD)
        if should_save_and_log(accelerator):
            LOGGER.info(f"Starting epoch {epoch} with learning rate: {current_lr:.8f}")

        epoch_start_momentum = momentum_schedule[
            min(global_step, len(momentum_schedule) - 1)
        ]
        LOGGER.info(
            f"Teacher momentum for epoch {epoch} (step {global_step}): {epoch_start_momentum:.6f}"
        )

        detect_anomaly = config["training"].get("autograd_detect_anomaly", False)
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            train_metrics, global_step = train_epoch(
                student_backbone=student_backbone,
                student_dino_head=student_dino_head,
                student_ibot_head=student_ibot_head,
                teacher_backbone=teacher_backbone,
                teacher_dino_head=teacher_dino_head,
                teacher_ibot_head=teacher_ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                dino_scale_embedding=dino_scale_embedding,
                dino_loss=dino_loss,
                ibot_loss=ibot_loss,
                gram_loss=gram_loss,
                koleo_loss=koleo_loss,
                optimizer=optimizer,
                dataloader=train_dataloader,
                device=device,
                config=config,
                augmenter=augmenter,
                momentum_schedule=momentum_schedule,
                use_bf16=use_bf16,
                accelerator=accelerator,
                wandb_run=wandb_run,
                global_step=global_step,
                scheduler=scheduler,
            )

        # Save intermediate checkpoint (before validation)
        if should_save_and_log(accelerator):
            checkpoint_dict = get_checkpoint_dict(
                student_backbone=student_backbone,
                student_dino_head=student_dino_head,
                student_ibot_head=student_ibot_head,
                teacher_backbone=teacher_backbone,
                teacher_dino_head=teacher_dino_head,
                teacher_ibot_head=teacher_ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                dino_scale_embedding=dino_scale_embedding,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                epoch=epoch,
                train_loss=train_metrics,
                val_loss={"loss": float("inf")},
                num_stale_epochs=num_stale_epochs,
            )
            save_checkpoint(
                checkpoint_dict=checkpoint_dict,
                config=config,
                epoch_num=epoch,
            )

        # Release train DataLoader worker prefetch queues before validation to
        # avoid holding both training and validation batches in RAM simultaneously.
        gc.collect()

        val_metrics = validate(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            student_ibot_head=student_ibot_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            teacher_ibot_head=teacher_ibot_head,
            ibot_pos_embedding=ibot_pos_embedding,
            dino_scale_embedding=dino_scale_embedding,
            dino_loss=dino_loss,
            ibot_loss=ibot_loss,
            gram_loss=gram_loss,
            koleo_loss=koleo_loss,
            dataloader=val_dataloader,
            device=device,
            config=config,
            augmenter=augmenter,
            use_bf16=use_bf16,
            accelerator=accelerator,
        )

        # ---------------------------------------------------------------- #
        # Step epoch-level loss schedulers                                 #
        # ---------------------------------------------------------------- #
        _get_loss_module(dino_loss).step_epoch()
        if ibot_loss is not None:
            _get_loss_module(ibot_loss).step_epoch()
        if gram_loss is not None:
            _get_loss_module(gram_loss).step_epoch()
        if koleo_loss is not None:
            _get_loss_module(koleo_loss).step_epoch()

        # Step weight decay scheduler
        if wd_scheduler is not None:
            current_wd = wd_scheduler.step()
            if should_save_and_log(accelerator):
                LOGGER.info(f"Weight decay after epoch {epoch}: {current_wd:.6f}")

        val_loss = val_metrics["loss"]
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # ---------------------------------------------------------------- #
        # Reset best_val_loss and patience if this epoch is a reset epoch  #
        # (fires after validation, so epoch+1 is tracked fresh)            #
        # ---------------------------------------------------------------- #
        if epoch in reset_best_val_loss_epochs:
            if should_save_and_log(accelerator):
                LOGGER.info(
                    f"[reset_best_val_loss] Epoch {epoch} is a designated reset epoch. "
                    f"Resetting best_val_loss ({best_val_loss:.8f} → inf) and "
                    f"num_stale_epochs ({num_stale_epochs} → 0). "
                    f"Epoch {epoch + 1} will be the first epoch tracked under the "
                    f"new loss landscape."
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
            # NOTE: logs stem LR (param_groups[0]), which is the most decayed
            # group under LLRD. Head LR = max(g["lr"] for g in param_groups).
            wandb_metrics = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            wandb_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            wandb_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            dino_mod = _get_loss_module(dino_loss)
            wandb_metrics["loss/dino_weight"] = dino_mod.get_current_weight()
            wandb_metrics["loss/dino_teacher_temp"] = (
                dino_mod.get_current_teacher_temp()
            )
            if ibot_loss is not None:
                ibot_mod = _get_loss_module(ibot_loss)
                wandb_metrics["loss/ibot_weight"] = ibot_mod.get_current_weight()
                wandb_metrics["loss/ibot_teacher_temp"] = (
                    ibot_mod.get_current_teacher_temp()
                )
            if gram_loss is not None:
                wandb_metrics["loss/gram_weight"] = _get_loss_module(
                    gram_loss
                ).get_current_weight()
            if koleo_loss is not None:
                wandb_metrics["loss/koleo_weight"] = _get_loss_module(
                    koleo_loss
                ).get_current_weight()
            wandb_run.log(wandb_metrics, step=global_step)

        # ---------------------------------------------------------------- #
        # Patience tracking & best checkpoint                              #
        # ---------------------------------------------------------------- #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_stale_epochs = 0

            if should_save_and_log(accelerator):
                checkpoint_dict = get_checkpoint_dict(
                    student_backbone=student_backbone,
                    student_dino_head=student_dino_head,
                    student_ibot_head=student_ibot_head,
                    teacher_backbone=teacher_backbone,
                    teacher_dino_head=teacher_dino_head,
                    teacher_ibot_head=teacher_ibot_head,
                    ibot_pos_embedding=ibot_pos_embedding,
                    dino_scale_embedding=dino_scale_embedding,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    epoch=epoch,
                    train_loss=train_metrics,
                    val_loss=val_metrics,
                    num_stale_epochs=num_stale_epochs,
                )
                saved_path = save_checkpoint(
                    checkpoint_dict=checkpoint_dict,
                    config=config,
                    epoch_num="best",
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

        # Save final checkpoint for this epoch (with updated num_stale_epochs)
        if should_save_and_log(accelerator):
            checkpoint_dict = get_checkpoint_dict(
                student_backbone=student_backbone,
                student_dino_head=student_dino_head,
                student_ibot_head=student_ibot_head,
                teacher_backbone=teacher_backbone,
                teacher_dino_head=teacher_dino_head,
                teacher_ibot_head=teacher_ibot_head,
                ibot_pos_embedding=ibot_pos_embedding,
                dino_scale_embedding=dino_scale_embedding,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                epoch=epoch,
                train_loss=train_metrics,
                val_loss=val_metrics,
                num_stale_epochs=num_stale_epochs,
            )
            saved_path = save_checkpoint(
                checkpoint_dict=checkpoint_dict,
                config=config,
                epoch_num=epoch,
            )
            LOGGER.info(
                f"Checkpoint (keys: {list(checkpoint_dict.keys())}) saved to {saved_path}"
            )

        if should_stop:
            if (
                (accelerator is not None)
                and (accelerator.num_processes > 1)
                and (accelerator.is_main_process)
            ):
                accelerator.set_trigger()
            else:
                break

        if accelerator is not None:
            if accelerator.check_trigger():
                break

    if should_save_and_log(accelerator):
        checkpoint_dict = get_checkpoint_dict(
            student_backbone=student_backbone,
            student_dino_head=student_dino_head,
            student_ibot_head=student_ibot_head,
            teacher_backbone=teacher_backbone,
            teacher_dino_head=teacher_dino_head,
            teacher_ibot_head=teacher_ibot_head,
            ibot_pos_embedding=ibot_pos_embedding,
            dino_scale_embedding=dino_scale_embedding,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            epoch=epoch,
            train_loss=train_metrics,
            val_loss=val_metrics,
            num_stale_epochs=num_stale_epochs,
        )
        saved_path = save_checkpoint(
            checkpoint_dict=checkpoint_dict,
            config=config,
            epoch_num="final",
        )
        LOGGER.info(f"Final model saved to {saved_path}")

    if accelerator is not None:
        accelerator.end_training()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINO model")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="DEBUG",
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
        name="RINO Training",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    try:
        train(config=config, use_wandb=args.use_wandb)
    except Exception as e:
        LOGGER.error(f"An error occurred during training: {e}")
        raise e

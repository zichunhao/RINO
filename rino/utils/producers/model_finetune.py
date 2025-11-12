from pathlib import Path

import torch

from models import (
    JetTransformerEncoder,
    Head,
    ParticleTransformer,
    ModelWithNewHead,
)
from ..ckpt import get_checkpoints_path, process_placeholder
from ..logger import LOGGER
from .model import _load_state
from utils.params import count_parameters


def get_models_finetune(
    part_dim: int,
    config: dict,
    device: str,
    mode: str = "training",
    train_head: bool = False,
) -> tuple[ModelWithNewHead, None]:
    """
    Get student and teacher models for DINO.
    Returns a tuple of (teacher) models, both using the same architecture.
    """
    # Create model
    model_type = config.get("model_type", "jet_transformer")
    compile_model = config.get("compile_model", False)
    if "jet" in model_type.lower():
        model = JetTransformerEncoder(
            part_dim=part_dim,
            batch_first=True,
            **config["model_params"],
        ).to(device)
        LOGGER.info(f"Model: {model}")

        head = Head(
            input_dim=config["model_params"]["d_model"],
            **config["head_params"],
        ).to(device)
        LOGGER.info(f"Head: {head}")

    else:
        LOGGER.info(f"Using ParticleTransformer model (input_dim = {part_dim})")
        model = ParticleTransformer(
            input_dim=part_dim,
            **config["model_params"],
        ).to(device)
        LOGGER.info(f"Model: {model}")

        rep_dim = config["model_params"]["embed_dims"][-1]
        head = Head(
            input_dim=rep_dim,
            **config["head_params"],
        ).to(device)

    # particle normalization
    if config.get("pre_aug_norm", False):
        LOGGER.warning("Pre-augmentation normalization is deprecated.")
        part_batch_norm = None
    else:
        part_batch_norm = None

    if compile_model:
        try:
            model_compiled = torch.compile(
                model=model,
                mode="default",
                fullgraph=False,  # True require that the entire function be capturable into a single graph
                dynamic=True,  # for variable input size
                backend="inductor",
            )
            head_compiled = torch.compile(
                model=head,
                mode="default",
                fullgraph=False,  # True require that the entire function be capturable into a single graph
                dynamic=True,  # for variable input size
                backend="inductor",
            )
            # both model and head are compiled
            model = model_compiled
            head = head_compiled
        except Exception as e:
            LOGGER.error(
                f"Failed to compile model: {e}; continuing with uncompiled model"
            )

    # Load state if specified
    if train_head:
        LOGGER.info("Loading backbone model and creating new head for finetuning")
        if mode.startswith("train"):
            _load_model_states(
                mode=mode,
                model=model,
                head=head,
                config=config,
            )

    model = ModelWithNewHead(
        backbone=model,
        head=head,
    )

    # Load ModelWithNewHead state dict if finetuned (has head_params)
    if not train_head:
        LOGGER.info("Loading ModelWithNewHead states for finetuned model")
        _load_model_with_new_head_states(
            model=model,
            config=config,
            mode=mode,
        )

    # Count parameters
    table, num_params = count_parameters(model)
    _, num_params_head = count_parameters(head)
    LOGGER.info(f"Model parameters:\n{table}")
    LOGGER.info(f"Total trainable parameters: {num_params}")
    LOGGER.info(f"Total trainable head parameters: {num_params_head}")
    LOGGER.info(f"Total trainable backbone parameters: {num_params - num_params_head}")

    return model, part_batch_norm


def _load_model_states(
    model: JetTransformerEncoder,
    head: Head,
    config: dict,
    mode: str = "training",
) -> None:
    """
    Load model states with three options:
    1. Load from specified weight path (finetuning from DINO pretrained)
        - Specify "backbone_weight_path" in config in training and/or inference section
        - "load_epoch" is ignored
    2. Load from epoch checkpoint (continue training)
        - Do not specify "backbone_weight_path" in config
        - Specify "load_epoch" in config in training and/or inference section
    3. No loading (train from scratch)
        - Do not specify "backbone_weight_path" or "load_epoch" in config
    """
    mode = mode.lower()

    # Option 1: Load from specified weight path
    if mode.startswith("train"):
        weight_path = config["training"].get("backbone_weight_path")
        load_epoch = config["training"].get("load_epoch")
    elif mode.startswith("inf"):
        weight_path = config["inference"].get("backbone_weight_path")
        load_epoch = config["inference"].get("load_epoch")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Try Option 1: Load from weight path
    if weight_path:
        LOGGER.info(f"Loading backbone weights from: {weight_path}")
        ckpt_path = Path(
            process_placeholder(s=weight_path, config=config, epoch_num=None)
        )
        try:
            state_dict = torch.load(ckpt_path)
            # copy weights to current checkpoint directory
            ckpt_path_finetune = get_checkpoints_path(
                config=config, epoch_num=load_epoch
            ).parent
            ckpt_path_finetune.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, ckpt_path_finetune / "backbone_weights.pt")

        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load backbone from teacher weights (DINO pretrained)
        _load_state(module=model, state_dict=state_dict, key="teacher")

        LOGGER.info(
            f"Loaded model and classification head weights from weight path {weight_path}"
        )
        return  # Successfully loaded from weight path

    # Option 2: Load from epoch checkpoint
    if load_epoch:
        LOGGER.info(f"Loading checkpoint from epoch: {load_epoch}")
        ckpt_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        try:
            state_dict = torch.load(ckpt_path)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load backbone
        _load_state(module=model, state_dict=state_dict, key="model")

        # Load classification head
        _load_state(module=head, state_dict=state_dict, key="classification_head")
        LOGGER.info(
            f"Loaded model and classification head weights from checkpoint {ckpt_path}"
        )
        return  # Successfully loaded from epoch checkpoint

    # Option 3: No loading (training from scratch)
    LOGGER.info("No checkpoint loaded. Training from scratch.")


def _load_model_with_new_head_states(
    model: ModelWithNewHead,
    config: dict,
    mode: str = "inference",
) -> None:
    """
    Load ModelWithNewHead states for finetuned models.
    This function is called when head_params EXISTS in config.
    """
    if mode.startswith("train"):
        load_epoch = config["training"].get("load_epoch")
    elif mode.startswith("inf"):
        load_epoch = config["inference"].get("load_epoch")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if load_epoch:
        # Load from epoch checkpoint (ModelWithNewHead format)
        LOGGER.info(
            f"Loading finetuned ModelWithNewHead checkpoint from epoch: {load_epoch}"
        )
        ckpt_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        LOGGER.info(f"Checkpoint path: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load ModelWithNewHead
        if "model" in state_dict:
            weight_dict = state_dict["model"]
        else:
            weight_dict = state_dict

        new_state_dict = {}  # remove module. prefix
        for k, v in weight_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        model.load_state_dict(new_state_dict)

        LOGGER.info(
            f"Loaded finetuned ModelWithNewHead checkpoint from epoch {load_epoch} for dataset: {mode}"
        )
        return
    else:
        # No loading specified, training from scratch
        LOGGER.info("No checkpoint loaded for ModelWithNewHead. Training from scratch.")

from typing import Literal
import torch
from torch import nn
from pathlib import Path

from models import JetTransformerEncoder, JetCLRTransformer, MLPHead, AssembledModel
from ..ckpt import get_checkpoints_path, process_placeholder
from ..logger import LOGGER
from ..params import count_parameters


def load_weight(model, state_dict):
    """Load state dict with automatic prefix handling."""
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as e:
        original_error = e

    # Check what prefixes exist in the state_dict
    prefixes_to_try = ["module.", "_orig_mod.", "_orig_mod.module."]

    checkpoint_keys = list(state_dict.keys())
    model_keys = list(model.state_dict().keys())

    # Detect if checkpoint has a prefix
    checkpoint_prefix = None
    for prefix in prefixes_to_try:
        if all(k.startswith(prefix) for k in checkpoint_keys):
            checkpoint_prefix = prefix
            break

    # Detect if model has a prefix
    model_prefix = None
    for prefix in prefixes_to_try:
        if all(k.startswith(prefix) for k in model_keys):
            model_prefix = prefix
            break

    # Decide what to do
    if checkpoint_prefix and not model_prefix:
        # Remove prefix from checkpoint
        new_state_dict = {k[len(checkpoint_prefix) :]: v for k, v in state_dict.items()}
        LOGGER.debug(f"Removing prefix '{checkpoint_prefix}' from checkpoint keys")
    elif model_prefix and not checkpoint_prefix:
        # Add prefix to checkpoint
        new_state_dict = {model_prefix + k: v for k, v in state_dict.items()}
        LOGGER.debug(f"Adding prefix '{model_prefix}' to checkpoint keys")
    else:
        # Can't determine what to do
        LOGGER.error(
            f"Failed to load state_dict. Sample keys from checkpoint: {checkpoint_keys[:5]}"
        )
        LOGGER.error(f"Sample keys expected by model: {model_keys[:5]}")
        raise original_error

    # Try loading with the adjusted state_dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
        LOGGER.debug("Successfully loaded state_dict after prefix adjustment")
    except RuntimeError as e:
        LOGGER.error(
            f"Failed to load state_dict after prefix adjustment. Sample keys from checkpoint: {checkpoint_keys[:5]}"
        )
        LOGGER.error(f"Sample keys expected by model: {model_keys[:5]}")
        raise original_error


def get_model(
    part_dim: int,
    config: dict,
    device: str | None = None,
    mode: Literal["training", "inference"] = "training",
    assemble: bool = True,
) -> AssembledModel | tuple[JetTransformerEncoder | JetCLRTransformer, MLPHead]:
    """
    Get backbone model with projection head for single-model SSL methods.
    Returns a tuple of (backbone, head)
    """
    if mode not in ["training", "inference"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'inference'.")
    if "models" not in config:
        raise ValueError("Config must contain a 'models' section")
    models_config = config["models"]
    if "backbone" not in models_config:
        raise ValueError("Config 'models' section must contain a 'backbone' subsection")
    if "head" not in models_config:
        raise ValueError("Config 'models' section must contain a 'head' subsection")

    # Backbone model
    backbone_config = models_config["backbone"]
    backbone_params = backbone_config.get("params", {})
    if mode == "training":
        use_penultimate_layer = False
        if config.get(mode, {}).get("penultimate_layer", False):
            LOGGER.warning(
                "penultimate_layer is set to True during training; "
                "this will be ignored and the last layer will be used"
            )
    else:  # inference
        use_penultimate_layer = config.get(mode, {}).get(
            "penultimate_layer", False
        )
        if use_penultimate_layer:
            LOGGER.info("Using backbone's penultimate layer for inference")
            backbone_params["use_penultimate_layer"] = True
        else:
            LOGGER.info("Using backbone's last layer for inference")

    compile_model = config.get("compile_model", False)
    finetune_embeddings = config.get(mode, {}).get("finetune_embeddings", False)

    # Create backbone
    backbone_type = backbone_config.get("type", "JetTransformerEncoder")
    if "jetclr" in backbone_type.lower():
        backbone = JetCLRTransformer(
            input_dim=part_dim,
            batch_first=True,
            **backbone_params,
        )
    else:
        backbone = JetTransformerEncoder(
            part_dim=part_dim,
            **backbone_params,
        )
    if device:
        backbone = backbone.to(device)

    table, params_backbone = count_parameters(backbone)
    LOGGER.info(f"Backbone Model Parameters:\n{table}")
    LOGGER.info(f"Total Backbone Parameters: {params_backbone}")

    # Get d_model from backbone for head input dimension
    backbone_d_model = backbone.rep_dim if hasattr(backbone, "rep_dim") else backbone.d_model

    # Create projection head
    head_config = models_config["head"]
    head_params = head_config.get("params", {})

    head = MLPHead(input_dim=backbone_d_model, **head_params)

    if device:
        head = head.to(device)

    table, params_head = count_parameters(head)
    LOGGER.info(f"Head Parameters:\n{table}")
    LOGGER.info(f"Total Head Parameters: {params_head}")

    # Load state if specified
    if _should_load_checkpoint(config, mode):
        _load_model_states(
            backbone=backbone,
            head=head,
            config=config,
            mode=mode,
        )

    # Apply finetune_embeddings logic AFTER loading states
    if finetune_embeddings:
        LOGGER.info("Finetuning mode: Freeze all parameters except embedding layers")

        # Freeze ALL parameters first
        for param in backbone.parameters():
            param.requires_grad = False

        # Then unfreeze only embedding layers
        backbone.unfreeze_embedding_layers()

    if compile_model:
        try:
            LOGGER.info("Compiling model")
            backbone_compiled = torch.compile(
                model=backbone,
                mode="default",
                fullgraph=False,
                dynamic=True,
                backend="inductor",
            )
            backbone = backbone_compiled
        except Exception as e:
            LOGGER.error(
                f"Failed to compile model: {e}; continuing without compilation"
            )

    # Log total parameters
    total_params = params_backbone + params_head
    LOGGER.info(f"Total Model Parameters: {total_params}")

    if assemble:
        LOGGER.info("Assembling backbone and head into AssembledModel")
        model = AssembledModel(backbone=backbone, heads=head)
        return model

    return backbone, head


def _should_load_checkpoint(config: dict, mode: str) -> bool:
    """
    Check if we should load a checkpoint based on load_path or load_epoch.
    """
    mode = mode.lower()
    if mode.startswith("train"):
        section = config.get("training", {})
    elif mode.startswith("inf"):
        section = config.get("inference", {})
    else:
        return False

    load_path = section.get("load_path")
    load_epoch = section.get("load_epoch")

    return load_path is not None or load_epoch is not None


def _load_model_states(
    backbone: JetTransformerEncoder,
    head: MLPHead,
    config: dict,
    mode: str = "training",
) -> None:
    mode = mode.lower()
    if mode.startswith("train"):
        section = config.get("training", {})
    elif mode.startswith("inf"):
        section = config.get("inference", {})
    else:
        raise ValueError(f"Invalid mode: {mode}")

    load_path = section.get("load_path")
    load_epoch = section.get("load_epoch")

    # Check for conflicts and determine which to use
    if load_path and load_epoch:
        LOGGER.warning(
            f"Both load_path ({load_path}) and load_epoch ({load_epoch}) are specified. "
            f"Using load_path and ignoring load_epoch."
        )

    # Determine checkpoint path
    if load_path:
        # Use load_path (with placeholder processing)
        checkpoint_path = Path(
            process_placeholder(s=load_path, config=config, epoch_num=None)
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at load_path: {checkpoint_path}"
            )
        LOGGER.info(f"Loading checkpoint from load_path: {checkpoint_path}")
    elif load_epoch:
        # Use load_epoch
        checkpoint_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at load_epoch {load_epoch}: {checkpoint_path}"
            )
        LOGGER.info(
            f"Loading checkpoint from load_epoch {load_epoch}: {checkpoint_path}"
        )
    else:
        # No loading specified
        LOGGER.info(
            "No load_path or load_epoch specified, model will use random initialization"
        )
        return

    # Load the checkpoint
    try:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    except RuntimeError as e:
        LOGGER.error(f"Failed to load checkpoint {checkpoint_path}: {e}. Trying cpu.")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load backbone state
    _load_state(module=backbone, state_dict=state_dict, key="backbone")

    # Load head state
    try:
        _load_state(module=head, state_dict=state_dict, key="head")
    except ValueError:
        LOGGER.warning("No head state found in checkpoint, keeping initialized weights")


def _load_state(module: nn.Module, state_dict: dict, key: str):
    """Load state for a specific module from checkpoint with robust prefix handling."""
    if key in state_dict:
        module_state_dict = state_dict[key]
    else:
        raise ValueError(f"Checkpoint file does not contain the {key}")

    # Use the robust load_weight function
    load_weight(module, module_state_dict)
    LOGGER.info(f"Loaded {key} state from checkpoint")


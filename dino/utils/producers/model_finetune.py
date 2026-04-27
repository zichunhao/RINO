from pathlib import Path

import torch

from models import (
    JetTransformerEncoder,
    OldJetTransformerEncoder,
    ParticleTransformer,
    MLPHead,
    AssembledModel,
)
from models.head import DeepMLPHead, CrossAttentionHead
from ..ckpt import get_checkpoints_path, process_placeholder
from ..logger import LOGGER
from .model import _load_state, load_weight
from utils.params import count_parameters


def get_models_finetune(
    part_dim: int,
    config: dict,
    device: str | None = None,
    mode: str = "training",
    train_head: bool = False,
) -> AssembledModel:
    """
    Get models for finetuning.

    Args:
        part_dim: Dimension of particle features
        config: Configuration dictionary
        device: Device to place models on
        mode: 'training' or 'inference'
        train_head: If True, load backbone and create new head. If False, load full AssembledModel

    Returns:
        AssembledModel
    """
    if mode not in ["training", "inference"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'inference'.")
    if "models" not in config:
        raise ValueError("Config must contain a 'models' section")

    models_config = config["models"]

    # Backbone model
    if "backbone" not in models_config:
        raise ValueError("Config 'models' section must contain a 'backbone' subsection")

    backbone_config = models_config["backbone"]
    backbone_type = backbone_config.get("type", "JetTransformerEncoder")
    backbone_params = backbone_config.get("params", {})

    if "jet" in backbone_type.lower():
        LOGGER.info("Using JetTransformer backbone")
        if "old" in backbone_type.lower():
            LOGGER.info("Using OLD JetTransformer model")
            backbone_class = OldJetTransformerEncoder
            extra_params = {"batch_first": True}
        else:
            backbone_class = JetTransformerEncoder
            extra_params = {}
        backbone = backbone_class(
            part_dim=part_dim,
            **backbone_params,
            **extra_params,
        )
        if device:
            backbone = backbone.to(device)
    elif "particle" in backbone_type.lower():
        LOGGER.info("Using ParticleTransformer backbone")
        backbone = ParticleTransformer(
            input_dim=part_dim,
            **backbone_params,
        )
        if device:
            backbone = backbone.to(device)
    else:
        raise NotImplementedError(f"Backbone type {backbone_type} not implemented")

    table, params_backbone = count_parameters(backbone)
    LOGGER.info(f"Backbone Model Parameters:\n{table}")
    LOGGER.info(f"Total Backbone Parameters: {params_backbone}")

    # Head model
    if "head" not in models_config:
        raise ValueError("Config 'models' section must contain a 'head' subsection")

    head_config = models_config["head"]
    head_type = head_config.get("type", "MLPHead").lower()
    head_params = head_config.get("params", {})

    # Determine input dimension based on backbone type
    head_input_dim = (
        backbone.rep_dim if hasattr(backbone, "rep_dim") else backbone.d_model
    )

    if "cross_attention" in head_type or "crossattention" in head_type:
        head = CrossAttentionHead(
            d_model=head_input_dim,
            **head_params,
        )
    elif "deep" in head_type:
        head = DeepMLPHead(
            input_dim=head_input_dim,
            **head_params,
        )
    elif "mlp" in head_type:
        head = MLPHead(
            input_dim=head_input_dim,
            **head_params,
        )
    else:
        raise NotImplementedError(f"Head type {head_type} not implemented")

    if device:
        head = head.to(device)

    table, params_head = count_parameters(head)
    LOGGER.info(f"Head Model Parameters:\n{table}")
    LOGGER.info(f"Total Head Parameters: {params_head}")

    # Create assembled model
    assembled_model = AssembledModel(
        embedding=None,  # No separate embedding in DINO
        backbone=backbone,
        heads=head,
    )

    # Load state if specified
    if train_head:
        LOGGER.info("Loading backbone model and creating new head for finetuning")
        _load_model_states(
            backbone=backbone,
            head=head,
            config=config,
            mode=mode,
        )
    else:
        # Load full AssembledModel state dict for finetuned model
        LOGGER.info("Loading AssembledModel states for finetuned model")
        _load_assembled_model_states(
            model=assembled_model,
            config=config,
            mode=mode,
        )

    # Count parameters for assembled model
    table, num_params = count_parameters(assembled_model)

    LOGGER.info(f"Assembled Model parameters:\n{table}")
    LOGGER.info(f"Total trainable parameters: {num_params}")
    LOGGER.info(f"  - Backbone parameters: {params_backbone}")
    LOGGER.info(f"  - Head parameters: {params_head}")

    return assembled_model


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
    backbone_weight_path = section.get("backbone_weight_path")
    load_epoch = section.get("load_epoch")

    return (
        load_path is not None
        or backbone_weight_path is not None
        or load_epoch is not None
    )


def _load_model_states(
    backbone: JetTransformerEncoder | ParticleTransformer,
    head: MLPHead,
    config: dict,
    mode: str = "training",
) -> None:
    """
    Load model states with multiple options:
    1. Load from specified weight path (finetuning from DINO pretrained)
        - Specify "backbone_weight_path" in config in training and/or inference section
        - "load_epoch" is ignored if backbone_weight_path is specified
    2. Load from specified load_path (general checkpoint)
        - Specify "load_path" in config
    3. Load from epoch checkpoint (continue training)
        - Do not specify "backbone_weight_path" or "load_path" in config
        - Specify "load_epoch" in config in training and/or inference section
    4. No loading (train from scratch)
        - Do not specify any of the above in config
    """
    mode = mode.lower()

    # Determine which section to use
    if mode.startswith("train"):
        section = config.get("training", {})
    elif mode.startswith("inf"):
        section = config.get("inference", {})
    else:
        raise ValueError(f"Invalid mode: {mode}")

    load_path = section.get("load_path")
    backbone_weight_path = section.get("backbone_weight_path")
    load_epoch = section.get("load_epoch")

    # Check for conflicts and determine which to use
    if backbone_weight_path and load_path:
        LOGGER.warning(
            f"Both backbone_weight_path ({backbone_weight_path}) and load_path ({load_path}) are specified. "
            f"Using backbone_weight_path and ignoring load_path."
        )

    # Option 1: Load from specified backbone weight path
    if backbone_weight_path:
        LOGGER.info(f"Loading backbone weights from: {backbone_weight_path}")
        ckpt_path = Path(
            process_placeholder(s=backbone_weight_path, config=config, epoch_num=None)
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        try:
            state_dict = torch.load(ckpt_path, weights_only=False)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load backbone: try DINO teacher first, then JetCLR/single-model keys.
        # Filter out unexpected keys (e.g. pos_encoding from pretraining) to allow
        # loading into a backbone without positional encoding.
        # Also merge ibot_pos_embedding into the backbone state dict if the backbone
        # has pos_encoding.* keys (DINO saves PE separately, finetune has it in backbone).
        model_keys = set(backbone.state_dict().keys())
        loaded_key = None
        for candidate in ("teacher", "backbone", "student"):
            if candidate in state_dict:
                try:
                    candidate_sd = dict(state_dict[candidate])
                    # Merge ibot_pos_embedding into backbone state_dict
                    # DINO saves PE as a separate module; finetune backbone has it as pos_encoding.*
                    if "ibot_pos_embedding" in state_dict and state_dict["ibot_pos_embedding"] is not None:
                        pe_keys_in_model = [k for k in model_keys if k.startswith("pos_encoding.")]
                        if pe_keys_in_model:
                            for pe_key, pe_val in state_dict["ibot_pos_embedding"].items():
                                full_key = f"pos_encoding.{pe_key}"
                                if full_key in model_keys:
                                    candidate_sd[full_key] = pe_val
                                    LOGGER.info(f"Merged ibot_pos_embedding key '{pe_key}' -> '{full_key}'")
                    unexpected = [k for k in candidate_sd if k not in model_keys]
                    if unexpected:
                        LOGGER.warning(
                            f"Ignoring {len(unexpected)} unexpected key(s) from '{candidate}' "
                            f"checkpoint: {unexpected[:5]}"
                        )
                        candidate_sd = {k: v for k, v in candidate_sd.items() if k in model_keys}
                    missing = [k for k in model_keys if k not in candidate_sd]
                    if missing:
                        LOGGER.warning(
                            f"Missing {len(missing)} key(s) in '{candidate}' checkpoint "
                            f"(will use random init): {missing[:5]}"
                        )
                        # Load with strict=False to allow missing keys
                        backbone.load_state_dict(candidate_sd, strict=False)
                        loaded_key = candidate
                        break
                    state_dict_filtered = dict(state_dict)
                    state_dict_filtered[candidate] = candidate_sd
                    _load_state(module=backbone, state_dict=state_dict_filtered, key=candidate)
                    loaded_key = candidate
                    break
                except (ValueError, RuntimeError) as e:
                    LOGGER.warning(f"Failed loading backbone from key '{candidate}': {e}")
        if loaded_key is None:
            raise ValueError(
                f"No backbone weights found in {ckpt_path} under keys "
                f"'teacher'/'backbone'/'student'"
            )

        LOGGER.info(
            f"Loaded backbone weights from backbone_weight_path {backbone_weight_path} "
            f"(key='{loaded_key}')"
        )
        return  # Successfully loaded from backbone weight path

    # Option 2: Load from specified load_path
    if load_path:
        LOGGER.info(f"Loading checkpoint from load_path: {load_path}")
        ckpt_path = Path(
            process_placeholder(s=load_path, config=config, epoch_num=None)
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        try:
            state_dict = torch.load(ckpt_path, weights_only=False)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load backbone (could be from "backbone" key)
        try:
            _load_state(module=backbone, state_dict=state_dict, key="backbone")
        except ValueError:
            LOGGER.warning(
                "No backbone state found in checkpoint, keeping initialized weights"
            )

        # Load head if available
        try:
            _load_state(module=head, state_dict=state_dict, key="head")
        except ValueError:
            LOGGER.warning(
                "No head state found in checkpoint, keeping initialized weights"
            )

        LOGGER.info(f"Loaded checkpoint from load_path {load_path}")
        return  # Successfully loaded from load_path

    # Option 3: Load from epoch checkpoint
    if load_epoch is not None:
        LOGGER.info(f"Loading checkpoint from epoch: {load_epoch}")
        ckpt_path = get_checkpoints_path(config=config, epoch_num=load_epoch)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        try:
            state_dict = torch.load(ckpt_path, weights_only=False)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load backbone
        try:
            _load_state(module=backbone, state_dict=state_dict, key="backbone")
        except ValueError:
            LOGGER.warning(
                "No backbone state found in checkpoint, keeping initialized weights"
            )

        # Load head
        try:
            _load_state(module=head, state_dict=state_dict, key="head")
        except ValueError:
            LOGGER.warning(
                "No head state found in checkpoint, keeping initialized weights"
            )

        LOGGER.info(f"Loaded model and head weights from checkpoint {ckpt_path}")
        return  # Successfully loaded from epoch checkpoint

    # Option 4: No loading (training from scratch)
    LOGGER.info(
        "No load_path, backbone_weight_path, or load_epoch specified, models will use random initialization"
    )


def _load_assembled_model_states(
    model: AssembledModel,
    config: dict,
    mode: str = "inference",
) -> None:
    """
    Load AssembledModel states for finetuned models.
    This function is called when the full finetuned model needs to be loaded.
    """
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
        LOGGER.info(
            f"Loading AssembledModel checkpoint from load_path: {checkpoint_path}"
        )
    elif load_epoch is not None:
        # Use load_epoch
        checkpoint_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at load_epoch {load_epoch}: {checkpoint_path}"
            )
        LOGGER.info(
            f"Loading AssembledModel checkpoint from load_epoch {load_epoch}: {checkpoint_path}"
        )
    else:
        # No loading specified
        LOGGER.info(
            "No load_path or load_epoch specified for AssembledModel, keeping initialized weights"
        )
        return

    # Load the checkpoint
    try:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    except RuntimeError as e:
        LOGGER.error(f"Failed to load checkpoint {checkpoint_path}: {e}. Trying cpu.")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load AssembledModel
    if "model" in state_dict:
        weight_dict = state_dict["model"]
    else:
        weight_dict = state_dict

    # Use the robust load_weight function to handle prefix issues
    load_weight(model, weight_dict)

    LOGGER.info(f"Loaded finetuned AssembledModel checkpoint from {checkpoint_path}")

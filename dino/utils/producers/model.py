from typing import Literal
import torch
from torch import nn
from pathlib import Path

from models import (
    JetTransformerEncoder,
    OldJetTransformerEncoder,
    ParticleTransformer,
    DINOHead,
    MLPHead,
    PositionalEncoding,
    ScaleConditioning,
)
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


def get_models(
    part_dim: int,
    config: dict,
    device: str | None = None,
    mode: Literal["training", "inference"] = "training",
) -> tuple[
    tuple[
        JetTransformerEncoder | ParticleTransformer,
        DINOHead | MLPHead,
        DINOHead | MLPHead | None,
    ],
    tuple[
        JetTransformerEncoder | ParticleTransformer,
        DINOHead | MLPHead,
        DINOHead | MLPHead | None,
    ],
    PositionalEncoding | None,
]:
    """
    Get student and teacher models for DINO with separate heads.
    Returns a tuple of ((student_backbone, student_dino_head, student_ibot_head),
                        (teacher_backbone, teacher_dino_head, teacher_ibot_head),
                        ibot_pos_embedding)
    """
    if mode not in ["training", "inference"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'inference'.")
    if "models" not in config:
        raise ValueError("Config must contain a 'models' section")
    models_config = config["models"]
    if "backbone" not in models_config:
        raise ValueError("Config 'models' section must contain a 'backbone' subsection")
    if "dino_head" not in models_config:
        raise ValueError(
            "Config 'models' section must contain a 'dino_head' subsection"
        )

    # Backbone model
    backbone_config = models_config["backbone"]
    backbone_type = backbone_config.get("type", "JetTransformerEncoder")
    backbone_params = backbone_config.get("params", {})
    rep_opts = config[mode].get("rep", {})
    if mode == "training":
        use_penultimate_layer = False
        if rep_opts.get("penultimate_layer", False):
            LOGGER.warning(
                "penultimate_layer is set to True during training; "
                "this will be ignored and the last layer will be used"
            )
    else:  # inference
        use_penultimate_layer = rep_opts.get("penultimate_layer", False)
        if use_penultimate_layer:
            LOGGER.info("Using backbone's penultimate layer for inference")
            backbone_params["use_penultimate_layer"] = True
        else:
            LOGGER.info("Using backbone's last layer for inference")

    compile_model = config.get("compile_model", False)

    if "jet" in backbone_type.lower():
        if "old" in backbone_type.lower():
            backbone_class = OldJetTransformerEncoder
            extra_params = {"batch_first": True}
        else:
            backbone_class = JetTransformerEncoder
            extra_params = {}

        LOGGER.info("Using JetTransformer model")
        # Create student model
        student_backbone = backbone_class(
            part_dim=part_dim,
            **backbone_params,
            **extra_params,
        )
        if device:
            student_backbone = student_backbone.to(device)

        # Create teacher model with same architecture
        teacher_backbone = backbone_class(
            part_dim=part_dim,
            **backbone_params,
            **extra_params,
        )
        if device:
            teacher_backbone = teacher_backbone.to(device)
    elif "particle" in backbone_type.lower():
        LOGGER.info(f"Using ParticleTransformer model (input_dim = {part_dim})")
        student_backbone = ParticleTransformer(
            input_dim=part_dim,
            **backbone_params,
        )
        if device:
            student_backbone = student_backbone.to(device)

        teacher_backbone = ParticleTransformer(
            input_dim=part_dim,
            **backbone_params,
        )
        if device:
            teacher_backbone = teacher_backbone.to(device)
    else:
        raise NotImplementedError(f"Backbone type {backbone_type} not implemented")
    LOGGER.info(f"Backbone model: {teacher_backbone}")

    table, params_backbone = count_parameters(student_backbone)
    LOGGER.info(f"Backbone Model Parameters:\n{table}")
    LOGGER.info(f"Total Backbone Parameters: {params_backbone}")

    # Get d_model from backbone for head input dimension
    backbone_d_model = (
        student_backbone.rep_dim
        if hasattr(student_backbone, "rep_dim")
        else student_backbone.d_model
    )

    # DINO head
    dino_head_config = models_config["dino_head"]
    dino_head_type = dino_head_config.get("type", "DINOHead").lower()
    dino_head_params = dino_head_config.get("params", {})

    if "dino" in dino_head_type:
        student_dino_head = DINOHead(in_dim=backbone_d_model, **dino_head_params)
        teacher_dino_head = DINOHead(in_dim=backbone_d_model, **dino_head_params)
    elif "mlp" in dino_head_type:
        student_dino_head = MLPHead(input_dim=backbone_d_model, **dino_head_params)
        teacher_dino_head = MLPHead(input_dim=backbone_d_model, **dino_head_params)
    else:
        raise NotImplementedError(f"DINO head type {dino_head_type} not implemented")
    LOGGER.info(f"Dino head: {student_dino_head}")

    if device:
        student_dino_head = student_dino_head.to(device)
        teacher_dino_head = teacher_dino_head.to(device)

    table, params_dino_head = count_parameters(student_dino_head)
    LOGGER.info(f"DINO Head Parameters:\n{table}")
    LOGGER.info(f"Total DINO Head Parameters: {params_dino_head}")

    # iBOT head (optional)
    if "ibot_head" not in models_config:
        student_ibot_head = None
        teacher_ibot_head = None
        LOGGER.info("No iBOT head specified in config")
    else:
        ibot_head_config = models_config["ibot_head"]
        ibot_head_type = ibot_head_config.get("type", "DINOHead").lower()
        ibot_head_params = ibot_head_config.get("params", {})
        ibot_in_dim = student_backbone.d_model

        if "dino" in ibot_head_type:
            student_ibot_head = DINOHead(in_dim=ibot_in_dim, **ibot_head_params)
            teacher_ibot_head = DINOHead(in_dim=ibot_in_dim, **ibot_head_params)
        elif "mlp" in ibot_head_type:
            student_ibot_head = MLPHead(in_dim=ibot_in_dim, **ibot_head_params)
            teacher_ibot_head = MLPHead(in_dim=ibot_in_dim, **ibot_head_params)
        else:
            raise NotImplementedError(
                f"iBOT head type {ibot_head_type} not implemented"
            )

        if device:
            student_ibot_head = student_ibot_head.to(device)
            teacher_ibot_head = teacher_ibot_head.to(device)

        table, params_ibot_head = count_parameters(student_ibot_head)

        LOGGER.info(f"iBOT head: {student_ibot_head}")

        LOGGER.info(f"iBOT Head Parameters:\n{table}")
        LOGGER.info(f"Total iBOT Head Parameters: {params_ibot_head}")

    # Handle finetune_embeddings BEFORE loading states
    finetune_embeddings = config.get("training", {}).get("finetune_embeddings", False)

    # DINO scale embedding (optional, post-backbone RG-scale conditioning)
    dino_scale_embedding = None
    if "dino_scale_embedding" in models_config:
        scale_emb_params = models_config["dino_scale_embedding"].get("params", {})
        dino_scale_embedding = ScaleConditioning(
            d_model=backbone_d_model,
            **scale_emb_params,
        )
        if device:
            dino_scale_embedding = dino_scale_embedding.to(device)
        LOGGER.info(
            f"DINO scale embedding: ScaleConditioning("
            f"d_model={backbone_d_model}, params={scale_emb_params})"
        )

    # iBOT positional embedding (optional, post-backbone coordinate encoder)
    ibot_pos_embedding = None
    if "ibot_pos_embedding" in models_config:

        pos_emb_config = models_config["ibot_pos_embedding"]
        pos_emb_params = pos_emb_config.get("params", {})
        input_indices = pos_emb_params.get("input_indices", None)

        # Derive dimensions automatically
        if input_indices is None:
            in_features = 0
        else:
            in_features = len(input_indices)
        out_features = backbone_d_model  # must match d_model for additive injection

        ibot_pos_embedding = PositionalEncoding(
            out_features=out_features,
            **pos_emb_params,
        )

        if device:
            ibot_pos_embedding = ibot_pos_embedding.to(device)

        LOGGER.info(
            f"iBOT positional embedding: PositionalEncoding("
            f"in_features={in_features}, out_features={out_features}, "
            f"params={pos_emb_params})"
        )

    # Load state if specified (both training and inference)
    _load_model_states(
        mode=mode,
        student_backbone=student_backbone,
        student_dino_head=student_dino_head,
        student_ibot_head=student_ibot_head,
        teacher_backbone=teacher_backbone,
        teacher_dino_head=teacher_dino_head,
        teacher_ibot_head=teacher_ibot_head,
        ibot_pos_embedding=ibot_pos_embedding,
        dino_scale_embedding=dino_scale_embedding,
        config=config,
    )

    # If no state is loaded, teacher starts with same weights as student
    if not _should_load_checkpoint(config, mode):
        LOGGER.info("No checkpoint loaded; copying student weights to teacher")
        load_weight(teacher_backbone, student_backbone.state_dict())
        load_weight(teacher_dino_head, student_dino_head.state_dict())
        if teacher_ibot_head is not None and student_ibot_head is not None:
            load_weight(teacher_ibot_head, student_ibot_head.state_dict())

    # Teacher's parameters should not require gradients
    for param in teacher_backbone.parameters():
        param.requires_grad = False
    for param in teacher_dino_head.parameters():
        param.requires_grad = False
    if teacher_ibot_head is not None:
        for param in teacher_ibot_head.parameters():
            param.requires_grad = False

    # Apply finetune_embeddings logic AFTER loading states and setting up teacher
    if finetune_embeddings:
        LOGGER.info("Finetuning mode: Freeze all parameters except embedding layers")

        # Freeze ALL student parameters first
        for param in student_backbone.parameters():
            param.requires_grad = False

        # Then unfreeze only embedding layers
        student_backbone.unfreeze_embedding_layers()

    if compile_model:
        try:
            LOGGER.info("Compiling student and teacher models")
            student_backbone_compiled = torch.compile(
                model=student_backbone,
                mode="default",
                fullgraph=False,
                dynamic=True,
                backend="inductor",
            )
            teacher_backbone_compiled = torch.compile(
                model=teacher_backbone,
                mode="default",
                fullgraph=False,
                dynamic=True,
                backend="inductor",
            )
            # Replace with compiled versions
            student_backbone = student_backbone_compiled
            teacher_backbone = teacher_backbone_compiled
        except Exception as e:
            LOGGER.error(
                f"Failed to compile models: {e}; continuing without compilation"
            )

    # Log total parameters
    total_params = params_backbone + params_dino_head
    if student_ibot_head is not None:
        total_params += params_ibot_head
    LOGGER.info(f"Total Model Parameters: {total_params}")

    return (
        (student_backbone, student_dino_head, student_ibot_head),
        (teacher_backbone, teacher_dino_head, teacher_ibot_head),
        ibot_pos_embedding,
        dino_scale_embedding,
    )


def get_models_single(
    part_dim: int,
    config: dict,
    device: str | None = None,
    mode: Literal["training", "inference"] = "training",
) -> tuple[
    JetTransformerEncoder | ParticleTransformer,
    DINOHead | MLPHead,
    DINOHead | MLPHead | None,
    PositionalEncoding | None,
]:
    """Get a single model (no teacher) for contrastive training (JetCLR).

    Returns (backbone, projection_head, ibot_head, ibot_pos_embedding).
    ibot_head and ibot_pos_embedding are None if not configured.
    """
    if mode not in ["training", "inference"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'training' or 'inference'.")
    if "models" not in config:
        raise ValueError("Config must contain a 'models' section")
    models_config = config["models"]
    if "backbone" not in models_config:
        raise ValueError("Config 'models' section must contain a 'backbone' subsection")
    if "projection_head" not in models_config and "dino_head" not in models_config:
        raise ValueError(
            "Config 'models' section must contain a 'projection_head' or 'dino_head' subsection"
        )

    # Backbone
    backbone_config = models_config["backbone"]
    backbone_type = backbone_config.get("type", "JetTransformerEncoder")
    backbone_params = backbone_config.get("params", {})
    rep_opts = config[mode].get("rep", {})
    if mode != "training":
        use_penultimate_layer = rep_opts.get("penultimate_layer", False)
        if use_penultimate_layer:
            LOGGER.info("Using backbone's penultimate layer for inference")
            backbone_params["use_penultimate_layer"] = True

    compile_model = config.get("compile_model", False)

    if "jet" in backbone_type.lower():
        if "old" in backbone_type.lower():
            backbone_class = OldJetTransformerEncoder
            extra_params = {"batch_first": True}
        else:
            backbone_class = JetTransformerEncoder
            extra_params = {}

        LOGGER.info("Using JetTransformer model")
        backbone = backbone_class(part_dim=part_dim, **backbone_params, **extra_params)
    elif "particle" in backbone_type.lower():
        LOGGER.info(f"Using ParticleTransformer model (input_dim = {part_dim})")
        backbone = ParticleTransformer(input_dim=part_dim, **backbone_params)
    else:
        raise NotImplementedError(f"Backbone type {backbone_type} not implemented")

    if device:
        backbone = backbone.to(device)
    LOGGER.info(f"Backbone model: {backbone}")

    table, params_backbone = count_parameters(backbone)
    LOGGER.info(f"Backbone Model Parameters:\n{table}")
    LOGGER.info(f"Total Backbone Parameters: {params_backbone}")

    backbone_d_model = (
        backbone.rep_dim if hasattr(backbone, "rep_dim") else backbone.d_model
    )

    # Projection head (accept either "projection_head" or "dino_head" key)
    head_key = "projection_head" if "projection_head" in models_config else "dino_head"
    head_config = models_config[head_key]
    head_type = head_config.get("type", "DINOHead").lower()
    head_params = head_config.get("params", {})

    if "dino" in head_type:
        projection_head = DINOHead(in_dim=backbone_d_model, **head_params)
    elif "mlp" in head_type:
        projection_head = MLPHead(input_dim=backbone_d_model, **head_params)
    else:
        raise NotImplementedError(f"Projection head type {head_type} not implemented")
    LOGGER.info(f"Projection head: {projection_head}")

    if device:
        projection_head = projection_head.to(device)

    table, params_head = count_parameters(projection_head)
    LOGGER.info(f"Projection Head Parameters:\n{table}")

    # iBOT head (optional)
    ibot_head = None
    if "ibot_head" in models_config:
        ibot_head_config = models_config["ibot_head"]
        ibot_head_type = ibot_head_config.get("type", "DINOHead").lower()
        ibot_head_params = ibot_head_config.get("params", {})
        ibot_in_dim = backbone.d_model

        if "dino" in ibot_head_type:
            ibot_head = DINOHead(in_dim=ibot_in_dim, **ibot_head_params)
        elif "mlp" in ibot_head_type:
            ibot_head = MLPHead(in_dim=ibot_in_dim, **ibot_head_params)
        else:
            raise NotImplementedError(
                f"iBOT head type {ibot_head_type} not implemented"
            )
        if device:
            ibot_head = ibot_head.to(device)

        table, params_ibot = count_parameters(ibot_head)
        LOGGER.info(f"iBOT head: {ibot_head}")
        LOGGER.info(f"iBOT Head Parameters:\n{table}")

    # iBOT positional embedding (optional)
    ibot_pos_embedding = None
    if "ibot_pos_embedding" in models_config:
        pos_emb_config = models_config["ibot_pos_embedding"]
        pos_emb_params = pos_emb_config.get("params", {})
        input_indices = pos_emb_params.get("input_indices", None)
        out_features = backbone.d_model

        ibot_pos_embedding = PositionalEncoding(out_features=out_features, **pos_emb_params)
        if device:
            ibot_pos_embedding = ibot_pos_embedding.to(device)
        LOGGER.info(
            f"iBOT positional embedding: PositionalEncoding("
            f"in_features={len(input_indices) if input_indices else 0}, "
            f"out_features={out_features}, params={pos_emb_params})"
        )

    # Load checkpoint if specified
    _load_single_model_states(
        backbone=backbone,
        projection_head=projection_head,
        ibot_head=ibot_head,
        ibot_pos_embedding=ibot_pos_embedding,
        config=config,
        mode=mode,
    )

    if compile_model:
        try:
            LOGGER.info("Compiling backbone model")
            backbone = torch.compile(
                model=backbone, mode="default", fullgraph=False,
                dynamic=True, backend="inductor",
            )
        except Exception as e:
            LOGGER.error(f"Failed to compile model: {e}; continuing without compilation")

    total_params = params_backbone + params_head
    if ibot_head is not None:
        total_params += params_ibot
    LOGGER.info(f"Total Model Parameters: {total_params}")

    return backbone, projection_head, ibot_head, ibot_pos_embedding


def _load_single_model_states(
    backbone: nn.Module,
    projection_head: nn.Module,
    ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    config: dict,
    mode: str = "training",
) -> None:
    """Load checkpoint for single-model (JetCLR) setup."""
    mode = mode.lower()
    if mode.startswith("train"):
        section = config.get("training", {})
    elif mode.startswith("inf"):
        section = config.get("inference", {})
    else:
        raise ValueError(f"Invalid mode: {mode}")

    load_path = section.get("load_path")
    load_epoch = section.get("load_epoch")

    if load_path is not None and load_epoch is not None:
        LOGGER.warning(
            f"Both load_path ({load_path}) and load_epoch ({load_epoch}) are specified. "
            f"Using load_path and ignoring load_epoch."
        )

    if load_path is not None:
        from ..ckpt import process_placeholder
        checkpoint_path = Path(process_placeholder(s=load_path, config=config, epoch_num=None))
    elif load_epoch is not None:
        from ..ckpt import get_checkpoints_path
        checkpoint_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
    else:
        LOGGER.info("No load_path or load_epoch specified, using random initialization")
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    LOGGER.info(f"Loading checkpoint from: {checkpoint_path}")

    try:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    except RuntimeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try "backbone" key first (JetCLR format), then "student" (DINO format)
    for key in ("backbone", "student"):
        if key in state_dict:
            load_weight(backbone, state_dict[key])
            LOGGER.info(f"Loaded backbone state from checkpoint key '{key}'")
            break
    else:
        LOGGER.warning("No backbone/student state found in checkpoint")

    for key in ("projection_head", "student_dino_head"):
        if key in state_dict:
            load_weight(projection_head, state_dict[key])
            LOGGER.info(f"Loaded projection_head state from checkpoint key '{key}'")
            break
    else:
        LOGGER.warning("No projection_head state found in checkpoint")

    if ibot_head is not None:
        for key in ("ibot_head", "student_ibot_head"):
            if key in state_dict:
                load_weight(ibot_head, state_dict[key])
                LOGGER.info(f"Loaded ibot_head state from checkpoint key '{key}'")
                break
        else:
            LOGGER.warning("No ibot_head state found in checkpoint")

    if ibot_pos_embedding is not None:
        if "ibot_pos_embedding" in state_dict:
            load_weight(ibot_pos_embedding, state_dict["ibot_pos_embedding"])
            LOGGER.info("Loaded ibot_pos_embedding state from checkpoint")
        else:
            LOGGER.warning("ibot_pos_embedding not found in checkpoint")


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
    student_backbone: JetTransformerEncoder | ParticleTransformer,
    student_dino_head: DINOHead | MLPHead,
    student_ibot_head: DINOHead | MLPHead | None,
    teacher_backbone: JetTransformerEncoder | ParticleTransformer,
    teacher_dino_head: DINOHead | MLPHead,
    teacher_ibot_head: DINOHead | MLPHead | None,
    config: dict,
    ibot_pos_embedding: nn.Module | None = None,
    dino_scale_embedding: nn.Module | None = None,
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
    if load_path is not None and load_epoch is not None:
        LOGGER.warning(
            f"Both load_path ({load_path}) and load_epoch ({load_epoch}) are specified. "
            f"Using load_path and ignoring load_epoch."
        )

    # Determine checkpoint path
    if load_path is not None:
        checkpoint_path = Path(
            process_placeholder(s=load_path, config=config, epoch_num=None)
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at load_path: {checkpoint_path}"
            )
        LOGGER.info(f"Loading checkpoint from load_path: {checkpoint_path}")
    elif load_epoch is not None:
        checkpoint_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at load_epoch {load_epoch}: {checkpoint_path}"
            )
        LOGGER.info(
            f"Loading checkpoint from load_epoch {load_epoch}: {checkpoint_path}"
        )
    else:
        LOGGER.info(
            "No load_path or load_epoch specified, models will use random initialization"
        )
        return

    # Load the checkpoint
    try:
        state_dict = torch.load(checkpoint_path, weights_only=False)
    except RuntimeError as e:
        LOGGER.error(f"Failed to load checkpoint {checkpoint_path}: {e}. Trying cpu.")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load student backbone state
    _load_state(module=student_backbone, state_dict=state_dict, key="student")

    # Load student DINO head state
    try:
        _load_state(
            module=student_dino_head, state_dict=state_dict, key="student_dino_head"
        )
    except ValueError:
        LOGGER.warning(
            "No student_dino_head state found in checkpoint, keeping initialized weights"
        )

    # Load student iBOT head state if it exists
    if student_ibot_head is not None:
        try:
            _load_state(
                module=student_ibot_head, state_dict=state_dict, key="student_ibot_head"
            )
        except ValueError:
            LOGGER.warning(
                "No student_ibot_head state found in checkpoint, keeping initialized weights"
            )

    # Load teacher backbone state if available, otherwise copy from student
    try:
        _load_state(module=teacher_backbone, state_dict=state_dict, key="teacher")
    except ValueError:
        LOGGER.info("No teacher state found in checkpoint, copying from student")
        load_weight(teacher_backbone, student_backbone.state_dict())

    # Load teacher DINO head state if available, otherwise copy from student
    try:
        _load_state(
            module=teacher_dino_head, state_dict=state_dict, key="teacher_dino_head"
        )
    except ValueError:
        LOGGER.info(
            "No teacher_dino_head state found in checkpoint, copying from student"
        )
        load_weight(teacher_dino_head, student_dino_head.state_dict())

    # Load teacher iBOT head state if it exists
    if teacher_ibot_head is not None:
        try:
            _load_state(
                module=teacher_ibot_head, state_dict=state_dict, key="teacher_ibot_head"
            )
        except ValueError:
            if student_ibot_head is not None:
                LOGGER.info(
                    "No teacher_ibot_head state found in checkpoint, copying from student"
                )
                load_weight(teacher_ibot_head, student_ibot_head.state_dict())
            else:
                LOGGER.warning(
                    "No teacher_ibot_head state found in checkpoint, keeping initialized weights"
                )

    # Load ibot_pos_embedding if present
    if ibot_pos_embedding is not None:
        try:
            _load_state(
                module=ibot_pos_embedding,
                state_dict=state_dict,
                key="ibot_pos_embedding",
            )
        except ValueError:
            LOGGER.warning(
                "ibot_pos_embedding not found in checkpoint, "
                "keeping initialized weights"
            )

    # Load dino_scale_embedding if present
    if dino_scale_embedding is not None:
        try:
            _load_state(
                module=dino_scale_embedding,
                state_dict=state_dict,
                key="dino_scale_embedding",
            )
        except ValueError:
            LOGGER.warning(
                "dino_scale_embedding not found in checkpoint, "
                "keeping initialized weights"
            )


def _load_state(module: nn.Module, state_dict: dict, key: str):
    """Load state for a specific module from checkpoint with robust prefix handling."""
    if key in state_dict:
        module_state_dict = state_dict[key]
    else:
        raise ValueError(f"Checkpoint file does not contain the {key}")

    # Use the robust load_weight function
    load_weight(module, module_state_dict)
    LOGGER.info(f"Loaded {key} state from checkpoint")

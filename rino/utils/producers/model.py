import torch
from torch import nn
from pathlib import Path

from models import JetTransformerEncoder, ParticleTransformer
from ..ckpt import get_checkpoints_path, process_placeholder
from ..logger import LOGGER
from ..params import count_parameters


def load_weight(model, state_dict):
    """Load state dict with automatic prefix handling."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    
    # Try removing common prefixes
    prefixes_to_try = ["module.", "_orig_mod.", "_orig_mod.module."]
    
    for prefix in prefixes_to_try:
        try:
            new_state_dict = {
                k.replace(prefix, ""): v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if new_state_dict:  # Only try if we actually found keys with this prefix
                model.load_state_dict(new_state_dict)
                return
        except RuntimeError:
            continue
    
    # If none worked, raise the original error
    model.load_state_dict(state_dict)


def get_models(
    part_dim: int,
    config: dict,
    device: str | None = None,
    mode: str = "training",
) -> tuple[
    JetTransformerEncoder | ParticleTransformer,
    JetTransformerEncoder | ParticleTransformer,
    None,
]:
    """
    Get student and teacher models for DINO.
    Returns a tuple of (student, teacher, part_batch_norm) models, both using the same architecture.
    """

    model_type = config.get("model_type", "jet_transformer")
    compile_model = config.get("compile_model", False)
    if "jet" in model_type.lower():
        LOGGER.info("Using JetTransformer model")
        # Create student model
        student = JetTransformerEncoder(
            part_dim=part_dim,
            batch_first=True,
            **config["model_params"],
        )
        if device:
            student = student.to(device)
        LOGGER.info(f"Student Model: {student}")

        # Create teacher model with same architecture
        teacher = JetTransformerEncoder(
            part_dim=part_dim,
            batch_first=True,
            **config["model_params"],
        )
        if device:
            teacher = teacher.to(device)
        LOGGER.info(f"Teacher Model: {teacher}")
    else:
        # Particle Transformer
        LOGGER.info(f"Using ParticleTransformer model (input_dim = {part_dim})")
        student = ParticleTransformer(
            input_dim=part_dim,
            **config["model_params"],
        )
        if device:
            student = student.to(device)
        LOGGER.info(f"Student Model: {student}")

        teacher = ParticleTransformer(
            input_dim=part_dim,
            **config["model_params"],
        )
        if device:
            teacher = teacher.to(device)
        LOGGER.info(f"Teacher Model: {teacher}")

    # particle normalization
    if config.get("pre_aug_norm", False):
        LOGGER.warning("Pre-augmentation normalization is deprecated.")
        part_batch_norm = None
    else:
        part_batch_norm = None

    # Handle finetune_embeddings BEFORE loading states
    finetune_embeddings = config.get("training", {}).get("finetune_embeddings", False)
    
    # Load state if specified
    if mode.startswith("train"):
        _load_model_states(
            mode=mode,
            student=student,
            teacher=teacher,
            config=config,
        )

    # If no state is loaded, teacher starts with same weights as student
    if not _should_load_checkpoint(config, mode):
        load_weight(teacher, student.state_dict())

    # Teacher's parameters should not require gradients
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Apply finetune_embeddings logic AFTER loading states and setting up teacher
    if finetune_embeddings:
        LOGGER.info("Finetuning mode: Freeze all parameters except embedding layers")
        
        # Freeze ALL student parameters first
        for param in student.parameters():
            param.requires_grad = False
        
        # Then unfreeze only embedding layers
        student.unfreeze_embedding_layers()

    if compile_model:
        try:
            LOGGER.info("Compiling student and teacher models")
            student_compiled = torch.compile(
                model=student,
                mode="default",
                fullgraph=False,  # True require that the entire function be capturable into a single graph
                dynamic=True,  # for variable input size
                backend="inductor",
            )
            teacher_compiled = torch.compile(
                model=teacher,
                mode="default",
                fullgraph=False,  # True require that the entire function be capturable into a single graph
                dynamic=True,  # for variable input size
                backend="inductor",
            )
            # will not compile batch norm
            # after both models are compiled, replace the original models
            student = student_compiled
            teacher = teacher_compiled
        except Exception as e:
            LOGGER.error(
                f"Failed to compile models: {e}; continuing without compilation"
            )
            
    # Log model parameters
    # Only count parameters for the student model since teacher is a copy with no gradients
    table, params = count_parameters(student)
    LOGGER.info(f"Model Parameters:\n{table}")
    LOGGER.info(f"Total Parameters: {params}")
    _, params_head = count_parameters(student.head)
    LOGGER.info(f"Total Head Parameters: {params_head}")
    LOGGER.info(f"Backbone Parameters: {params - params_head}")

    return student, teacher, part_batch_norm


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
    student: JetTransformerEncoder,
    teacher: JetTransformerEncoder,
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
        checkpoint_path = Path(process_placeholder(s=load_path, config=config, epoch_num=None))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at load_path: {checkpoint_path}")
        LOGGER.info(f"Loading checkpoint from load_path: {checkpoint_path}")
    elif load_epoch:
        # Use load_epoch
        checkpoint_path = get_checkpoints_path(config=config, epoch_num=load_epoch)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at load_epoch {load_epoch}: {checkpoint_path}")
        LOGGER.info(f"Loading checkpoint from load_epoch {load_epoch}: {checkpoint_path}")
    else:
        # No loading specified
        LOGGER.info("No load_path or load_epoch specified, models will use random initialization")
        return

    # Load the checkpoint
    try:
        state_dict = torch.load(checkpoint_path)
    except RuntimeError as e:
        LOGGER.error(f"Failed to load checkpoint {checkpoint_path}: {e}. Trying cpu.")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load student state
    _load_state(module=student, state_dict=state_dict, key="student")

    # Load teacher state if available, otherwise copy from student
    try:
        _load_state(module=teacher, state_dict=state_dict, key="teacher")
    except ValueError:
        LOGGER.info("No teacher state found in checkpoint, copying from student")
        load_weight(teacher, student.state_dict())


def _load_state(module: nn.Module, state_dict: dict, key: str):
    """Load state for a specific module from checkpoint with robust prefix handling."""
    if key in state_dict:
        module_state_dict = state_dict[key]
    else:
        raise ValueError(f"Checkpoint file does not contain the {key}")

    # Use the robust load_weight function
    load_weight(module, module_state_dict)
    LOGGER.info(f"Loaded {key} state from checkpoint")
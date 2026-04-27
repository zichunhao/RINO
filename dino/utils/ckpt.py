import re
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def process_placeholder(
    s: str, config: dict[str, Any], epoch_num: str | int | None
) -> str:
    """
    Replace placeholders in a string with values from the config.
    - JOBNAME: The name of the job, specified by "name" in the config.
    - EPOCHNUM: The epoch number, given by the epoch_num argument.
    - PROJECT_ROOT: The root directory of the project, which is ../dino.
    - LAST: Resolves to the last (largest epoch number) checkpoint in the
      directory. First looks for ``model_checkpoint_LAST.pt`` style names,
      finds all ``model_checkpoint_<N>.pt`` siblings, and picks the largest N.
    """
    job_name = config.get("name", "")
    if job_name is None:
        job_name = ""
    if epoch_num is not None:
        s = s.replace("EPOCHNUM", str(epoch_num))
    s = s.replace("JOBNAME", job_name)
    s = s.replace("PROJECT_ROOT", str(PROJECT_ROOT))

    # Resolve LAST: scan directory for the largest epoch number
    if "LAST" in s:
        s = _resolve_last(s)

    return s


def _resolve_last(path_str: str) -> str:
    """Replace ``LAST`` in *path_str* with the largest epoch number found.

    Strategy: treat ``LAST`` as a numeric wildcard, build a regex from the
    filename, and scan the parent directory for matches.  Falls back to
    ``best`` if the directory doesn't exist or contains no numbered
    checkpoints.
    """
    p = Path(path_str)
    parent = p.parent
    filename = p.name  # e.g. "model_checkpoint_LAST.pt"

    if not parent.is_dir():
        # Directory doesn't exist yet — fall back to "best"
        return path_str.replace("LAST", "best")

    # Build regex: escape the filename, then swap the literal "LAST" for (\d+)
    pattern = re.escape(filename).replace("LAST", r"(\d+)")
    regex = re.compile(pattern)

    best_epoch = None
    for f in parent.iterdir():
        m = regex.fullmatch(f.name)
        if m:
            epoch = int(m.group(1))
            if best_epoch is None or epoch > best_epoch:
                best_epoch = epoch

    if best_epoch is not None:
        return path_str.replace("LAST", str(best_epoch))

    # No numbered checkpoints found — fall back to "best"
    return path_str.replace("LAST", "best")


def get_checkpoints_path(config: dict[str, Any], epoch_num: str) -> Path:
    """
    Get the path to the checkpoints file based on the training config and epoch number.

    Args:
        config: The training config. The relevant config is config["training],
            specifically the "checkpoints_dir" and "checkpoints_filename" keys.
        epoch_num: The epoch number.

    Returns:
        The path to the checkpoints file.
    """
    job_name = config.get("name", "")
    if job_name is None:
        job_name = ""
    training_params = config["training"]
    checkpoints_filename = training_params["checkpoints_filename"]
    checkpoints_filename = process_placeholder(
        s=checkpoints_filename, config=config, epoch_num=epoch_num
    )

    checkpoints_dir = training_params["checkpoints_dir"]
    checkpoints_dir = process_placeholder(
        s=checkpoints_dir, config=config, epoch_num=epoch_num
    )

    return Path(checkpoints_dir) / checkpoints_filename


def find_latest_checkpoint_epoch(config: dict[str, Any]) -> int | None:
    """Find the latest checkpoint epoch number in the checkpoints directory.

    Returns the epoch number (int) or None if no checkpoints exist.
    """
    # Build the checkpoints directory path (use epoch 0 as dummy to resolve placeholders)
    ckpt_dir = get_checkpoints_path(config, 0).parent
    if not ckpt_dir.is_dir():
        return None

    # Extract the filename template and build a regex from it
    filename_template = config["training"]["checkpoints_filename"]
    # Replace JOBNAME placeholder so it doesn't interfere with the regex
    job_name = config.get("name", "") or ""
    filename_template = filename_template.replace("JOBNAME", job_name)
    # Escape everything except EPOCHNUM, then replace EPOCHNUM with a capture group
    pattern = re.escape(filename_template).replace("EPOCHNUM", r"(\d+)")
    regex = re.compile(pattern)

    latest_epoch = None
    for f in ckpt_dir.iterdir():
        m = regex.fullmatch(f.name)
        if m:
            epoch = int(m.group(1))
            if latest_epoch is None or epoch > latest_epoch:
                latest_epoch = epoch
    return latest_epoch


def save_checkpoint(
    checkpoint_dict: dict[str, any],
    config: dict[str, any],
    epoch_num: int,
) -> Path:
    """
    Save the checkpoint to the specified path.

    Args:
        checkpoint_dict: The checkpoint dictionary to save.
        config: The training config. The relevant config is config["training],
            specifically the "checkpoints_dir" and "checkpoints_filename" keys.
        epoch_num: The epoch number.

    Returns:
        The path to which the checkpoint was saved.
    """

    path = get_checkpoints_path(config=config, epoch_num=epoch_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict, path)
    return path


def get_state_dict(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def load_checkpoint(
    config: dict[str, Any],
    device: torch.device,
    epoch: int | str,
) -> dict[str, Any]:
    checkpoint_path = get_checkpoints_path(config, epoch)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def get_checkpoint_dict(
    student_backbone: nn.Module,
    student_dino_head: nn.Module,
    student_ibot_head: nn.Module | None,
    teacher_backbone: nn.Module,
    teacher_dino_head: nn.Module,
    teacher_ibot_head: nn.Module | None,
    ibot_pos_embedding: nn.Module | None,
    optimizer: optim.Optimizer,
    scheduler: _LRScheduler | ReduceLROnPlateau | None,
    wd_scheduler: Any | None,
    epoch: int,
    train_loss: dict[str, float],
    val_loss: dict[str, float],
    dino_scale_embedding: nn.Module | None = None,
    num_stale_epochs: int = 0,
) -> dict[str, Any]:
    checkpoint_dict = {
        "epoch": epoch,
        "student": get_state_dict(student_backbone),
        "student_dino_head": get_state_dict(student_dino_head),
        "student_ibot_head": (
            get_state_dict(student_ibot_head) if student_ibot_head is not None else None
        ),
        "teacher": get_state_dict(teacher_backbone),
        "teacher_dino_head": get_state_dict(teacher_dino_head),
        "teacher_ibot_head": (
            get_state_dict(teacher_ibot_head) if teacher_ibot_head is not None else None
        ),
        "ibot_pos_embedding": (
            get_state_dict(ibot_pos_embedding)
            if ibot_pos_embedding is not None
            else None
        ),
        "dino_scale_embedding": (
            get_state_dict(dino_scale_embedding)
            if dino_scale_embedding is not None
            else None
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

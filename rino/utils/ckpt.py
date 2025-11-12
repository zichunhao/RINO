from pathlib import Path
from typing import Any
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def process_placeholder(
    s: str, config: dict[str, Any], epoch_num: str | int | None
) -> str:
    """
    Replace placeholders in a string with values from the config.
    - JOBNAME: The name of the job, specified by "name" in the config.
    - EPOCHNUM: The epoch number, given by the epoch_num argument.
    - PROJECT_ROOT: The root directory of the project, which is ../dino.
    """
    job_name = config.get("name", "")
    if job_name is None:
        job_name = ""
    if epoch_num is not None:
        s = s.replace("EPOCHNUM", str(epoch_num))
    s = s.replace("JOBNAME", job_name)
    s = s.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    return s


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

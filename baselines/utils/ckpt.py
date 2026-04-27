import re
from pathlib import Path
from typing import Any
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
print(f"Project root: {PROJECT_ROOT}")


def process_placeholder(
    s: str, config: dict[str, Any], epoch_num: str | int | None
) -> str:
    """
    Replace placeholders in a string with values from the config.
    - JOBNAME: The name of the job, specified by "name" in the config.
    - EPOCHNUM: The epoch number, given by the epoch_num argument.
    - PROJECT_ROOT: The root directory of the project, which is ../dino.
    - LAST: Resolves to the last (largest epoch number) checkpoint in the
      directory. Falls back to "best" if no numbered checkpoints exist.
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

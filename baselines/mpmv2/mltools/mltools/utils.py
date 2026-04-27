"""General mix of utility functions mainly concerning dictionaries."""

import argparse
import json
import math
import operator
import re
from collections.abc import Generator, Iterable
from functools import reduce
from itertools import chain, islice
from pathlib import Path
from typing import Any

import numpy as np


def standard_job_array(
    job_name: str,
    work_dir: str,
    image_path: str,
    command: str,
    log_dir: str,
    opt_dict: dict,
    n_gpus: int = 0,
    n_cpus: int = 1,
    time_hrs: int = 1,
    mem_gb: int = 4,
    vram_per_gpu: int = 0,
    gpu_type: str = "",
    use_dashes: bool = True,
    extra_slurm: str = "",
    is_grid: bool = True,
) -> None:
    """Save a slurm submission file for running on the baobab cluster."""
    # First make sure that all the options are lists
    for key, vals in opt_dict.items():
        if not isinstance(vals, list):
            opt_dict[key] = [vals]

    # Calculate the total number of jobs to perform
    # If it is a grid search, then we need all combinations, which is the product
    if is_grid:
        n_jobs = np.prod([len(vals) for vals in opt_dict.values()])
    # Otherwise we just take the number of options in the first key
    # But we check that all the options are lists of the same length
    else:
        n_jobs = max(len(v) for v in opt_dict.values())
        for v in opt_dict.values():
            if len(v) == 1:
                continue  # No need to check single length
            if len(v) != n_jobs:
                raise ValueError("All options must have the same number of values")

    # Creating the slurm submision file
    with open(f"{job_name}.sh", "w", newline="\n", encoding="utf-8") as f:
        f.write("#!/bin/sh\n\n")
        f.write(f"#SBATCH --cpus-per-task={n_cpus}\n")
        f.write(f"#SBATCH --mem={mem_gb}GB\n")
        f.write(f"#SBATCH --time={time_hrs // 24}-{time_hrs % 24:02d}:00:00\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output={log_dir}/%A_%a.out\n")
        f.write(f"#SBATCH --chdir={work_dir}\n")
        if n_gpus:
            f.write("#SBATCH --partition=shared-gpu,private-dpnc-gpu\n")
            s = "#SBATCH --gres=gpu:"
            if gpu_type:
                s += f"{gpu_type}:"
            s += f"{n_gpus}"
            if vram_per_gpu:
                s += f",VramPerGpu:{vram_per_gpu}G"
            f.write(f"{s}\n")
        else:
            f.write("#SBATCH --partition=shared-cpu,private-dpnc-cpu\n")

        # Include the extra slurm here
        f.write(extra_slurm + "\n")

        # The job array setup using the number of jobs
        f.write(f"\n#SBATCH -a 0-{n_jobs - 1}\n\n")

        # Creating the bash lists of the job arguments
        pattern = r"[./\\+]"
        simple_keys = [re.sub(pattern, "", k) for k in opt_dict]

        for i, vals in enumerate(opt_dict.values()):
            if len(vals) == 1:  # No need for lists if one length
                continue
            f.write(f"{simple_keys[i]}=(")
            for v in vals:
                f.write(" " + str(v))
            f.write(" )\n")
        f.write("\n")

        # The command line arguments
        f.write('export XDG_RUNTIME_DIR=""\n')

        # Creating the base apptainer execution script
        f.write("srun apptainer exec --nv -B /srv,/home \\\n")
        f.write(f"   {image_path} \\\n")
        f.write(f"   {command} \\\n")

        # Now include the job array options using the bash lists
        run_tot = 1
        dashdash = "--" if use_dashes else ""
        for i, (opt, vals) in enumerate(opt_dict.items()):
            if len(vals) == 1:
                f.write(f"       {dashdash}{opt}={vals[0]} \\\n")
            elif is_grid:
                f.write(f"       {dashdash}{opt}=${{{simple_keys[i]}")
                f.write(f"[`expr ${{SLURM_ARRAY_TASK_ID}} / {run_tot} % {len(vals)}`]")
                f.write("} \\\n")
                run_tot *= len(vals)
            else:
                f.write(f"       {dashdash}{opt}=${{{simple_keys[i]}")
                f.write("[${SLURM_ARRAY_TASK_ID}]")
                f.write("} \\\n")

    print(f"--saved to {job_name}.sh")


def resursive_search(obj: dict, key: Any) -> Any | None:
    """Recursively search through a dictionary for a key and return the first."""
    if key in obj:
        return obj[key]
    for v in obj.values():
        if isinstance(v, dict):
            item = resursive_search(v, key)
            if item is not None:
                return item
    return None


def str2bool(mystring: str) -> bool:
    """Convert a string object into a boolean."""
    if isinstance(mystring, bool):
        return mystring
    if mystring.lower() in {"yes", "true", "t", "y", "1"}:
        return True
    if mystring.lower() in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def merge_dict(source: dict, update: dict) -> dict:
    """Merge two deep dictionaries recursively.

    Slow with deep dictionaries.

    Args
    ----
    source:
        The source dict, will be copied (not modified)
    update:
        Will be used to overwrite and append values to the source
    """
    # Make a copy of the source dictionary
    merged = source.copy()

    # Cycle through all of the keys in the update
    for key in update:
        # If the key not in the source then add move on
        if key not in merged:
            merged[key] = update[key]
            continue

        # Check type of variable
        dict_in_upt = isinstance(update[key], dict)
        dict_in_src = isinstance(source[key], dict)

        # If neither are a dict, then simply replace the leaf variable
        if not dict_in_upt and not dict_in_src:
            merged[key] = update[key]

        # If both are dicts, then implement recursion
        elif dict_in_upt and dict_in_src:
            merged[key] = merge_dict(source[key], update[key])

        # Otherwise one is a dict and the other is a leaf, so fail!
        else:
            raise ValueError(
                f"Trying to merge dicts but {key} is a leaf node in one not other"
            )

    return merged


def print_dict(dic: dict, indent: int = 1) -> None:
    """Recursively print a dictionary using json."""
    print(json.dumps(dic, indent=indent))


def get_from_dict(data_dict: dict, key_list: list, default=None) -> Any:
    """Return a value from a nested dictionary using list of keys."""
    try:
        return reduce(operator.getitem, key_list, data_dict)
    except KeyError:
        return default


def set_in_dict(data_dict: dict, key_list: list, value: Any):
    """Set a value in a nested dictionary using a list of keys."""
    get_from_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def flatlist(xs: any) -> list:
    """Return a flat list of any iterable or single element."""

    def flatten(xs) -> Generator:
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str | bytes)):
                yield from flatten(x)
            else:
                yield x

    return list(flatten([xs]))


def key_prefix(pref: str, dic: dict) -> dict:
    """Add a prefix to each key in a dictionary."""
    return {f"{pref}{key}": val for key, val in dic.items()}


def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Change the key used in a dictionary inplace only if it exists."""
    # If the original key is not present, nothing changes
    if old_key not in dic:
        return

    # Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    # Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]


def remove_keys_starting_with(dic: dict, match: str) -> dict:
    """Remove all keys from the dictionary if they start with.

    - Returns a copy of the dictionary
    """
    return {key: val for key, val in dic.items() if key[: len(match)] != match}


def insert_if_not_present(dictionary: dict, key: str, value: Any) -> None:
    """Add an entry to a dictionary if it isnt already present."""
    if key not in dictionary:
        dictionary[key] = value


def signed_angle_diff(angle1: float, angle2: float = 0) -> float:
    """Calculate diff between two angles reduced to the interval of [-pi, pi]."""
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi


def batched(iterable: Iterable, n: int) -> Generator:
    """Batch data into tuples of length n.

    The last batch may be shorter.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    assert n >= 1
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def distribute(sequence):
    """Enumerate the sequence evenly over the interval (0, 1).

    >>> list(distribute('abc'))
    [(0.25, 'a'), (0.5, 'b'), (0.75, 'c')]
    """
    m = len(sequence) + 1
    for i, x in enumerate(sequence, 1):
        yield i / m, x


def intersperse(*sequences):
    """Evenly intersperse the sequences.

    Based on https://stackoverflow.com/a/19293603/4518341

    >>> list(intersperse(range(10), 'abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
    >>> list(intersperse('XY', range(10), 'abc'))
    [0, 1, 'a', 2, 'X', 3, 4, 'b', 5, 6, 'Y', 7, 'c', 8, 9]
    >>> ''.join(intersperse('hlwl', 'eood', 'l r!'))
    'hello world!'
    """
    distributions = map(distribute, sequences)
    get0 = operator.itemgetter(0)
    for _, x in sorted(chain(*distributions), key=get0):
        yield x


def save_declaration(path: str = "done.txt") -> None:
    """Save a simple file which declares a job to be finished.

    This is typically called at the end of training for pipelines managers such as
    snakemake to be able to track the progress of the pipeline.
    """
    with open(Path(path).with_suffix(".txt"), "w") as f:
        # Add some text describing the job
        f.write("This file was created to declare the job as finisehd!\n")
        f.write("This is often called at the end of training for workflow managers\n")
        f.write("like snakemake to be able to track the progress of the pipeline.\n")

"""Classification fine-tuning for JetCLR.

Thin wrapper around dino/classification_train.py. All training logic lives there
because JetCLR and DINO share the same backbone (JetTransformerEncoder), the same
PARCEL data pipeline, and the same utils package.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Locate the PARCEL project root so we can import from `utils` and `dino`.
_script_dir = Path(__file__).resolve().parent
_baseline_dir = _script_dir.parent.parent.parent  # …/baselines/
_project_dir = _baseline_dir.parent  # PARCEL root
sys.path.insert(0, str(_project_dir))
sys.path.insert(0, str(_project_dir / "dino"))

from utils.logger import LOGGER, configure_logger
from classification_train import train  # reuse DINO training logic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification fine-tuning for JetCLR"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lf",
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file. If not specified, logs are written to stdout.",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Run index for parallel runs. When set, checkpoints/outputs are "
        "saved under a run-{N}/ subdirectory for mean/std aggregation.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Replicate DINO's run-index handling so indexed k8s Jobs can launch N
    # parallel pods that each write to their own run-{N}/ subdir.
    if args.run_index is not None:
        run_subdir = f"run-{args.run_index}"
        ckpt_dir = config["training"]["checkpoints_dir"]
        ckpt_path = Path(ckpt_dir)
        config["training"]["checkpoints_dir"] = str(
            ckpt_path.parent / run_subdir / ckpt_path.name
        )
        if "inference" in config and "output_dir" in config["inference"]:
            out_dir = config["inference"]["output_dir"]
            out_path = Path(out_dir)
            config["inference"]["output_dir"] = str(
                out_path.parent / run_subdir / out_path.name
            )

    configure_logger(
        logger=LOGGER,
        name="JetCLR Classification Finetuning",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    train(config=config, use_wandb=args.use_wandb)

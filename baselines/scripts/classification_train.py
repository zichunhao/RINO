"""Self-contained classification fine-tuning for baselines (JetCLR, etc.).

Uses its own model_factory.py for checkpoint loading (no DINO teacher/student).
Shares model classes and dataloaders from dino/ (generic infrastructure).

Usage:
    cd $PARCEL_ROOT
    python baselines/scripts/classification_train.py \
        -c baselines/configs/finetune/jetclr-modern-linear-unfreeze.yaml
"""

import sys
import argparse
import yaml
import importlib.util
from pathlib import Path

# Add dino/ to path for shared infrastructure (models, dataloaders, utils)
_project_root = Path(__file__).resolve().parent.parent.parent
_dino_dir = str(_project_root / "dino")
_baselines_scripts_dir = str(Path(__file__).resolve().parent)

# Insert dino/ FIRST so 'classification_train' resolves to dino's, not this file
sys.path.insert(0, _dino_dir)

# Load dino's classification_train by absolute path to avoid self-import
_dino_ct_path = _project_root / "dino" / "classification_train.py"
_spec = importlib.util.spec_from_file_location("dino_classification_train", _dino_ct_path)
_ct = importlib.util.module_from_spec(_spec)

# Before executing, add baselines/scripts/ so model_factory is importable
sys.path.insert(0, _baselines_scripts_dir)

_spec.loader.exec_module(_ct)

# Replace DINO's get_models_finetune with our baselines version
from model_factory import get_models_finetune as _baselines_get_models_finetune
_ct.get_models_finetune = _baselines_get_models_finetune

from utils.logger import LOGGER, configure_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baselines finetune classification")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-lv", "--log-level", type=str, default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("-lf", "--log-file", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--run-index", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Insert run subdirectory for parallel indexed jobs
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
        name="Baselines Classification Finetuning",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    _ct.train(config=config, use_wandb=args.use_wandb)

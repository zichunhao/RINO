"""Inference script for JetCLR+iBOT+KoLeo models.

Reuses the full inference/probe pipeline from dino_inference.py but loads a
single-model checkpoint (no teacher) via get_models_single.

Usage:
    python dino/jetclr_inference.py -c configs/jetclr/<config>.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml

from utils.device import get_available_device
from utils.producers import get_config
from utils.logger import LOGGER, configure_logger
from utils.ckpt import get_checkpoints_path
from models import AssembledModel
from models.jet_transformer_encoder import JetTransformerEncoder

# Reuse inference, save_batches, probe functions from dino_inference
from dino_inference import inference as _dino_inference, check_bf16_support

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
torch.set_float32_matmul_precision("high")


def load_model(
    config: dict, device: torch.device
) -> tuple[AssembledModel, bool]:
    """Load a JetCLR single-model checkpoint for inference."""
    LOGGER.debug("Infer part_dim from dataloader config")
    inference_dl = config["inference"]["dataloader"]
    first_split = next(iter(inference_dl.keys()))
    preprocessed = inference_dl[first_split].get("preprocessed", False)
    dataloader_config = get_config(
        config=config, mode="inference", preprocessed=preprocessed
    )

    particle_features = dataloader_config.outputs.sequence
    part_dim = len(particle_features)

    # Load backbone only — for representation-only inference + kNN/linear probe
    # the SSL projection head is not needed.
    LOGGER.info("Loading JetCLR backbone from pretrained checkpoint")
    backbone_params = config["models"]["backbone"].get("params", {})
    backbone = JetTransformerEncoder(part_dim=part_dim, **backbone_params).to(device)

    # Resolve checkpoint path using checkpoints_dir + checkpoints_filename from training config
    epoch_num = config["inference"].get("load_epoch", "best")
    ckpt_path = get_checkpoints_path(config=config, epoch_num=str(epoch_num))
    LOGGER.info(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "backbone" in state_dict:
        missing, unexpected = backbone.load_state_dict(state_dict["backbone"], strict=False)
        LOGGER.info(f"Backbone loaded (missing={missing}, unexpected={unexpected})")
    else:
        raise ValueError(f"No 'backbone' key in checkpoint. Keys: {list(state_dict.keys())}")

    model = AssembledModel(embedding=None, backbone=backbone, heads=None)
    has_new_head = False
    return model, has_new_head


# Monkey-patch the load_model in dino_inference so the shared inference()
# function uses our single-model loader.
import dino_inference as _di
_di.load_model = load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for JetCLR model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--include-head-output", action="store_true",
        help="Also include the head's output in the results.",
    )
    parser.add_argument(
        "--include-input", action="store_true",
        help="Include the original inputs in the output.",
    )
    parser.add_argument(
        "-lv", "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("-lf", "--log-file", type=str, default=None)
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--run-index", type=int, default=None,
        help="Run index for parallel runs.",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    if args.run_index is not None:
        run_subdir = f"run-{args.run_index}"
        if "training" in config and "checkpoints_dir" in config["training"]:
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
        name="JetCLR Inference",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    float32_matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(float32_matmul_precision)

    device = config.get("device", None)
    if device is None:
        device = get_available_device()
    else:
        device = torch.device(device)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            _base_output_dir = Path(
                config["inference"]["output_dir"]
                .replace("PROJECT_ROOT", str(PROJECT_ROOT))
                .replace("JOBNAME", config.get("name", ""))
                .replace("EPOCHNUM", "")
            )
            _wandb_id_file = _base_output_dir / "wandb_run_id.txt"

            wandb_id = None
            if _wandb_id_file.exists():
                wandb_id = _wandb_id_file.read_text().strip()
                LOGGER.info(f"Resuming W&B run {wandb_id}")

            wandb_run = wandb.init(
                project="JetCLR-inference",
                name=config.get("name", "jetclr"),
                config=config,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            _base_output_dir.mkdir(parents=True, exist_ok=True)
            _wandb_id_file.write_text(wandb_run.id)
            LOGGER.info(f"W&B run initialized (id={wandb_run.id})")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    _dino_inference(
        config,
        device,
        include_head_output=args.include_head_output,
        include_input=args.include_input,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

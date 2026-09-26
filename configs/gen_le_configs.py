#!/usr/bin/env python3
"""Generate label efficiency + final eval configs for all models × tasks × fractions.

Configs are written to configs/finetune/<subdir>/<name>.yaml (see TASK_SETTINGS
for the subdirectories), e.g. configs/finetune/finetune-jetnet-final/finetune-jn-rino.yaml.

Usage:
    # Generate TT 100% configs for all main models
    python configs/gen_le_configs.py --task tt --fractions 1.0

    # Generate JetNet LE for specific fractions
    python configs/gen_le_configs.py --task jn --fractions 0.001 0.01 0.1 --models rino sup

    # Generate TT LE for all fractions
    python configs/gen_le_configs.py --task tt --fractions 0.0001 0.001 0.01 0.1 0.2 0.5

    # Dry run
    python configs/gen_le_configs.py --task tt --fractions 1.0 --dry-run
"""

import argparse
import copy
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "configs" / "finetune"

# Polar-binned positional encoding of the RINO pretraining config
PBIN_PE = {
    "mode": "polar_binned",
    "input_indices": [-2, -1],
    "kwargs": {"n_r_bins": 100, "n_phi_bins": 100, "r_max": 5.6, "r_scale": 0.96},
}

# ============================================================
# Model definitions
# ============================================================
MODELS = {
    # RINO (tuned production model), pretrained with configs/dino/rino.yaml
    "rino": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/rino/checkpoints/model_checkpoint_best.pt",
        "freeze_backbone": 5,
        "pos_encoding_kwargs": PBIN_PE,
        "head_hidden_dims": [256, 128],
    },
    "mpmv1": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/mpmv1/mpmv1-rino/backbone.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
    "mpmv2": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/mpmv2/pretrain-rino-a100/backbone.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
    "jetclr-orig": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/jetclr/jetclr-rinomodel/backbone.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
    "omnijet": {
        "pooling": "last_token",
        "backbone_weight_path": "PROJECT_ROOT/experiments/omnijet_alpha/omnijet-rino/backbone.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
    # OmniJet-alpha with mean pooling, as in the main comparison table
    "omnijet-mean": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/omnijet_alpha/omnijet-rino/backbone.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
    # Supervised baseline: randomly initialized backbone trained end-to-end (no
    # frozen-backbone phase), CLS-token pooling and a 5-epoch LR warmup (see make_config)
    "sup": {
        "pooling": "cls_token",
        "backbone_weight_path": None,
        "freeze_backbone": False,
        "head_hidden_dims": [256, 128],
    },
    "jclr-3tier-recon": {
        "pooling": "mean",
        "backbone_weight_path": "PROJECT_ROOT/experiments/jetclr-3tier-recon/checkpoints/model_checkpoint_best.pt",
        "freeze_backbone": 5,
        "head_hidden_dims": [256, 128],
    },
}

# Fraction name mapping
FRAC_NAMES = {
    0.0001: "fp01", 0.001: "fp1", 0.01: "f01", 0.05: "f05",
    0.1: "f10", 0.2: "f20", 0.25: "f25", 0.5: "f50", 1.0: "full",
}

# Backbone base params (shared)
BACKBONE_BASE = {
    "d_model": 256, "nhead": 16, "num_layers": 8, "norm": "RMSNorm",
    "layer_scale_init": 0.01, "num_registers": 4, "jet_dim": 4,
    "mlp_ratio": 4, "apply_final_norm": True, "apply_embedding_norm": True,
}

# Task-specific settings
TASK_SETTINGS = {
    "jn": {
        "train_dl_config": "PROJECT_ROOT/configs/dataloaders/jetnet/kinematics.yaml",
        "train_patterns": ["gqt_train.h5"], "train_batch_size": 1024,
        "val_patterns": ["gqt_val.h5"],
        "bce_pos_weight": 2.0,
        "config_subdir_le": "finetune-label-eff",
        "config_subdir_full": "finetune-jetnet-final",
        "exp_subdir_le": "finetune-label-eff",
        "exp_subdir_full": "finetune-jetnet-final",
    },
    "tt": {
        "train_dl_config": "PROJECT_ROOT/configs/dataloaders/toptagging/kinematics.yaml",
        "train_patterns": ["train.h5"], "train_batch_size": 256,
        "val_patterns": ["validation.h5"],
        "bce_pos_weight": 1.0,
        "accumulation_steps": 4,
        "config_subdir_le": "finetune-toptagging-le",
        "config_subdir_full": "finetune-toptagging-final",
        "exp_subdir_le": "finetune-toptagging-le",
        "exp_subdir_full": "finetune-toptagging-final",
    },
}


def make_config(model_name: str, task_name: str, fraction: float) -> tuple[dict, str]:
    """Build the finetuning config for one model, task and label fraction.

    Returns the config dict and the config subdirectory it belongs to.
    """
    model = MODELS[model_name]
    task = TASK_SETTINGS[task_name]
    is_le = fraction < 1.0
    frac_name = FRAC_NAMES.get(fraction, f"f{int(fraction*100):02d}")

    config_subdir = task["config_subdir_le"] if is_le else task["config_subdir_full"]
    exp_subdir = task["exp_subdir_le"] if is_le else task["exp_subdir_full"]

    if is_le:
        if task_name == "jn":
            name = f"finetune-jn-le-{model_name}-{frac_name}"
        else:
            name = f"finetune-tt-le-{model_name}-{frac_name}"
    else:
        if task_name == "jn":
            name = f"finetune-jn-{model_name}"
        else:
            name = f"finetune-tt-{model_name}"

    # Backbone
    params = copy.deepcopy(BACKBONE_BASE)
    params["pooling"] = model["pooling"]
    if "pos_encoding_kwargs" in model:
        params["pos_encoding_kwargs"] = copy.deepcopy(model["pos_encoding_kwargs"])

    # Head
    dims = model["head_hidden_dims"]
    head = {
        "type": "MLPHead",
        "params": {
            "output_dim": 1, "hidden_dims": dims,
            "activations": ["ReLU"] * len(dims),
            "batch_norms": [False] * len(dims),
            "dropouts": [0.1] * len(dims),
            "l2_norm": False, "weight_norm": False,
        },
    }

    # Train dataloader
    train_kwargs = {
        "batch_size": task["train_batch_size"],
        "patterns": task["train_patterns"],
        "cache_size": 60, "size_multiplier": 2.5, "preload_workers": 10,
    }
    if is_le:
        train_kwargs["train_fraction"] = fraction

    # LR warmup epochs: 10 for pretrained backbones, 5 for the supervised baseline
    warmup = 5 if model_name == "sup" else 10

    cfg = {
        "name": name,
        "device": "cuda", "accelerate": True, "use_bf16": True,
        "float32_matmul_precision": "high",
        "models": {"backbone": {"type": "JetTransformerEncoder", "params": params}, "head": head},
        "shuffle": {"training": True, "inference": False},
        "training": {
            "dataloader": {
                "train": {
                    "preprocessed": True, "cached": True,
                    "config": task["train_dl_config"],
                    "num_workers": 6, "prefetch_factor": 8,
                    "kwargs": train_kwargs,
                },
                "val": {
                    "preprocessed": True, "cached": True,
                    "config": task["train_dl_config"],
                    "num_workers": 6, "prefetch_factor": 8,
                    "kwargs": {
                        "batch_size": 2048, "patterns": task["val_patterns"],
                        "cache_size": 20, "size_multiplier": 2.5, "preload_workers": 10,
                    },
                },
            },
            "checkpoints_filename": "model_checkpoint_EPOCHNUM.pt",
            "checkpoints_dir": f"PROJECT_ROOT/experiments/{exp_subdir}/JOBNAME/checkpoints",
            "load_epoch": None,
            "freeze_backbone": model["freeze_backbone"],
            "freeze_embedding": False,
            "num_epochs": 55, "patience": 5,
            "bce_pos_weight": task["bce_pos_weight"],
            "use_focal_loss": False, "label_smoothing": 0,
            "lwf_alpha": 0, "l2sp_alpha": 0, "grad_clip": 1.0,
            "optimizer": {
                "name": "AdamW",
                "params": {"lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-06, "weight_decay": 0.05},
                "backbone_lr_factor": 1.0,
            },
            "scheduler": {
                "name": "SequentialLR",
                "params": {
                    "milestones": [warmup],
                    "schedulers": [
                        {"name": "CosineScheduler", "params": {"base_value": 0.0001, "final_value": 1.0, "total_iters": warmup}},
                        {"name": "CosineScheduler", "params": {"base_value": 1.0, "final_value": 0.001, "total_iters": 55 - warmup}},
                    ],
                },
            },
        },
        "inference": {
            "load_epoch": "best",
            "output_filename": "output_SPLIT_EPOCHNUM.pt",
            "output_dir": f"PROJECT_ROOT/experiments/{exp_subdir}/JOBNAME/inference",
            "splits": [],
            "dataloader": {},
        },
    }

    if model["backbone_weight_path"]:
        cfg["training"]["backbone_weight_path"] = model["backbone_weight_path"]
    if "accumulation_steps" in task:
        cfg["training"]["accumulation_steps"] = task["accumulation_steps"]

    # Inference dataloaders
    inf = cfg["inference"]
    dl_config = task["train_dl_config"]
    if task_name == "jn":
        inf["splits"] = ["val_jetnet", "test_jetnet", "test_jetclass"]
        inf["dataloader"]["val_jetnet"] = {"preprocessed": True, "cached": True, "config": dl_config, "num_workers": 6, "prefetch_factor": 8, "kwargs": {"batch_size": 2048, "patterns": ["gqt_val.h5"], "cache_size": 20, "size_multiplier": 2.5, "preload_workers": 10}}
        inf["dataloader"]["test_jetnet"] = {"preprocessed": True, "cached": True, "config": dl_config, "num_workers": 6, "prefetch_factor": 8, "kwargs": {"batch_size": 2048, "patterns": ["gqt_test.h5"], "cache_size": 20, "size_multiplier": 2.5, "preload_workers": 10}}
    else:
        inf["splits"] = ["val_toptagging", "test_toptagging", "test_jetclass"]
        inf["dataloader"]["val_toptagging"] = {"preprocessed": True, "cached": True, "config": dl_config, "num_workers": 6, "prefetch_factor": 8, "kwargs": {"batch_size": 2048, "patterns": ["validation.h5"], "cache_size": 20, "size_multiplier": 2.5, "preload_workers": 10}}
        inf["dataloader"]["test_toptagging"] = {"preprocessed": True, "cached": True, "config": dl_config, "num_workers": 6, "prefetch_factor": 8, "kwargs": {"batch_size": 2048, "patterns": ["test.h5"], "cache_size": 20, "size_multiplier": 2.5, "preload_workers": 10}}

    inf["dataloader"]["test_jetclass"] = {
        "eval_subsets": {"Tbqq_vs_QCD": {"positive": [8], "negative": [0]}},
        "preprocessed": False, "cached": True,
        "config": "PROJECT_ROOT/configs/dataloaders/jetclass-raw/kinematics.yaml",
        "num_workers": 6, "prefetch_factor": 8,
        "kwargs": {"batch_size": 500, "batch_size_atomic": 500, "patterns": ["**/TTBar_*.root", "**/ZJetsToNuNu_*.root"], "cache_size": 40, "size_multiplier": 2.5, "preload_workers": 6},
    }

    return cfg, config_subdir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=["jn", "tt"],
                        help="jn: JetNet, tt: Top Quark Tagging")
    parser.add_argument("--fractions", nargs="+", type=float, required=True,
                        help="Training label fractions (1.0 = full training set)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Keys of MODELS to generate (default: all)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output root; configs go to <out-dir>/<subdir>/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = args.models or list(MODELS.keys())
    count = 0
    for model_name in models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        for frac in args.fractions:
            cfg, subdir = make_config(model_name, args.task, frac)
            out_dir = args.out_dir / subdir
            out_path = out_dir / f"{cfg['name']}.yaml"
            if args.dry_run:
                print(f"  {out_path.name}")
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                print(f"  {out_path.name}")
            count += 1

    print(f"\n{'Would create' if args.dry_run else 'Created'} {count} configs")


if __name__ == "__main__":
    main()

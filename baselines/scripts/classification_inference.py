"""Standalone classification inference for baselines (JetCLR, OmniJet, etc.).

Loads model via baselines model_factory (NOT dino's get_models_finetune).
Implements the inference loop directly — no monkey-patching of dino_inference.

Usage:
    cd $PARCEL_ROOT
    python baselines/scripts/classification_inference.py \
        -c baselines/configs/finetune/jetclr-rinomodel-mlp-vanilla.yaml \
        --run-index 0 --include-head-output
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix as sk_confusion_matrix,
)

# ── path setup ─────────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
_dino_dir = str(_project_root / "dino")
_baselines_scripts_dir = str(Path(__file__).resolve().parent)

sys.path.insert(0, _dino_dir)           # shared utils, models, dataloader
sys.path.insert(0, _baselines_scripts_dir)  # model_factory

# ── shared dino infrastructure ──────────────────────────────────────────────────
from utils.logger import LOGGER, configure_logger
from utils.device import get_available_device
from utils.producers import get_dataloader_and_config, get_config

# ── baselines model factory ─────────────────────────────────────────────────────
from model_factory import get_models_finetune

PROJECT_ROOT = _project_root


# ── metrics helpers ─────────────────────────────────────────────────────────────

def _compute_binary_metrics(logits_m: torch.Tensor, labels_m: torch.Tensor):
    """Compute binary classification metrics from logits (B,1) and integer labels (B,).

    Returns:
        metrics: dict  — acc, precision, recall, f1, auc (all floats)
        curves:  dict  — roc_fpr/tpr/thresholds (float32 tensors), roc_auc, confusion_matrix
    """
    probs    = torch.sigmoid(logits_m[:, 0]).numpy()
    preds    = (probs > 0.5).astype(int)
    labels_np = labels_m.numpy()

    acc       = float((preds == labels_np).mean())
    precision = float(precision_score(labels_np, preds, zero_division=0))
    recall    = float(recall_score(labels_np, preds, zero_division=0))
    f1        = float(f1_score(labels_np, preds, zero_division=0))
    auc       = float(roc_auc_score(labels_np, probs))
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    cm        = sk_confusion_matrix(labels_np, preds)

    metrics = {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc,
        "roc_fpr":          fpr.tolist(),
        "roc_tpr":          tpr.tolist(),
        "roc_thresholds":   thresholds.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


# ── model loading ───────────────────────────────────────────────────────────────

def load_model(config: dict, device: torch.device):
    """Build and load the finetuned baselines model from checkpoint."""
    inference_dl = config["inference"]["dataloader"]
    first_split = next(iter(inference_dl.keys()))
    preprocessed = inference_dl[first_split].get("preprocessed", False)
    dataloader_config = get_config(
        config=config, mode="inference", preprocessed=preprocessed
    )
    particle_features = dataloader_config.outputs.sequence
    part_dim = len(particle_features)
    LOGGER.info(f"part_dim={part_dim} (from dataloader config)")

    has_new_head = "head" in config.get("models", {})
    LOGGER.info(f"Loading baselines model (has_new_head={has_new_head})")

    model = get_models_finetune(
        part_dim=part_dim,
        config=config,
        mode="inference",
        device=device,
        train_head=False,
    )
    return model, has_new_head


# ── output path helpers (inline, no dependency on dino_inference globals) ───────

def _resolve_path_str(s: str, config: dict, epoch) -> str:
    job_name = config.get("name", "")
    s = s.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    s = s.replace("JOBNAME", job_name)
    s = s.replace("EPOCHNUM", str(epoch))
    return s


def _inference_output_path(config: dict, split: str, epoch, file_idx: int = 0) -> Path:
    output_dir = _resolve_path_str(config["inference"]["output_dir"], config, epoch)
    output_filename = config["inference"]["output_filename"]
    output_filename = output_filename.replace("SPLIT", split)
    output_filename = _resolve_path_str(output_filename, config, epoch)
    name, ext = os.path.splitext(output_filename)
    output_filename = f"{name}-{file_idx}{ext}"
    return Path(output_dir) / output_filename


def _save_results(
    results: dict,
    config: dict,
    split: str,
    file_idx: int,
    wandb_run=None,
):
    epoch = config["inference"]["load_epoch"]
    output_dir = Path(_resolve_path_str(config["inference"]["output_dir"], config, epoch))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate lists → tensors
    for key in list(results.keys()):
        tensors = results[key]
        if not tensors:
            continue
        try:
            results[key] = torch.cat(tensors, dim=0)
        except RuntimeError:
            # pad to max size on non-batch dims then cat
            max_sizes = list(tensors[0].shape)
            for t in tensors[1:]:
                for i in range(1, len(t.shape)):
                    max_sizes[i] = max(max_sizes[i], t.shape[i])
            padded = []
            for t in tensors:
                padding = []
                for i in range(len(t.shape) - 1, 0, -1):
                    padding.extend([0, max_sizes[i] - t.shape[i]])
                padded.append(
                    torch.nn.functional.pad(t, padding) if any(p > 0 for p in padding) else t
                )
            results[key] = torch.cat(padded, dim=0)

    # Compute metrics if labels + logits present
    split_metrics = None
    if "label" in results and "logits" in results:
        logits = results["logits"]
        labels = results["label"]
        if "jetclass" in split:
            mask = labels != 9          # remove leptonic top
            logits_m = logits[mask]
            labels_m = labels[mask]
            if logits_m.shape[-1] == 1:
                labels_m = (labels_m != 0).long()
        else:
            logits_m, labels_m = logits, labels

        if logits_m.shape[-1] == 1:
            split_metrics = _compute_binary_metrics(logits_m, labels_m)
        else:
            # Multi-class: only acc for now
            preds = logits_m.argmax(-1)
            split_metrics = {"acc": float((preds == labels_m).float().mean().item())}

        results["acc"] = split_metrics["acc"]
        LOGGER.info(
            f"[{split}] acc={split_metrics['acc']:.4f}"
            + (f"  precision={split_metrics['precision']:.4f}"
               f"  recall={split_metrics['recall']:.4f}"
               f"  f1={split_metrics['f1']:.4f}"
               f"  auc={split_metrics['auc']:.4f}" if "auc" in split_metrics else "")
        )
        if wandb_run is not None:
            step = epoch if isinstance(epoch, int) else None
            log_kw = {"step": step} if step is not None else {}
            wandb_run.log(
                {f"{split}/{k}": v for k, v in split_metrics.items() if isinstance(v, float)},
                **log_kw,
            )

    # Build output filename
    output_filename = config["inference"]["output_filename"]
    output_filename = output_filename.replace("SPLIT", split)
    output_filename = _resolve_path_str(output_filename, config, epoch)
    name, ext = os.path.splitext(output_filename)
    output_filename = f"{name}-{file_idx}{ext}"

    output_path = output_dir / output_filename
    torch.save(results, output_path)
    LOGGER.info(f"Saved {split} (file {file_idx}) → {output_path}")
    return split_metrics


# ── main inference loop ─────────────────────────────────────────────────────────

@torch.no_grad()
def inference(
    config: dict,
    device: torch.device,
    include_head_output: bool = False,
    include_input: bool = False,
    wandb_run=None,
):
    want_bf16 = config.get("use_bf16", False)
    use_bf16 = want_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    LOGGER.info(f"use_bf16={use_bf16} (requested={want_bf16})")

    float32_matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(float32_matmul_precision)

    splits = config["inference"].get("splits", ["val", "test"])
    load_epoch = config["inference"].get("load_epoch", "best")
    load_epochs = load_epoch if isinstance(load_epoch, list) else [load_epoch]
    auto_resume = config["inference"].get("auto_resume", True)
    batches_per_file = config["inference"].get("batches_per_file")

    # Accumulate {ep: {split: metrics}} across all splits; write incrementally
    all_metrics: dict = {}   # {ep: {split: metrics_dict}}

    for split in splits:
        LOGGER.info(f"── Split: {split} ──")
        dataloader, _ = get_dataloader_and_config(
            config=config, mode="inference", split=split
        )

        for ep in load_epochs:
            if auto_resume:
                out_dir = _inference_output_path(config, split, ep).parent
                metrics_path = out_dir / f"metrics_{ep}.json"
                pt_path = _inference_output_path(config, split, ep)
                if metrics_path.exists():
                    # Check this split is already in the metrics file
                    try:
                        with open(metrics_path) as f:
                            existing = json.load(f)
                        if split in existing:
                            LOGGER.info(f"Skipping {split} epoch={ep}: metrics exist at {metrics_path}")
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass
                if pt_path.exists():
                    LOGGER.info(f"Skipping {split} epoch={ep}: output exists at {pt_path}")
                    continue

            config["inference"]["load_epoch"] = ep
            LOGGER.info(f"Loading model epoch={ep}")
            model, has_new_head = load_model(config=config, device=device)
            model.eval()

            results: dict[str, list] = {"rep": []}
            if has_new_head:
                results["logits"] = []
            elif include_head_output:
                results["proj"] = []

            if include_input:
                results["jets"] = []
                results["particles"] = []
                results["mask"] = []

            current_batch = 0
            file_idx = 0

            with tqdm(dataloader, desc=f"Inference {split}", unit="batch") as pbar:
                for batch in pbar:
                    particles = torch.tensor(
                        batch["sequence"], dtype=torch.float32, device=device
                    )
                    jets = torch.tensor(
                        batch["class_"], dtype=torch.float32, device=device
                    )
                    mask = torch.tensor(batch["mask"], dtype=torch.bool, device=device)

                    if include_input:
                        results["jets"].append(jets.cpu())
                        results["particles"].append(particles.cpu())
                        results["mask"].append(mask.cpu())

                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        if has_new_head:
                            rep, logits = model(
                                particles=particles, jets=jets, mask=mask
                            )
                            results["rep"].append(rep.float().cpu())
                            results["logits"].append(logits.float().cpu())
                        elif include_head_output:
                            rep, proj = model(
                                particles=particles, jets=jets, mask=mask, rep_only=False
                            )
                            results["rep"].append(rep.float().cpu())
                            results["proj"].append(proj.float().cpu())
                        else:
                            rep = model(
                                particles=particles, jets=jets, mask=mask, rep_only=True
                            )
                            results["rep"].append(rep.float().cpu())

                    # Aux fields (label, etc.)
                    for aux_key, aux_val in batch.get("aux", {}).items():
                        if aux_key not in results:
                            results[aux_key] = []
                        results[aux_key].append(torch.tensor(aux_val).cpu())

                    current_batch += 1
                    if batches_per_file and current_batch >= batches_per_file:
                        _save_results(results, config, split, file_idx, wandb_run)
                        file_idx += 1
                        current_batch = 0
                        results = {k: [] for k in results}

            if any(len(v) > 0 for v in results.values()):
                split_metrics = _save_results(results, config, split, file_idx, wandb_run)
                # Accumulate and write incrementally after each split
                if split_metrics is not None:
                    all_metrics.setdefault(ep, {})[split] = split_metrics
                    ep_out_dir = Path(_resolve_path_str(
                        config["inference"]["output_dir"], config, ep
                    ))
                    ep_out_dir.mkdir(parents=True, exist_ok=True)
                    json_path = ep_out_dir / f"metrics_{ep}.json"
                    with open(json_path, "w") as f:
                        json.dump(all_metrics[ep], f, indent=2)
                    LOGGER.info(f"Metrics JSON → {json_path}")

        del dataloader
        torch.cuda.empty_cache()


# ── entrypoint ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baselines finetune inference (standalone)")
    parser.add_argument("-c", "--config", required=True, help="Path to finetune config YAML")
    parser.add_argument("--include-head-output", action="store_true")
    parser.add_argument("--include-input", action="store_true")
    parser.add_argument("--run-index", type=int, default=None,
                        help="Insert run-N subdir into checkpoint and output paths")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("-lv", "--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-lf", "--log-file", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Insert run-N subdir for indexed parallel jobs
    if args.run_index is not None:
        run_subdir = f"run-{args.run_index}"
        if "training" in config and "checkpoints_dir" in config["training"]:
            p = Path(config["training"]["checkpoints_dir"])
            config["training"]["checkpoints_dir"] = str(p.parent / run_subdir / p.name)
        if "inference" in config and "output_dir" in config["inference"]:
            p = Path(config["inference"]["output_dir"])
            config["inference"]["output_dir"] = str(p.parent / run_subdir / p.name)

    configure_logger(logger=LOGGER, name="Baselines Inference",
                     log_file=args.log_file, log_level=args.log_level)

    torch.set_float32_matmul_precision(config.get("float32_matmul_precision", "highest"))

    device = torch.device(config.get("device") or get_available_device())
    LOGGER.info(f"Device: {device}")

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.get("wandb_project", "RINO-inference"),
            name=config.get("name", "baselines-inference"),
            config=config,
            resume="allow",
        )

    inference(
        config=config,
        device=device,
        include_head_output=args.include_head_output,
        include_input=args.include_input,
        wandb_run=wandb_run,
    )

    if wandb_run:
        wandb_run.finish()

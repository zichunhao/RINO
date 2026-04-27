import argparse
import json
from pathlib import Path
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
import copy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    confusion_matrix as sk_confusion_matrix,
)

from utils.device import get_available_device
from utils.producers import (
    get_models,
    get_models_finetune,
    get_dataloader_and_config,
    get_config,
)
from utils.logger import LOGGER, configure_logger
from models import AssembledModel
from dino_train import check_bf16_support

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
torch.set_float32_matmul_precision("high")


def _compute_binary_metrics(logits_m: torch.Tensor, labels_m: torch.Tensor):
    """Compute binary classification metrics from logits (B,1) and integer labels (B,).

    Returns:
        metrics: dict  — acc, precision, recall, f1, auc (all floats)
        curves:  dict  — roc_fpr/tpr/thresholds (float32 tensors), roc_auc, confusion_matrix
    """
    probs     = torch.sigmoid(logits_m[:, 0]).numpy()
    preds     = (probs > 0.5).astype(int)
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


def load_model(
    config: dict, device: torch.device
) -> tuple[AssembledModel, torch.nn.Module | None, bool]:
    LOGGER.debug(f"Infer part_dim from dataloader config")
    # Determine if the first inference split uses preprocessed data
    inference_dl = config["inference"]["dataloader"]
    first_split = next(iter(inference_dl.keys()))
    preprocessed = inference_dl[first_split].get("preprocessed", False)
    dataloader_config = get_config(
        config=config, mode="inference", preprocessed=preprocessed
    )

    particle_features = dataloader_config.outputs.sequence
    part_dim = len(particle_features)

    has_new_head = "head" in config.get("models", {})
    if not has_new_head:
        # For DINO, we only need the teacher model for inference
        LOGGER.info("Loading DINO teacher model for inference")
        (
            (student_backbone, student_dino_head, student_ibot_head),
            (teacher_backbone, teacher_dino_head, teacher_ibot_head),
            ibot_pos_embedding,
            dino_scale_embedding,
        ) = get_models(
            part_dim=part_dim,
            config=config,
            mode="inference",
            device=device,
        )
        # Use teacher for inference
        model = AssembledModel(
            backbone=teacher_backbone,
            embedding=None,  # No separate embedding in DINO
            heads=teacher_dino_head,
        )
    else:
        # For classification heads, we load the finetuned model
        LOGGER.info("Loading finetuned DINO model with new head for inference")
        model = get_models_finetune(
            part_dim=part_dim,
            config=config,
            mode="inference",
            device=device,
            train_head=False,  # No new head for inference
        )
    return model, has_new_head


@torch.no_grad()
def inference(
    config: dict,
    device: torch.device,
    include_head_output: bool = False,
    include_input: bool = False,
    wandb_run=None,
):
    float32_matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(float32_matmul_precision)

    want_bf16 = config.get("use_bf16", False)
    if want_bf16:
        bf16_supported = check_bf16_support(device)
        if not bf16_supported:
            LOGGER.warning(
                f"use_bf16=True requested but bfloat16 is NOT supported on device "
                f"'{device}'. Falling back to float32."
            )
        use_bf16 = bf16_supported
    else:
        use_bf16 = False

    LOGGER.info(f"PyTorch version: {torch.__version__}")
    LOGGER.info(f"CUDA version: {torch.version.cuda}")
    LOGGER.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    LOGGER.info(f"float32_matmul_precision: {float32_matmul_precision}")
    LOGGER.info(f"use_bf16: {use_bf16} (requested: {want_bf16})")
    LOGGER.info(f"Inference with config: {config}")

    splits = config["inference"].get("splits", ["train", "val", "test"])
    LOGGER.info(f"Splits to process: {splits}")

    load_epoch = config["inference"].get("load_epoch", "best")
    LOGGER.info(f"Model load_epoch setting: {load_epoch}")
    if isinstance(load_epoch, list):
        load_epochs = load_epoch
    else:
        load_epochs = [load_epoch]

    # Accumulate {ep: {split: metrics}} across all splits; write incrementally
    all_metrics: dict = {}   # {ep: {split: metrics_dict}}

    for split in splits:
        LOGGER.info(f"Processing split: {split}")
        # Get dataloader
        dataloader, dataloader_config = get_dataloader_and_config(
            config=config,
            mode="inference",
            split=split,
        )

        # Determine cat_mean_pool: True if global flag or any acc_test task needs it
        probe_test_dict = config["inference"].get("acc_tests", {})
        rep_opts = config["inference"].get("rep", {})
        global_cat_mean = rep_opts.get("cat_mean_pool", False)
        any_task_needs_concat = any(
            isinstance(task_cfg, dict) and task_cfg.get("cat_mean_pool", False)
            for task_cfg in probe_test_dict.values()
        )
        cat_mean_pool = global_cat_mean or any_task_needs_concat
        if cat_mean_pool:
            LOGGER.info(
                "Concatenating mean-pooled particle features to the output representation"
            )

        auto_resume = config["inference"].get("auto_resume", True)

        for load_epoch in load_epochs:
            if auto_resume:
                # Primary check: per-split metrics JSON (includes probes if run)
                output_dir_check = Path(
                    _inference_output_path(config, split, load_epoch).parent
                )
                metrics_json_path = output_dir_check / f"metrics_{split}_{load_epoch}.json"
                pt_path = _inference_output_path(config, split, load_epoch)
                if metrics_json_path.exists():
                    LOGGER.info(
                        f"Skipping {split}/{load_epoch}: metrics JSON exists at "
                        f"{metrics_json_path} (auto_resume=True)"
                    )
                    continue
                if pt_path.exists():
                    LOGGER.info(
                        f"Skipping {split}/{load_epoch}: output .pt exists at "
                        f"{pt_path} (auto_resume=True)"
                    )
                    continue
            LOGGER.info(f"Loading model from epoch: {load_epoch}")
            config["inference"]["load_epoch"] = load_epoch
            # Load the teacher model for inference
            model, has_new_head = load_model(config=config, device=device)
            model.eval()

            results = {
                "rep": [],  # Store representations from teacher model
            }

            if has_new_head:
                if not include_head_output:
                    LOGGER.warning(
                        "--include-head-output is on for finetuned models with new head; logits will be saved."
                    )
                results["logits"] = []
            else:
                if include_head_output:
                    results["proj"] = []
                else:
                    LOGGER.info(
                        "Only saving representations from the teacher model; not the head's output"
                    )

            if include_input:
                LOGGER.info("Storing inputs (particles and jets) in results")
                results["jets"] = []  # Store jets
                results["particles"] = []  # Store particles
                results["mask"] = []  # Store mask for particles

            batches_per_file = config["inference"].get("batches_per_file")
            current_batch = 0
            file_idx = 0

            with tqdm(dataloader, desc=f"Inference {split}", unit="batch") as pbar:
                for batch in pbar:
                    particles = torch.tensor(batch["sequence"], dtype=torch.float32).to(
                        device
                    )
                    jets = torch.tensor(batch["class_"], dtype=torch.float32).to(device)
                    mask = torch.tensor(batch["mask"], dtype=torch.bool).to(device)

                    if include_input:
                        results["jets"].append(jets.cpu())
                        results["particles"].append(particles.cpu())
                        results["mask"].append(mask.cpu())

                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        if not has_new_head:
                            if include_head_output:
                                rep, proj = model(
                                    particles=particles,
                                    jets=jets,
                                    mask=mask,
                                    rep_only=False,
                                    cat_mean_pool=cat_mean_pool,
                                )
                                results["rep"].append(
                                    rep.float().cpu()
                                )  # cast back to fp32 before storing
                                results["proj"].append(proj.float().cpu())
                            else:
                                rep = model(
                                    particles=particles,
                                    jets=jets,
                                    mask=mask,
                                    rep_only=True,
                                    cat_mean_pool=cat_mean_pool,
                                )
                                results["rep"].append(rep.float().cpu())
                        else:
                            rep, logits = model(
                                particles=particles,
                                jets=jets,
                                mask=mask,
                                cat_mean_pool=cat_mean_pool,
                            )
                            results["rep"].append(rep.float().cpu())
                            results["logits"].append(logits.float().cpu())

                    # Store auxiliary information
                    aux = batch["aux"]
                    for aux_key, aux_val in aux.items():
                        if aux_key not in results:
                            results[aux_key] = []
                        results[aux_key].append(torch.tensor(aux_val).cpu())

                    if batches_per_file and current_batch >= batches_per_file:
                        save_batches(
                            results=results,
                            config=config,
                            split=split,
                            file_idx=file_idx,
                            cat_mean_pool=cat_mean_pool,
                            wandb_run=wandb_run,
                        )
                        current_batch = 0
                        file_idx += 1
                        results = {k: [] for k in results.keys()}

                    current_batch += 1

            # Save the remaining batches or save all
            if len(next(iter(results.values()))) != 0:
                perform_acc_test = "jetclass" in split
                split_metrics = save_batches(
                    results=results,
                    config=config,
                    split=split,
                    file_idx=file_idx,
                    perform_acc_test=perform_acc_test,
                    cat_mean_pool=cat_mean_pool,
                    wandb_run=wandb_run,
                )
                # Accumulate and write combined metrics file incrementally
                if split_metrics is not None:
                    all_metrics.setdefault(load_epoch, {})[split] = split_metrics
                    ep_output_dir = _inference_output_path(config, split, load_epoch).parent
                    ep_output_dir.mkdir(parents=True, exist_ok=True)
                    json_path = ep_output_dir / f"metrics_{load_epoch}.json"
                    with open(json_path, "w") as f:
                        json.dump(all_metrics[load_epoch], f, indent=2)
                    LOGGER.info(f"Combined metrics JSON → {json_path}")

        # free memory
        del dataloader
        torch.cuda.empty_cache()


def _summarize_knn(knn_metrics: dict) -> dict:
    """Summarize kNN probe results into JSON-friendly dict with mean/std."""
    summary = {}
    for k, metrics in knn_metrics.items():
        entry = {}
        for metric_name in ("acc", "f1", "precision", "recall"):
            vals = metrics.get(metric_name, [])
            if vals:
                entry[f"{metric_name}_mean"] = float(np.mean(vals))
                entry[f"{metric_name}_std"] = float(np.std(vals))
                entry[f"{metric_name}_values"] = [float(v) for v in vals]
        summary[str(k)] = entry
    return summary


def _summarize_linear_probe(lp_metrics: dict) -> dict:
    """Summarize linear probe results into JSON-friendly dict with mean/std."""
    summary = {}
    for metric_name in ("train", "test", "f1", "precision", "recall"):
        vals = lp_metrics.get(metric_name, [])
        if vals:
            summary[f"{metric_name}_mean"] = float(np.mean(vals))
            summary[f"{metric_name}_std"] = float(np.std(vals))
            summary[f"{metric_name}_values"] = [float(v) for v in vals]
    return summary


def _inference_output_path(
    config: dict, split: str, epoch: int | str, file_idx: int = 0
) -> Path:
    """Resolve the output file path for a given split, epoch, and file index."""
    job_name = config.get("name", "")
    output_dir = config["inference"]["output_dir"]
    output_dir = output_dir.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    output_dir = output_dir.replace("JOBNAME", job_name)
    output_dir = output_dir.replace("EPOCHNUM", str(epoch))
    output_filename = config["inference"]["output_filename"]
    output_filename = output_filename.replace("SPLIT", split)
    output_filename = output_filename.replace("JOBNAME", job_name)
    output_filename = output_filename.replace("EPOCHNUM", str(epoch))
    name, ext = os.path.splitext(output_filename)
    output_filename = f"{name}-{file_idx}{ext}"
    return Path(output_dir) / output_filename


def save_batches(
    results: dict[str, list[torch.Tensor]],
    config: dict,
    split: str,
    file_idx: int | None,
    perform_acc_test: bool = False,
    cat_mean_pool: bool = False,
    wandb_run=None,
):
    job_name = config.get("name", "")
    epoch = config["inference"]["load_epoch"]

    output_dir = config["inference"]["output_dir"]
    output_dir = output_dir.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    output_dir = output_dir.replace("JOBNAME", job_name)
    output_dir = output_dir.replace("EPOCHNUM", str(epoch))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine results
    for key in results:
        try:
            results[key] = torch.cat(results[key], dim=0)
        except RuntimeError as e:
            # Handle size mismatch by padding tensors
            tensor_list = results[key]

            if len(tensor_list) == 0:
                continue

            max_sizes = list(tensor_list[0].shape)
            for tensor in tensor_list[1:]:
                for i in range(1, len(tensor.shape)):  # Skip dim 0 (batch dimension)
                    max_sizes[i] = max(max_sizes[i], tensor.shape[i])

            # Pad tensors to match max sizes
            padded_tensors = []
            for tensor in tensor_list:
                # Calculate padding needed for each dimension
                padding = []
                for i in range(len(tensor.shape) - 1, 0, -1):  # Reverse order for F.pad
                    diff = max_sizes[i] - tensor.shape[i]
                    padding.extend([0, diff])  # [left_pad, right_pad] for each dim

                if any(p > 0 for p in padding):
                    padded_tensor = torch.nn.functional.pad(
                        tensor, padding, mode="constant", value=0
                    )
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)

            results[key] = torch.cat(padded_tensors, dim=0)

    split_metrics = None
    if "label" in results and "logits" in results:
        logits = results["logits"]
        labels = results["label"]

        # --- Per-subset evaluation ---
        # Config can define eval_subsets per split to compute metrics on
        # specific label groupings. Example:
        #   eval_subsets:
        #     Tbqq_vs_QCD: {positive: [8], negative: [0]}
        #     Tbl_vs_QCD:  {positive: [9], negative: [0]}
        #     top_vs_QCD:  {positive: [8, 9], negative: [0]}
        split_dl_cfg = config.get("inference", {}).get("dataloader", {}).get(split, {})
        eval_subsets = split_dl_cfg.get("eval_subsets", {})

        if eval_subsets:
            subset_metrics_all = {}
            for subset_name, subset_def in eval_subsets.items():
                pos_labels = subset_def["positive"]
                neg_labels = subset_def["negative"]
                pos_mask = torch.zeros(len(labels), dtype=torch.bool)
                neg_mask = torch.zeros(len(labels), dtype=torch.bool)
                for pl in pos_labels:
                    pos_mask |= labels == pl
                for nl in neg_labels:
                    neg_mask |= labels == nl
                mask = pos_mask | neg_mask
                if mask.sum() == 0:
                    continue
                sub_labels = pos_mask[mask].long()  # 1=positive, 0=negative

                if logits.shape[-1] == 1:
                    # Binary model: use logits directly
                    sub_logits = logits[mask]
                else:
                    # Multi-class model: convert to binary via
                    # P(signal) / [P(signal) + P(background)]
                    signal_cls = subset_def.get("signal_class")
                    bg_cls = subset_def.get("background_class", 0)
                    if signal_cls is None:
                        continue  # skip subsets without class mapping
                    probs_all = torch.softmax(logits[mask], dim=-1)
                    p_sig = probs_all[:, signal_cls]
                    p_bg = probs_all[:, bg_cls]
                    # Convert to logit scale for _compute_binary_metrics
                    binary_prob = p_sig / (p_sig + p_bg + 1e-8)
                    sub_logits = torch.log(binary_prob / (1 - binary_prob + 1e-8)).unsqueeze(-1)

                sub_metrics = _compute_binary_metrics(sub_logits, sub_labels)
                sub_metrics["n_positive"] = int(pos_mask.sum())
                sub_metrics["n_negative"] = int(neg_mask.sum())
                subset_metrics_all[subset_name] = sub_metrics
                LOGGER.info(
                    f"[{split}/{subset_name}] acc={sub_metrics['acc']:.4f}"
                    f"  recall={sub_metrics['recall']:.4f}"
                    f"  precision={sub_metrics['precision']:.4f}"
                    f"  auc={sub_metrics['auc']:.4f}"
                    f"  (n+={sub_metrics['n_positive']}, n-={sub_metrics['n_negative']})"
                )
            results["subset_metrics"] = subset_metrics_all

        # --- Overall metrics ---
        if "jetclass" in split and logits.shape[-1] == 1:
            # Binary: QCD (0) → 0, everything else → 1
            mask_logits = logits
            mask_labels = (labels != 0).long()
        else:
            mask_logits = logits
            mask_labels = labels

        if mask_logits.shape[-1] == 1:
            split_metrics = _compute_binary_metrics(mask_logits, mask_labels)
        else:
            preds = mask_logits.argmax(dim=-1)
            split_metrics = {"acc": float((preds == mask_labels).float().mean().item())}

        results["acc"] = split_metrics["acc"]
        LOGGER.info(
            f"[{split}] acc={split_metrics['acc']:.4f}"
            + (f"  precision={split_metrics['precision']:.4f}"
               f"  recall={split_metrics['recall']:.4f}"
               f"  f1={split_metrics['f1']:.4f}"
               f"  auc={split_metrics['auc']:.4f}" if "auc" in split_metrics else "")
        )
        if wandb_run is not None:
            epoch = config["inference"]["load_epoch"]
            step = epoch if isinstance(epoch, int) else None
            log_kwargs = {"step": step} if step is not None else {}
            wandb_run.log(
                {f"{split}/{k}": v for k, v in split_metrics.items() if isinstance(v, float)},
                **log_kwargs,
            )

    # Save results
    output_filename = config["inference"]["output_filename"]
    output_filename = output_filename.replace("SPLIT", split)
    output_filename = output_filename.replace("JOBNAME", job_name)
    output_filename = output_filename.replace("EPOCHNUM", str(epoch))

    if file_idx is not None:
        # Add file index before the file extension
        name, ext = os.path.splitext(output_filename)
        output_filename = f"{name}-{file_idx}{ext}"

    output_path = output_dir / output_filename
    torch.save(results, output_path)
    LOGGER.info(f"Results for {split} (batch {file_idx}) saved to {output_path}")

    if perform_acc_test:
        probe_test_dict = config["inference"].get("acc_tests", {})
        epoch = config["inference"]["load_epoch"]
        probe_step = epoch if isinstance(epoch, int) else None

        global_cat_mean = config["inference"].get("rep", {}).get("cat_mean_pool", False)

        def _rep_for_task(task_cfg: dict) -> dict:
            """Return results with rep sliced to cls-token half if task doesn't want concat."""
            task_cat_mean_pool = task_cfg.get("cat_mean_pool", global_cat_mean)
            if not task_cat_mean_pool and cat_mean_pool:
                # Inference ran with concat; this task only wants the cls-token (first half)
                sliced = dict(results)
                sliced["rep"] = results["rep"][:, : results["rep"].shape[1] // 2]
                return sliced
            return results

        def _strip_task_keys(task_cfg: dict) -> dict:
            """Remove keys that are not valid kwargs for probe functions."""
            return {k: v for k, v in task_cfg.items() if k != "cat_mean_pool"}

        metrics_json = {"split": split, "epoch": str(epoch)}
        if split_metrics is not None:
            metrics_json["classification"] = split_metrics
        if "subset_metrics" in results:
            metrics_json["subsets"] = {
                name: {k: v for k, v in m.items()
                       if k not in ("roc_fpr", "roc_tpr", "roc_thresholds", "confusion_matrix")}
                for name, m in results["subset_metrics"].items()
            }

        knn_dict = probe_test_dict.get("knn", {})
        if knn_dict:
            knn_results = train_knn(
                _rep_for_task(knn_dict),
                wandb_run=wandb_run,
                prefix=f"{split}/knn",
                step=probe_step,
                **_strip_task_keys(knn_dict),
            )
            metrics_json["knn"] = _summarize_knn(knn_results)

        linear_probe_dict = probe_test_dict.get("linear_probe", {})
        if linear_probe_dict:
            lp_results = train_linear_probe(
                _rep_for_task(linear_probe_dict),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                wandb_run=wandb_run,
                prefix=f"{split}/linear_probe",
                step=probe_step,
                **_strip_task_keys(linear_probe_dict),
            )
            metrics_json["linear_probe"] = _summarize_linear_probe(lp_results)

        # Save per-split metrics JSON (includes knn/linear_probe if present)
        json_path = output_dir / f"metrics_{split}_{epoch}.json"
        with open(json_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        LOGGER.info(f"Metrics JSON saved to {json_path}")

        if wandb_run is not None and probe_step is not None:
            wandb_run.log({}, step=probe_step, commit=True)

    return split_metrics


def train_knn(
    results: dict[str, torch.Tensor],
    k_list: list[int] = [1, 5, 10, 20, 100],
    labels_to_include: list[int] = [0, 8],
    sample_size: int = 200000,
    n_experiments: int = 10,
    wandb_run=None,
    prefix: str = "knn",
    step: int | None = None,
) -> dict[int, tuple[float, float]]:

    rep = results["rep"].cpu().detach()
    label = results["label"].cpu().detach()

    # random sample
    if sample_size > 0:
        sample_size = min(sample_size, rep.shape[0])
        idx = torch.randperm(rep.shape[0], device=rep.device)[:sample_size]
        rep = rep[idx]
        label = label[idx]

    rep = rep.numpy()
    label = label.numpy()

    mask = np.isin(label, labels_to_include)
    rep = rep[mask]
    label = label[mask]

    labels_to_include = [l for l in labels_to_include if l in label]
    labels_to_include = np.array(labels_to_include)
    label = np.searchsorted(labels_to_include, label)

    LOGGER.info(
        f"kNN probe with k values: {k_list}, sample size: {sample_size}, number of experiments: {n_experiments}, labels included: {labels_to_include.tolist()}"
    )

    sample_size = min(sample_size, rep.shape[0])
    idx = np.random.choice(rep.shape[0], sample_size, replace=False)
    rep = rep[idx]
    label = label[idx]

    num_classes = len(np.unique(label))
    cv = StratifiedKFold(n_splits=n_experiments, shuffle=True, random_state=42)

    all_metrics = {}
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)

        accs, f1s, precisions, recalls = [], [], [], []
        conf_matrices = []

        for train_idx, test_idx in cv.split(rep, label):
            knn.fit(rep[train_idx], label[train_idx])
            preds = knn.predict(rep[test_idx])
            y_true = label[test_idx]

            accs.append(accuracy_score(y_true, preds))
            f1s.append(f1_score(y_true, preds, average="macro", zero_division=0))
            precisions.append(
                precision_score(y_true, preds, average="macro", zero_division=0)
            )
            recalls.append(
                recall_score(y_true, preds, average="macro", zero_division=0)
            )
            conf_matrices.append(
                confusion_matrix(y_true, preds, labels=list(range(num_classes)))
            )

        # Average confusion matrix across folds within this run — same as linear probe
        mean_cm = np.mean(conf_matrices, axis=0)
        norm_cm = mean_cm / mean_cm.sum(axis=1, keepdims=True)

        # Return raw lists; caller aggregates across model init runs
        all_metrics[k] = {
            "acc": accs,  # list of n_experiments floats
            "f1": f1s,
            "precision": precisions,
            "recall": recalls,
            "confusion_matrix": mean_cm.tolist(),
            "confusion_matrix_normalized": norm_cm.tolist(),
        }
        LOGGER.info(
            f"kNN (k={k}) | acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
            f"f1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
            f"precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f} | "
            f"recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}"
        )
        if wandb_run is not None:
            log_kwargs = {"step": step} if step is not None else {}
            wandb_run.log(
                {
                    f"{prefix}/k{k}/acc": float(np.mean(accs)),
                    f"{prefix}/k{k}/f1": float(np.mean(f1s)),
                    f"{prefix}/k{k}/precision": float(np.mean(precisions)),
                    f"{prefix}/k{k}/recall": float(np.mean(recalls)),
                },
                **log_kwargs,
            )

    return all_metrics


@torch.enable_grad()
def train_linear_probe(
    results: dict[str, torch.Tensor],
    n_experiments: int = 10,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 2048,
    sample_size: int = -1,
    labels_to_include: list[int] = [0, 8],
    patience: int = 15,
    val_fraction: float = 0.1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    wandb_run=None,
    prefix: str = "linear_probe",
    step: int | None = None,
) -> dict[str, tuple[float, float]]:

    LOGGER.info(
        f"Linear probe | max_epochs={max_epochs}, lr={learning_rate}, "
        f"batch_size={batch_size}, sample_size={sample_size}, "
        f"n_experiments={n_experiments}, labels_to_include={labels_to_include}"
    )

    rep = results["rep"].detach()
    label = results["label"].detach()

    if sample_size > 0:
        sample_size = min(sample_size, rep.shape[0])
        idx = torch.randperm(rep.shape[0])[:sample_size]
        rep = rep[idx]
        label = label[idx]

    labels_to_include_t = torch.tensor(labels_to_include, dtype=label.dtype)
    mask = (label.unsqueeze(1) == labels_to_include_t).any(dim=1)
    rep = rep[mask]
    label = label[mask]

    labels_present = label.unique(sorted=True)
    label = torch.searchsorted(labels_present, label)

    num_classes = len(np.unique(label))
    LOGGER.info(f"Unique labels: {np.unique(label)}")
    input_dim = rep.shape[1]
    n = rep.shape[0]
    LOGGER.info(
        f"Linear probe: input_dim={input_dim}, num_classes={num_classes}, n={n}"
    )

    train_accs = []
    test_accs = []
    test_f1s = []
    test_precisions = []
    test_recalls = []
    conf_matrices = []

    fold_size = n // n_experiments
    torch.manual_seed(42)
    perm = torch.randperm(n)

    for fold in range(n_experiments):
        test_idx = perm[fold * fold_size : (fold + 1) * fold_size]
        train_val_idx = torch.cat(
            [perm[: fold * fold_size], perm[(fold + 1) * fold_size :]]
        )

        val_size = max(1, int(len(train_val_idx) * val_fraction))
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

        X_train, y_train = rep[train_idx], label[train_idx]
        X_val, y_val = rep[val_idx], label[val_idx]
        X_test, y_test = rep[test_idx], label[test_idx]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        linear_classifier = torch.nn.Linear(input_dim, num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=max(5, patience // 4),
            min_lr=learning_rate * 1e-4,
        )

        best_val_loss = float("inf")
        best_train_acc = 0.0
        best_state_dict = None
        patience_counter = 0
        effective_patience = max_epochs + 1 if patience < 0 else patience

        for epoch in tqdm(
            range(max_epochs), desc=f"Fold {fold+1}/{n_experiments}", unit="epoch"
        ):
            # --- train ---
            linear_classifier.train()
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = linear_classifier(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            train_acc = correct / total

            # --- val ---
            linear_classifier.eval()
            with torch.no_grad():
                val_outputs = linear_classifier(X_val.to(device))
                val_loss = criterion(val_outputs, y_val.to(device)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_state_dict = copy.deepcopy(linear_classifier.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= effective_patience:
                LOGGER.info(
                    f"Early stopping at epoch {epoch+1} | "
                    f"best_val_loss={best_val_loss:.4f}, best_train_acc={best_train_acc:.4f}"
                )
                break

        # --- test ---
        linear_classifier.load_state_dict(best_state_dict)
        linear_classifier.eval()
        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_test, y_test),
                batch_size=batch_size,
                shuffle=False,
            )
            all_predicted = []
            for batch_X, _ in test_loader:
                outputs = linear_classifier(batch_X.to(device))
                _, predicted = outputs.max(1)
                all_predicted.append(predicted.cpu())

        all_predicted = torch.cat(all_predicted).numpy()
        all_labels = y_test.numpy()

        test_acc = (all_predicted == all_labels).mean()
        test_f1 = f1_score(all_labels, all_predicted, average="macro", zero_division=0)
        test_prec = precision_score(
            all_labels, all_predicted, average="macro", zero_division=0
        )
        test_rec = recall_score(
            all_labels, all_predicted, average="macro", zero_division=0
        )
        cm = confusion_matrix(
            all_labels, all_predicted, labels=list(range(num_classes))
        )

        train_accs.append(best_train_acc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_precisions.append(test_prec)
        test_recalls.append(test_rec)
        conf_matrices.append(cm)

        LOGGER.info(
            f"Fold {fold+1} | acc={test_acc:.4f} | f1={test_f1:.4f} | "
            f"prec={test_prec:.4f} | rec={test_rec:.4f}"
        )

    mean_cm = np.mean(conf_matrices, axis=0)
    norm_cm = mean_cm / mean_cm.sum(axis=1, keepdims=True)

    train_mean, train_std = float(np.mean(train_accs)), float(np.std(train_accs))
    test_mean, test_std = float(np.mean(test_accs)), float(np.std(test_accs))
    f1_mean, f1_std = float(np.mean(test_f1s)), float(np.std(test_f1s))
    prec_mean, prec_std = float(np.mean(test_precisions)), float(
        np.std(test_precisions)
    )
    rec_mean, rec_std = float(np.mean(test_recalls)), float(np.std(test_recalls))

    LOGGER.info(f"Linear probe train accuracy:  {train_mean:.4f} ± {train_std:.4f}")
    LOGGER.info(f"Linear probe test  accuracy:  {test_mean:.4f}  ± {test_std:.4f}")
    LOGGER.info(f"Linear probe test  F1 (macro):{f1_mean:.4f}    ± {f1_std:.4f}")
    LOGGER.info(f"Linear probe test  precision: {prec_mean:.4f}  ± {prec_std:.4f}")
    LOGGER.info(f"Linear probe test  recall:    {rec_mean:.4f}   ± {rec_std:.4f}")
    LOGGER.info(f"Normalized confusion matrix:\n{np.round(norm_cm, 4)}")

    if wandb_run is not None:
        log_kwargs = {"step": step} if step is not None else {}
        wandb_run.log(
            {
                f"{prefix}/test_acc": test_mean,
                f"{prefix}/test_f1": f1_mean,
                f"{prefix}/test_precision": prec_mean,
                f"{prefix}/test_recall": rec_mean,
                f"{prefix}/train_acc": train_mean,
            },
            **log_kwargs,
        )

    return {
        "train": train_accs,
        "test": test_accs,
        "f1": test_f1s,
        "precision": test_precisions,
        "recall": test_recalls,
        "confusion_matrix": mean_cm.tolist(),
        "confusion_matrix_normalized": norm_cm.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for DINO model")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--include-head-output",
        action="store_true",
        help="Also include the head's output in the results. Always true for finetuned models.",
    )
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Include the original inputs (particles and jets) in the output.",
    )
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "-lf",
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file. If not specified, logs will be written to stdout.",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Run index for parallel runs. When set, outputs and checkpoints "
        "are read/written under a run-{N}/ subdirectory.",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Insert run subdirectory into paths for parallel indexed jobs
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
        name="DINO Inference",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    # ------------------------------------------------------------------ #
    # float32 matmul precision — apply globally before any tensor ops     #
    # ------------------------------------------------------------------ #
    float32_matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(float32_matmul_precision)

    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    device = config.get("device", None)
    if device is None:
        device = get_available_device()
    else:
        device = torch.device(device)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb  # noqa: PLC0415

            # Resolve base output dir (no epoch substitution) for run ID file
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
                project="RINO-inference",
                name=config.get("name", "rino"),
                config=config,
                id=wandb_id,
                resume="allow" if wandb_id is not None else None,
            )
            _base_output_dir.mkdir(parents=True, exist_ok=True)
            _wandb_id_file.write_text(wandb_run.id)
            LOGGER.info(f"W&B run initialized (id={wandb_run.id})")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    inference(
        config,
        device,
        include_head_output=args.include_head_output,
        include_input=args.include_input,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb_run.finish()

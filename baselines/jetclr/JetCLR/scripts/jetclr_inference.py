import sys
import copy
import json
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from modules.jet_augs import rescale_pts, translate_jets, rino_post_normalize

baseline_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
project_dir = baseline_dir.parent
sys.path.append(str(baseline_dir))

from utils.logger import LOGGER, configure_logger
from utils.producers import get_dataloader_and_config, get_model
from utils.producers.model import load_weight
from utils.device import get_available_device


def load_model(config: dict, device: torch.device):
    """Build model architecture and load trained checkpoint."""
    config_no_load = copy.deepcopy(config)
    config_no_load.get("inference", {}).pop("load_epoch", None)
    config_no_load.get("inference", {}).pop("load_path", None)

    _aug_cfg = config_no_load.get("augmentation", {})
    _part_dim = 7 if _aug_cfg.get("post_normalize") == "rino" else 3
    model = get_model(
        part_dim=_part_dim,
        config=config_no_load,
        device=device,
        mode="inference",
        assemble=True,
    )

    # Resolve experiment_dir placeholders
    expt_dir = config["experiment_dir"]
    expt_dir = expt_dir.replace("PROJECT_ROOT", str(project_dir))
    expt_dir = expt_dir.replace("JOBNAME", config.get("name", ""))
    expt_dir = Path(expt_dir)

    load_epoch = config["inference"]["load_epoch"]
    if load_epoch == "best":
        checkpoint_path = expt_dir / "model_best.pt"
    elif load_epoch == "final":
        checkpoint_path = expt_dir / "model_final.pt"
    else:
        checkpoint_path = expt_dir / f"model_ep{load_epoch}.pt"

    LOGGER.debug(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    LOGGER.debug("Checkpoint loaded")

    LOGGER.debug("Loading model backbone weights")
    load_weight(model.backbone, checkpoint["backbone"])
    try:
        LOGGER.debug("Loading model head weights")
        load_weight(model.heads, checkpoint["head"])
    except (ValueError, KeyError):
        LOGGER.warning("No head state found in checkpoint; keeping initialized weights")

    LOGGER.debug("Model weights loaded")
    model = model.to(device)
    model.eval()
    LOGGER.debug(f"Model loaded and moved to {device}")
    return model


@torch.no_grad()
def inference(config: dict, device: torch.device, wandb_run=None) -> None:
    """Run inference on specified splits and save representations."""
    LOGGER.info(f"PyTorch version: {torch.__version__}")
    LOGGER.info(f"Inference with config: {config}")

    apply_translation = config["inference"].get("apply_translation", False)
    translation_width = config["inference"].get("translation_width", 1.0)
    splits = config["inference"].get("splits", ["train", "val", "test"])
    acc_test_splits = config["inference"].get("acc_test_splits", ["val", "test"])

    aug_config = config.get("augmentation", {})
    use_rino = aug_config.get("post_normalize") == "rino"
    feature_map = aug_config.get("feature_map", None)
    if feature_map is not None:
        pt_idx = feature_map["pt"]
        eta_idx = feature_map["eta"]
        phi_idx = feature_map["phi"]
        energy_idx = feature_map.get("energy")
    else:
        pt_idx, eta_idx, phi_idx = 0, 1, 2
        energy_idx = None

    load_epoch_cfg = config["inference"].get("load_epoch", "best")
    load_epochs = (
        load_epoch_cfg if isinstance(load_epoch_cfg, list) else [load_epoch_cfg]
    )
    LOGGER.info(f"Processing splits: {splits}, load_epochs: {load_epochs}")

    for split in splits:
        LOGGER.info(f"Processing split: {split}")

        LOGGER.debug(f"Initializing dataloader for split: {split}")
        dataloader, _ = get_dataloader_and_config(
            config=config,
            split=split,
            mode="inference",
        )
        LOGGER.debug(f"Dataloader initialized for split: {split}")

        auto_resume = config["inference"].get("auto_resume", True)

        for load_epoch in load_epochs:
            if auto_resume:
                output_dir = Path(config["inference"]["output_dir"]) / config["name"]
                output_path = output_dir / f"{split}_epoch{load_epoch}_reps.pt"
                if output_path.exists():
                    LOGGER.info(
                        f"Skipping epoch {load_epoch} for split {split}: "
                        f"output already exists at {output_path} (auto_resume=True)"
                    )
                    continue

            LOGGER.info(f"Loading model from epoch: {load_epoch}")
            config["inference"]["load_epoch"] = load_epoch
            model = load_model(config=config, device=device)

            results = {"representations": [], "labels": []}

            LOGGER.debug("Entering batch loop — waiting for first batch")
            with tqdm(
                dataloader, desc=f"Inference {split} (epoch={load_epoch})", unit="batch"
            ) as pbar:
                for i, batch in enumerate(pbar):
                    if i == 0:
                        LOGGER.debug("First batch received")
                    x = torch.tensor(batch["sequence"]).to(
                        device, dtype=torch.float32, non_blocking=True
                    )
                    mask = torch.tensor(batch["mask"]).to(
                        device, dtype=torch.bool, non_blocking=True
                    )

                    label = batch.get("label")
                    if label is None:
                        label = batch["aux"]["label"]
                    label = torch.tensor(label, dtype=torch.long)

                    if apply_translation:
                        x = translate_jets(
                            x, width=translation_width,
                            pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx,
                        )

                    if use_rino:
                        cls = batch["class_"]
                        if not isinstance(cls, torch.Tensor):
                            cls = torch.tensor(cls)
                        cls = cls.to(device, dtype=torch.float32, non_blocking=True)
                        x = rino_post_normalize(
                            x, cls[:, 0], cls[:, 1], mask,
                            pt_idx=pt_idx, energy_idx=energy_idx,
                            eta_idx=eta_idx, phi_idx=phi_idx,
                        )
                    else:
                        x = rescale_pts(x, pt_idx=pt_idx)

                    rep, _ = model(x, mask=mask)

                    results["representations"].append(rep.cpu())
                    results["labels"].append(
                        label
                        if isinstance(label, torch.Tensor)
                        else torch.tensor(label)
                    )

                    if "aux" in batch:
                        for key, value in batch["aux"].items():
                            if key == "label":
                                continue
                            if key not in results:
                                results[key] = []
                            results[key].append(
                                value
                                if isinstance(value, torch.Tensor)
                                else torch.tensor(value)
                            )

            for key in results:
                results[key] = torch.cat(results[key], dim=0)

            output_dir = Path(config["inference"]["output_dir"])
            output_dir = output_dir / config["name"]
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{split}_epoch{load_epoch}_reps.pt"
            torch.save(results, output_path)
            LOGGER.info(f"Results saved to {output_path}")

            LOGGER.info(f"Unique labels: {torch.unique(results['labels']).tolist()}")
            LOGGER.info(f"Number of samples: {len(results['labels'])}")
            LOGGER.info(f"Representation shape: {results['representations'].shape}")

            if split in acc_test_splits:
                probe_results = dict(results)
                probe_results["rep"] = probe_results.pop("representations")
                probe_results["label"] = probe_results.pop("labels")

                probe_test_dict = config["inference"].get("acc_tests", {})

                step = load_epoch if isinstance(load_epoch, int) else None

                metrics_json = {"split": split, "epoch": str(load_epoch)}

                knn_dict = probe_test_dict.get("knn", {})
                if knn_dict:
                    LOGGER.info(
                        f"Running kNN probe on split '{split}' (epoch={load_epoch})"
                    )
                    knn_results = train_knn(
                        probe_results,
                        wandb_run=wandb_run,
                        prefix=f"{split}/knn",
                        step=step,
                        **knn_dict,
                    )
                    metrics_json["knn"] = _summarize_knn(knn_results)

                linear_probe_dict = probe_test_dict.get("linear_probe", {})
                if linear_probe_dict:
                    LOGGER.info(
                        f"Running linear probe on split '{split}' (epoch={load_epoch})"
                    )
                    lp_results = train_linear_probe(
                        probe_results,
                        device=device,
                        wandb_run=wandb_run,
                        prefix=f"{split}/linear_probe",
                        step=step,
                        **linear_probe_dict,
                    )
                    metrics_json["linear_probe"] = _summarize_linear_probe(lp_results)

                # Save metrics JSON
                json_path = output_dir / f"metrics_{split}_{load_epoch}.json"
                with open(json_path, "w") as f:
                    json.dump(metrics_json, f, indent=2)
                LOGGER.info(f"Metrics JSON saved to {json_path}")


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


def train_knn(
    results: dict[str, torch.Tensor],
    k_list: list[int] = [1, 5, 10, 20, 100],
    labels_to_include: list[int] = [0, 8],
    sample_size: int = 200000,
    n_experiments: int = 10,
    wandb_run=None,
    prefix: str = "knn",
    step: int | None = None,
) -> dict[int, dict]:

    rep = results["rep"].cpu().detach()
    label = results["label"].cpu().detach()

    if sample_size > 0:
        sample_size = min(sample_size, rep.shape[0])
        idx = torch.randperm(rep.shape[0])[:sample_size]
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
        f"kNN probe | k={k_list}, sample_size={sample_size}, "
        f"n_experiments={n_experiments}, labels={labels_to_include.tolist()}"
    )

    num_classes = len(np.unique(label))
    cv = StratifiedKFold(n_splits=n_experiments, shuffle=True, random_state=42)

    all_metrics = {}
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        accs, f1s, precisions, recalls, conf_matrices = [], [], [], [], []

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

        mean_cm = np.mean(conf_matrices, axis=0)
        norm_cm = mean_cm / mean_cm.sum(axis=1, keepdims=True)

        all_metrics[k] = {
            "acc": accs,
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
) -> dict[str, list]:

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

    num_classes = len(np.unique(label.numpy()))
    input_dim = rep.shape[1]
    n = rep.shape[0]
    LOGGER.info(
        f"Linear probe: input_dim={input_dim}, num_classes={num_classes}, n={n}"
    )

    train_accs, test_accs, test_f1s, test_precisions, test_recalls, conf_matrices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

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
            linear_classifier.train()
            correct = total = 0
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

            linear_classifier.eval()
            with torch.no_grad():
                val_loss = criterion(
                    linear_classifier(X_val.to(device)), y_val.to(device)
                ).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_state_dict = copy.deepcopy(linear_classifier.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= effective_patience:
                LOGGER.info(f"Early stopping at epoch {epoch+1}")
                break

        linear_classifier.load_state_dict(best_state_dict)
        linear_classifier.eval()
        with torch.no_grad():
            all_predicted = torch.cat(
                [
                    linear_classifier(bx.to(device)).max(1)[1].cpu()
                    for bx, _ in torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(X_test, y_test),
                        batch_size=batch_size,
                        shuffle=False,
                    )
                ]
            ).numpy()
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

    LOGGER.info(
        f"Linear probe train accuracy:  {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}"
    )
    LOGGER.info(
        f"Linear probe test  accuracy:  {np.mean(test_accs):.4f}  ± {np.std(test_accs):.4f}"
    )
    LOGGER.info(
        f"Linear probe test  F1 (macro):{np.mean(test_f1s):.4f}   ± {np.std(test_f1s):.4f}"
    )
    LOGGER.info(
        f"Linear probe test  precision: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}"
    )
    LOGGER.info(
        f"Linear probe test  recall:    {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}"
    )
    LOGGER.info(f"Normalised confusion matrix:\n{np.round(norm_cm, 4)}")

    if wandb_run is not None:
        log_kwargs = {"step": step} if step is not None else {}
        wandb_run.log(
            {
                f"{prefix}/test_acc": float(np.mean(test_accs)),
                f"{prefix}/test_f1": float(np.mean(test_f1s)),
                f"{prefix}/test_precision": float(np.mean(test_precisions)),
                f"{prefix}/test_recall": float(np.mean(test_recalls)),
                f"{prefix}/train_acc": float(np.mean(train_accs)),
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
    parser = argparse.ArgumentParser(description="jetCLR inference")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("-lf", "--log-file", type=str, default=None)
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    configure_logger(
        logger=LOGGER,
        name="jetCLR Inference",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    device = config.get("device", None)
    if device is None:
        device = get_available_device()
    else:
        device = torch.device(device)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project="RINO-JetCLR-inference",
                name=config.get("name", "jetclr"),
                config=config,
            )
            LOGGER.info("W&B run initialized")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    inference(config=config, device=device, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

import argparse
import itertools
from datetime import timedelta
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
import copy

from utils.device import get_available_device
from utils.producers import get_dataloader_and_config
from utils.logger import LOGGER, configure_logger
from models import JetTransformerEncoder
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DataLoaderConfiguration
from torch import nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

module_dir: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = module_dir.parent
# torch.set_float32_matmul_precision("high")


class JetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_particles: int,
        hidden_dim: int = 0,
        n_layers: int = 3,
        batch_norm: bool = False,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.n_particles = n_particles

        if n_layers < 1:
            self.net = nn.Linear(input_dim * n_particles, output_dim)
            self.head = None
        else:
            input_layer = nn.Linear(input_dim * n_particles, hidden_dim)
            hidden_layers = []
            for _ in range(n_layers - 1):
                hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if batch_norm:
                    hidden_layers.append(nn.BatchNorm1d(hidden_dim))
                hidden_layers.append(self.get_activation(activation))
            # Split backbone (everything up to the last linear) from the head
            self.net = nn.Sequential(
                input_layer,
                self.get_activation(activation),
                *hidden_layers,
                nn.Linear(hidden_dim, output_dim),
            )

        LOGGER.info(
            f"Initialized JetMLP | n_particles: {self.n_particles} | "
            f"self.net: {self.net}"
        )

    def forward(self, x, *args, **kwargs):
        """Forward pass.

        Args:
            x: particle tensor of shape (B, Np, Dp).
            return_rep: if True, return the final hidden-layer activations
                (pre-logits) instead of the class logits.  When ``n_layers < 1``
                there is no hidden layer, so the flattened input is returned.
        """
        B, Np, Dp = x.shape

        if Np > self.n_particles:
            x = x[:, : self.n_particles, :]
        elif Np < self.n_particles:
            padding = torch.zeros(B, self.n_particles - Np, Dp, device=x.device)
            x = torch.cat([x, padding], dim=1)

        x = x.reshape(B, -1)  # (B, n_particles * Dp)

        return self.net(x)

    def get_activation(self, name: str):
        activations = {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())


# ---------------------------------------------------------------------------
# Scan config helpers
# ---------------------------------------------------------------------------


def get_scan_combinations(config: dict) -> list[dict]:
    """Expand list-valued backbone params into all combinations."""
    params = config["models"]["backbone"].get("params", {})
    keys = list(params.keys())
    value_lists = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def get_output_dir(config: dict) -> Path:
    job_name = config.get("name", "scan")
    output_dir = config["inference"]["output_dir"]
    output_dir = output_dir.replace("PROJECT_ROOT", str(PROJECT_ROOT))
    output_dir = output_dir.replace("JOBNAME", job_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def train_knn(
    results: dict[str, torch.Tensor],
    k_list: list[int] = [1, 5, 10, 20, 100],
    labels_to_include: list[int] = [0, 8],
    sample_size: int = 200000,
    n_experiments: int = 10,
) -> dict[int, dict]:

    LOGGER.info(
        f"kNN probe with k values: {k_list}, sample size: {sample_size}, "
        f"number of experiments: {n_experiments}, labels_to_include: {labels_to_include}"
    )

    rep = results["rep"].cpu().detach()
    label = results["label"].cpu().detach()

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
) -> dict[str, tuple[float, float]]:

    lp_accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            dispatch_batches=False,
            split_batches=False,
        )
    )
    device = lp_accelerator.device

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

        X_train = torch.tensor(rep[train_idx], dtype=torch.float32)
        y_train = torch.tensor(label[train_idx], dtype=torch.long)
        X_val = torch.tensor(rep[val_idx], dtype=torch.float32)
        y_val = torch.tensor(label[val_idx], dtype=torch.long)
        X_test = torch.tensor(rep[test_idx], dtype=torch.float32)
        y_test = torch.tensor(label[test_idx], dtype=torch.long)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        linear_classifier = torch.nn.Linear(input_dim, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=max(5, patience // 4),
            min_lr=learning_rate * 1e-4,
        )

        linear_classifier, optimizer, train_loader, val_loader, test_loader = (
            lp_accelerator.prepare(
                linear_classifier, optimizer, train_loader, val_loader, test_loader
            )
        )

        best_val_loss = float("inf")
        best_train_acc = 0.0
        best_state_dict = None
        patience_counter = 0
        effective_patience = max_epochs + 1 if patience < 0 else patience

        for epoch in tqdm(
            range(max_epochs),
            desc=f"Fold {fold+1}/{n_experiments}",
            unit="epoch",
            disable=not lp_accelerator.is_main_process,
        ):
            # --- train ---
            linear_classifier.train()
            correct = torch.tensor(0, device=device)
            total = torch.tensor(0, device=device)

            for batch_X, batch_y in train_loader:
                outputs = linear_classifier(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                lp_accelerator.backward(loss)
                optimizer.step()

                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum()

            correct = lp_accelerator.reduce(correct, reduction="sum")
            total = lp_accelerator.reduce(total, reduction="sum")
            train_acc = (correct / total).item()

            # --- val ---
            linear_classifier.eval()
            total_val_loss = torch.tensor(0.0, device=device)
            val_batches = torch.tensor(0, device=device)
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_outputs = linear_classifier(batch_X)
                    total_val_loss += criterion(val_outputs, batch_y)
                    val_batches += 1

            total_val_loss = lp_accelerator.reduce(total_val_loss, reduction="sum")
            val_batches = lp_accelerator.reduce(val_batches, reduction="sum")
            val_loss = (total_val_loss / val_batches).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_state_dict = copy.deepcopy(
                    lp_accelerator.unwrap_model(linear_classifier).state_dict()
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= effective_patience:
                if lp_accelerator.is_main_process:
                    LOGGER.info(
                        f"Early stopping at epoch {epoch+1} | "
                        f"best_val_loss={best_val_loss:.4f}, best_train_acc={best_train_acc:.4f}"
                    )
                break

        # --- test ---
        lp_accelerator.unwrap_model(linear_classifier).load_state_dict(best_state_dict)
        linear_classifier.eval()
        with torch.no_grad():
            all_predicted = []
            all_labels = []
            for batch_X, batch_y in test_loader:
                outputs = linear_classifier(batch_X)
                _, predicted = outputs.max(1)
                all_predicted.append(lp_accelerator.gather_for_metrics(predicted).cpu())
                all_labels.append(lp_accelerator.gather_for_metrics(batch_y).cpu())

        if lp_accelerator.is_main_process:
            all_predicted = torch.cat(all_predicted).numpy()
            all_labels = torch.cat(all_labels).numpy()

            test_acc = (all_predicted == all_labels).mean()
            test_f1 = f1_score(
                all_labels, all_predicted, average="macro", zero_division=0
            )
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

        lp_accelerator.free_memory()

    lp_accelerator.wait_for_everyone()

    if lp_accelerator.is_main_process:
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

        return {
            "train": train_accs,  # list of n_experiments floats
            "test": test_accs,
            "f1": test_f1s,
            "precision": test_precisions,
            "recall": test_recalls,
            "confusion_matrix": mean_cm.tolist(),  # still averaged over folds within this call
            "confusion_matrix_normalized": norm_cm.tolist(),
        }

    return {
        "train": [],
        "test": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "confusion_matrix": [],
        "confusion_matrix_normalized": [],
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(
    backbone_type: str,
    backbone_params: dict,
    part_dim: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the backbone model from its type string and params dict."""
    if backbone_type == "JetTransformerEncoder":
        model = JetTransformerEncoder(part_dim=part_dim, **backbone_params)
    elif backbone_type == "JetMLP":
        model = JetMLP(input_dim=part_dim, **backbone_params)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type!r}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Inference for a single combination
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_combo(
    backbone_params: dict,
    config: dict,
    device: torch.device,
    dataloaders: dict,
    accelerator=None,
) -> dict:
    probe_cfg = config["inference"].get("acc_tests", {})
    n_experiments = config["inference"].get("n_experiments", 1)
    backbone_type = config["models"]["backbone"]["type"]

    # Raw lists accumulated across all model init runs
    raw: dict[str, list] = {}

    for split, (dataloader, dataloader_config) in dataloaders.items():
        particle_features = dataloader_config.outputs.sequence
        part_dim = len(particle_features)

        for run_idx in range(n_experiments):
            LOGGER.info(f"  --- Model init run {run_idx + 1}/{n_experiments} ---")

            model = build_model(backbone_type, backbone_params, part_dim, device)
            if accelerator is not None:
                model = accelerator.prepare(model)
            model.eval()

            results = {"rep": []}

            with torch.no_grad():
                with tqdm(
                    dataloader,
                    desc=f"Inference {split} (run {run_idx+1})",
                    unit="batch",
                    disable=accelerator is not None and not accelerator.is_main_process,
                ) as pbar:
                    for batch in pbar:
                        particles = torch.tensor(
                            batch["sequence"], dtype=torch.float32
                        ).to(device)

                        if backbone_type == "JetTransformerEncoder":
                            jets = torch.tensor(
                                batch["class_"], dtype=torch.float32
                            ).to(device)
                            mask = torch.tensor(batch["mask"], dtype=torch.bool).to(
                                device
                            )
                            rep, _ = model(particles=particles, jets=jets, mask=mask)
                        elif backbone_type == "JetMLP":
                            rep = model(particles, return_rep=True)
                        else:
                            raise ValueError(
                                f"Unknown backbone type: {backbone_type!r}"
                            )

                        results["rep"].append(rep.detach().cpu())

                        for aux_key, aux_val in batch["aux"].items():
                            results.setdefault(aux_key, []).append(
                                torch.tensor(aux_val).cpu()
                            )

            for key in results:
                if results[key]:
                    local = torch.cat(results[key], dim=0)
                    if accelerator is not None:
                        local = accelerator.gather_for_metrics(local.to(device)).cpu()
                    results[key] = local

            del model
            if accelerator is not None:
                accelerator.free_memory()
            else:
                torch.cuda.empty_cache()

            if accelerator is not None:
                accelerator.wait_for_everyone()

            if "label" not in results:
                LOGGER.warning(f"No labels in {split}, skipping accuracy tests.")
                continue

            if accelerator is None or accelerator.is_main_process:
                knn_cfg = probe_cfg.get("knn", {})
                if knn_cfg:
                    knn_metrics = train_knn(results, **knn_cfg)
                    for k, metrics in knn_metrics.items():
                        for metric in ("acc", "f1", "precision", "recall"):
                            raw.setdefault(f"knn_k{k}_{metric}", []).extend(
                                metrics[metric]
                            )
                        # confusion matrices: one per run, fold-averaged inside train_knn
                        if metrics["confusion_matrix"]:
                            raw.setdefault(f"knn_k{k}_confusion_matrices", []).append(
                                metrics["confusion_matrix"]
                            )
                            raw.setdefault(
                                f"knn_k{k}_confusion_matrices_normalized", []
                            ).append(metrics["confusion_matrix_normalized"])

            lp_cfg = probe_cfg.get("linear_probe", {})
            if lp_cfg:
                lp = train_linear_probe(results, **lp_cfg)
                if accelerator is None or accelerator.is_main_process:
                    for metric in ("train", "test", "f1", "precision", "recall"):
                        raw.setdefault(f"lp_{metric}", []).extend(lp[metric])
                    # Confusion matrices: keep one per run (already fold-averaged inside)
                    if lp["confusion_matrix"]:
                        raw.setdefault("lp_confusion_matrices", []).append(
                            lp["confusion_matrix"]
                        )
                        raw.setdefault("lp_confusion_matrices_normalized", []).append(
                            lp["confusion_matrix_normalized"]
                        )

    # -------------------------------------------------------------------------
    # Aggregate across all runs — only here, only once
    # -------------------------------------------------------------------------
    aggregated = {}
    if accelerator is None or accelerator.is_main_process:
        for metric_key, values in raw.items():
            if "confusion_matrices_normalized" in metric_key:
                mean_norm_cm = np.mean(values, axis=0)
                aggregated[metric_key] = {
                    "mean": mean_norm_cm.tolist(),
                    "n": len(values),
                }
            elif "confusion_matrices" in metric_key:
                mean_cm = np.mean(values, axis=0)
                aggregated[metric_key] = {
                    "mean": mean_cm.tolist(),
                    "n": len(values),
                }
            else:
                arr = np.array(values)
                aggregated[metric_key] = {
                    "acc": float(np.mean(arr)),
                    "err": float(np.std(arr)),
                    "n": len(arr),
                    "all": arr.tolist(),
                }

        scalar_summary = {
            k: f"{v['acc']:.4f} ± {v['err']:.4f} (n={v['n']})"
            for k, v in aggregated.items()
            if "confusion" not in k
        }
        LOGGER.info(
            f"Aggregated metrics over {n_experiments} model inits: {scalar_summary}"
        )

    return aggregated


def save_entry(entry: dict, output_dir: Path, idx: int) -> None:
    """Save a single combo result to its own yaml file."""
    combo_path = output_dir / f"combo_{idx:04d}.yaml"
    with open(combo_path, "w") as f:
        yaml.dump(entry, f, default_flow_style=False, sort_keys=False)
    LOGGER.info(f"  Saved combo {idx} to {combo_path}")


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------


def run_scan(config: dict, device: torch.device, accelerator=None):
    LOGGER.info(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")

    combinations = get_scan_combinations(config)
    LOGGER.info(f"Total combinations: {len(combinations)}")

    output_dir = get_output_dir(config)

    splits = config["inference"].get("splits", ["test_jetclass"])
    dataloaders = {
        split: get_dataloader_and_config(config=config, mode="inference", split=split)
        for split in splits
    }

    summary = []

    for idx, backbone_params in enumerate(combinations):
        LOGGER.info(
            f"===== Combo {idx + 1}/{len(combinations)}: {backbone_params} ====="
        )

        metrics = run_combo(
            backbone_params=backbone_params,
            config=config,
            device=device,
            dataloaders=dataloaders,
            accelerator=accelerator,
        )
        LOGGER.info(f"  metrics: {metrics}")

        entry = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in backbone_params.items()
        }
        entry.update(metrics)
        summary.append(entry)

        if accelerator is None or accelerator.is_main_process:
            save_entry(entry, output_dir, idx)

    if accelerator is None or accelerator.is_main_process:
        yaml_path = output_dir / "scan_summary.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        LOGGER.info(f"Scan summary saved to {yaml_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter scan (random init, linear probe)"
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-lv",
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("-lf", "--log-file", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    configure_logger(
        logger=LOGGER,
        name="DINO Scan",
        log_file=args.log_file,
        log_level=args.log_level,
    )

    use_accelerate = config.get("accelerate", False)
    if use_accelerate:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,
            split_batches=True,
        )
        kwargs = InitProcessGroupKwargs(timeout=timedelta(days=365))
        accelerator = Accelerator(
            dataloader_config=dataloader_config, kwargs_handlers=[kwargs]
        )
        device = accelerator.device
        LOGGER.addFilter(lambda record: accelerator.is_main_process)
        LOGGER.info(f"Using accelerate on device: {device}")
    else:
        accelerator = None
        device = config.get("device", None)
        device = torch.device(device) if device else get_available_device()
        LOGGER.info(f"Using device: {device}")

    run_scan(config, device, accelerator=accelerator)

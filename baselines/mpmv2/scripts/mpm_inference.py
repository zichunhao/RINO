import copy
import logging
from pathlib import Path

import hydra
import numpy as np
import rootutils
import torch as T
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path=str(root / "configs"), config_name="inference.yaml"
)
def main(cfg: DictConfig) -> None:
    log.info("Loading run information")
    orig_cfg = reload_original_config(ckpt_flag=cfg.ckpt_flag)

    # Override test_set path if specified in inference config
    if OmegaConf.select(cfg, "datamodule.test_set.path") is not None:
        new_path = cfg.datamodule.test_set.path
        log.info(f"Overriding test_set path: {new_path}")
        OmegaConf.update(orig_cfg, "datamodule.test_set.path", new_path, merge=True)

    log.info(f"Loading checkpoint: {orig_cfg.ckpt_path}")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cpu")
    model.eval()

    log.info("Setting up test datamodule")
    datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
    datamodule.setup("predict")
    test_loader = datamodule.test_dataloader()

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    model = model.to(device)

    reps, labels = [], []
    with T.no_grad():
        for batch in tqdm(test_loader, desc="Extracting representations"):
            data = {
                "csts": T.tensor(batch["csts"], dtype=T.float32).to(device),
                "csts_id": T.tensor(batch["csts_id"], dtype=T.long).to(device),
                "mask": T.tensor(batch["mask"], dtype=T.bool).to(device),
                "jets": T.tensor(batch["jets"], dtype=T.float32).to(device),
            }

            # Forward pass: no masking/null tokens (clean inference)
            x, new_mask = model(data)

            # Mean pool over valid tokens -> jet-level representation
            rep = (x * new_mask.unsqueeze(-1)).sum(dim=1)
            rep = rep / new_mask.sum(dim=1, keepdim=True).clamp(min=1)
            reps.append(rep.float().cpu())

            lbl = batch["labels"]
            if not isinstance(lbl, T.Tensor):
                lbl = T.tensor(lbl)
            labels.append(lbl.cpu())

    results = {
        "rep": T.cat(reps, dim=0),
        "label": T.cat(labels, dim=0),
    }

    log.info(f"Representation shape: {results['rep'].shape}")
    log.info(f"Unique labels: {T.unique(results['label']).tolist()}")
    log.info(f"Number of jets: {len(results['label'])}")

    output_dir = Path(orig_cfg.paths.full_path) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_reps.pt"
    T.save(results, output_path)
    log.info(f"Representations saved to {output_path}")

    # k-NN probe
    knn_cfg = OmegaConf.to_container(cfg.knn, resolve=True)
    log.info("Running k-NN probe")
    train_knn(results, **knn_cfg)

    # Linear probe
    linear_cfg = OmegaConf.to_container(cfg.linear_probe, resolve=True)
    log.info("Running linear probe")
    train_linear_probe(results, device=device, **linear_cfg)


def train_knn(
    results: dict[str, T.Tensor],
    k_list: list[int] = [1, 5, 10, 20, 100],
    labels_to_include: list[int] = [0, 8],
    sample_size: int = 200_000,
    n_experiments: int = 10,
) -> dict[int, dict]:
    rep = results["rep"].cpu().detach()
    label = results["label"].cpu().detach()

    if sample_size > 0:
        sample_size = min(sample_size, rep.shape[0])
        idx = T.randperm(rep.shape[0])[:sample_size]
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

    log.info(
        f"k-NN probe | k={k_list}, sample_size={sample_size}, "
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
        log.info(
            f"k-NN (k={k}) | acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | "
            f"f1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
            f"precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f} | "
            f"recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}"
        )

    return all_metrics


@T.enable_grad()
def train_linear_probe(
    results: dict[str, T.Tensor],
    n_experiments: int = 10,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 2048,
    sample_size: int = -1,
    labels_to_include: list[int] = [0, 8],
    patience: int = 15,
    val_fraction: float = 0.1,
    device: T.device = T.device("cuda" if T.cuda.is_available() else "cpu"),
) -> dict[str, list]:
    log.info(
        f"Linear probe | max_epochs={max_epochs}, lr={learning_rate}, "
        f"batch_size={batch_size}, sample_size={sample_size}, "
        f"n_experiments={n_experiments}, labels_to_include={labels_to_include}"
    )

    rep = results["rep"].detach()
    label = results["label"].detach()

    if sample_size > 0:
        sample_size = min(sample_size, rep.shape[0])
        idx = T.randperm(rep.shape[0])[:sample_size]
        rep = rep[idx]
        label = label[idx]

    labels_to_include_t = T.tensor(labels_to_include, dtype=label.dtype)
    mask = (label.unsqueeze(1) == labels_to_include_t).any(dim=1)
    rep = rep[mask]
    label = label[mask]

    labels_present = label.unique(sorted=True)
    label = T.searchsorted(labels_present, label)

    num_classes = len(np.unique(label.numpy()))
    input_dim = rep.shape[1]
    n = rep.shape[0]
    log.info(f"Linear probe: input_dim={input_dim}, num_classes={num_classes}, n={n}")

    train_accs, test_accs, test_f1s, test_precisions, test_recalls, conf_matrices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    fold_size = n // n_experiments
    T.manual_seed(42)
    perm = T.randperm(n)

    for fold in range(n_experiments):
        test_idx = perm[fold * fold_size : (fold + 1) * fold_size]
        train_val_idx = T.cat(
            [perm[: fold * fold_size], perm[(fold + 1) * fold_size :]]
        )

        val_size = max(1, int(len(train_val_idx) * val_fraction))
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

        X_train, y_train = rep[train_idx], label[train_idx]
        X_val, y_val = rep[val_idx], label[val_idx]
        X_test, y_test = rep[test_idx], label[test_idx]

        train_loader = T.utils.data.DataLoader(
            T.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        linear_classifier = T.nn.Linear(input_dim, num_classes).to(device)
        criterion = T.nn.CrossEntropyLoss()
        optimizer = T.optim.Adam(linear_classifier.parameters(), lr=learning_rate)
        scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(
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
            range(max_epochs), desc=f"Fold {fold + 1}/{n_experiments}", unit="epoch"
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
            with T.no_grad():
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
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        linear_classifier.load_state_dict(best_state_dict)
        linear_classifier.eval()
        with T.no_grad():
            all_predicted = T.cat(
                [
                    linear_classifier(bx.to(device)).max(1)[1].cpu()
                    for bx, _ in T.utils.data.DataLoader(
                        T.utils.data.TensorDataset(X_test, y_test),
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

        log.info(
            f"Fold {fold + 1} | acc={test_acc:.4f} | f1={test_f1:.4f} | "
            f"prec={test_prec:.4f} | rec={test_rec:.4f}"
        )

    mean_cm = np.mean(conf_matrices, axis=0)
    norm_cm = mean_cm / mean_cm.sum(axis=1, keepdims=True)

    log.info(
        f"Linear probe train accuracy:  {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}"
    )
    log.info(
        f"Linear probe test  accuracy:  {np.mean(test_accs):.4f}  ± {np.std(test_accs):.4f}"
    )
    log.info(
        f"Linear probe test  F1 (macro):{np.mean(test_f1s):.4f}   ± {np.std(test_f1s):.4f}"
    )
    log.info(
        f"Linear probe test  precision: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}"
    )
    log.info(
        f"Linear probe test  recall:    {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}"
    )
    log.info(f"Normalised confusion matrix:\n{np.round(norm_cm, 4)}")

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
    main()

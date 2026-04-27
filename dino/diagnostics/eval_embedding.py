from pathlib import Path
import argparse
import json

from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch

# knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# BDT
import xgboost as xgb

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix


def train_and_eval_bdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int = 1000,
    max_depth: int = 10,
    eta: float = 0.3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 10,
):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set hyperparameters for the model
    params = {
        # For multiclass classification (adjust if binary)
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_train)),  # Number of classes
        "max_depth": max_depth,
        "eta": eta,  # Learning rate
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        # Multiclass log loss and error rate
        "eval_metric": ["mlogloss", "merror"],
    }

    # Define which datasets to evaluate during training
    evallist = [(dtrain, "train"), (dtest, "validation")]

    # Train the model
    bst = xgb.train(
        params,
        dtrain,
        num_rounds,
        evallist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )

    # Evaluate the model
    y_pred = bst.predict(dtest)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm = (
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    )  # Normalize the confusion matrix

    return acc, cm


def train_and_eval_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int,
):
    knn = KNeighborsClassifier(n_neighbors=k)
    # train
    knn = knn.fit(X_train, y_train)
    # Predict
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate KNN and BDT models on embeddings for top vs QCD."
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[5, 20, 100],
        help="List of k values for KNN",
    )
    parser.add_argument(
        "-n",
        "--num-experiments",
        type=int,
        default=10,
        help="Number of experiments to average results over",
    )
    parser.add_argument(
        "--embedding-path", type=Path, required=True, help="Path to the embedding file"
    )
    parser.add_argument(
        "--include-leptonic",
        action="store_true",
        help="Whether to include leptonic top samples in the evaluation",
    )
    args = parser.parse_args()

    num_experiments = max(args.num_experiments, 1)

    data_path = Path(args.embedding_path)
    data = torch.load(data_path)
    rep = data["rep"]
    label = data["label"]

    if args.include_leptonic:
        tags = ["QCD", "Tbqq", "Tbl"]
    else:
        tags = ["QCD", "Tbqq"]

        mask = label != 9
        rep = rep[mask]
        label = label[mask]
    # label[label == 0] = 0
    label[label == 8] = 1
    label[label == 9] = 2

    # Convert to numpy
    rep = rep.cpu().numpy()
    label = label.cpu().numpy()

    # accs
    acc_dict = {}

    # knn
    knn_sample_size = rep.shape[0] // num_experiments
    print(f"Total samples: {rep.shape[0]}, Samples per experiment: {knn_sample_size}")
    np.random.seed(42)
    permuted_indices = np.random.permutation(rep.shape[0])  # k-fold indices
    knn_acc_dict = {k: [] for k in args.ks}

    for i in tqdm(range(num_experiments), desc="KNN Experiments"):
        start_idx = i * knn_sample_size
        end_idx = (i + 1) * knn_sample_size if i < num_experiments - 1 else rep.shape[0]
        idx = permuted_indices[start_idx:end_idx]

        rep_knn = rep[idx]
        label_knn = label[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            rep_knn, label_knn, test_size=0.2, random_state=42 + i
        )

        for k in args.ks:
            acc = train_and_eval_knn(X_train, y_train, X_test, y_test, k)
            knn_acc_dict[k].append(acc)

    # Compute statistics across all experiments
    for k, v in knn_acc_dict.items():
        acc = np.mean(v)
        std = np.std(v)
        acc_dict[f"knn_{k}"] = acc
        acc_dict[f"knn_{k}_std"] = std
        print(f"k-NN (k={k}) acc={acc:.4f} +/- {std:.4f}")

    # BDT
    bdt_accs = []
    bdt_cms = []
    for i in tqdm(range(num_experiments), desc="BDT Experiments"):
        start_idx = i * knn_sample_size
        end_idx = (i + 1) * knn_sample_size if i < num_experiments - 1 else rep.shape[0]
        idx = permuted_indices[start_idx:end_idx]

        rep_bdt = rep[idx]
        label_bdt = label[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            rep_bdt, label_bdt, test_size=0.2, random_state=42 + i
        )
        acc, cm = train_and_eval_bdt(X_train, y_train, X_test, y_test)

        bdt_accs.append(acc)
        bdt_cms.append(cm)

    acc = np.mean(bdt_accs)
    std = np.std(bdt_accs)
    print(f"BDT acc={acc:.4f} +/- {std:.4f}")
    acc_dict["bdt"] = acc
    acc_dict["bdt_std"] = std

    results_dir = data_path.parent / "embedding_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(results_dir / "accs.json", "w") as f:
        json.dump(acc_dict, f, indent=4)

    bdt_cm_mean = np.mean(bdt_cms, axis=0)
    # bdt_cm_std = np.std(bdt_cms, axis=0)

    plt.figure(figsize=(5, 5))
    sns.heatmap(bdt_cm_mean, annot=True, cmap="Blues", cbar=False, fmt=".3f")
    # plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=np.arange(len(tags)) + 0.5, labels=tags)
    plt.yticks(ticks=np.arange(len(tags)) + 0.5, labels=tags)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

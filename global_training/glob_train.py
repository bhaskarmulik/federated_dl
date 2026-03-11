import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dense neural network on MNIST fetched from OpenML."
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset reserved for testing.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Proportion of the remaining training data used for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="global_training/artifacts",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/global_model.pt",
        help="Path where the trained global model state dict will be stored.",
    )
    parser.add_argument(
        "--loader-output",
        type=str,
        default="global_training/artifacts/mnist_loaders.pt",
        help="File where serialized DataLoaders will be stored for clients.",
    )
    return parser.parse_args()


def load_mnist(
    batch_size: int = 128,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, DataLoader]:
    """Fetch MNIST from OpenML and prepare train/val/test data loaders."""
    features, targets = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False
    )
    features = features.astype(np.float32) / 255.0
    targets = targets.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=random_state,
    )

    def to_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x).to(torch.float32),
            torch.from_numpy(y).to(torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        "train": to_loader(X_train, y_train, shuffle=True),
        "val": to_loader(X_val, y_val, shuffle=False),
        "test": to_loader(X_test, y_test, shuffle=False),
    }


def save_model_state(model: nn.Module, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved trained model to: {output_path}")
    return output_path


def serialize_loaders(loaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, object]]:
    serialized: Dict[str, Dict[str, object]] = {}
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        if not isinstance(dataset, TensorDataset):
            raise TypeError(
                f"Only TensorDataset-backed loaders can be serialized (split '{split_name}' uses {type(dataset)!r})."
            )
        serialized[split_name] = {
            "tensors": [tensor.cpu() for tensor in dataset.tensors],
            "batch_size": loader.batch_size,
            "shuffle": isinstance(loader.sampler, RandomSampler),
        }
    return serialized


def save_serialized_loaders(loaders: Dict[str, DataLoader], output_path: Path) -> Path:
    payload = serialize_loaders(loaders)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved serialized DataLoaders to: {output_path}")
    return output_path


def load_serialized_loaders(payload_path: Path) -> Dict[str, DataLoader]:
    saved_payload: Dict[str, Dict[str, object]] = torch.load(payload_path, map_location="cpu")
    restored: Dict[str, DataLoader] = {}
    for split_name, cfg in saved_payload.items():
        tensors = cfg["tensors"]
        dataset = TensorDataset(*tensors)
        restored[split_name] = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=bool(cfg.get("shuffle", False)),
        )
    return restored


class DenseClassifier(nn.Module):
    """Simple feed-forward network for MNIST classification."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def epoch_metrics(
    preds: List[np.ndarray],
    targets: List[np.ndarray],
) -> Dict[str, float]:
    y_true = np.concatenate(targets).astype(np.int64)
    y_pred = np.concatenate(preds).astype(np.int64)
    return {
        "accuracy": (y_true == y_pred).mean(),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int = 15,
    learning_rate: float = 1e-3,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_r2": [],
        "val_r2": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y in loaders["train"]:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_preds.append(outputs.argmax(dim=1).cpu().numpy())
            train_targets.append(batch_y.cpu().numpy())

        train_loss /= len(loaders["train"].dataset)
        epoch_train_metrics = epoch_metrics(train_preds, train_targets)

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in loaders["val"]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_preds.append(outputs.argmax(dim=1).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_loss /= len(loaders["val"].dataset)
        epoch_val_metrics = epoch_metrics(val_preds, val_targets)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(epoch_train_metrics["accuracy"])
        history["val_accuracy"].append(epoch_val_metrics["accuracy"])
        history["train_rmse"].append(epoch_train_metrics["rmse"])
        history["val_rmse"].append(epoch_val_metrics["rmse"])
        history["train_r2"].append(epoch_train_metrics["r2"])
        history["val_r2"].append(epoch_val_metrics["r2"])
        history["train_f1"].append(epoch_train_metrics["f1"])
        history["val_f1"].append(epoch_val_metrics["f1"])

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Acc: {epoch_val_metrics['accuracy']:.4f}, "
            f"Val RMSE: {epoch_val_metrics['rmse']:.4f}, "
            f"Val R2: {epoch_val_metrics['r2']:.4f}, "
            f"Val F1: {epoch_val_metrics['f1']:.4f}"
        )

    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds.append(outputs.argmax(dim=1).cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    total_loss /= len(loader.dataset)
    metrics = epoch_metrics(preds, targets)
    y_true = np.concatenate(targets).astype(np.int64)
    y_pred = np.concatenate(preds).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    metrics["loss"] = total_loss
    return metrics, cm


def plot_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_accuracy"]) + 1)

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.set_title("Training vs Validation Loss")
    ax_loss.legend()
    fig_loss.tight_layout()
    loss_path = output_dir / "loss_history.png"
    fig_loss.savefig(loss_path, dpi=300)
    plt.close(fig_loss)

    fig, axes = plt.subplots(4, 1, figsize=(8, 16), sharex=True)
    axes[0].plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    axes[0].plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_rmse"], marker="o", label="Train RMSE")
    axes[1].plot(epochs, history["val_rmse"], marker="o", label="Validation RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Root Mean Squared Error")
    axes[1].legend()

    axes[2].plot(epochs, history["train_r2"], marker="o", label="Train R2")
    axes[2].plot(epochs, history["val_r2"], marker="o", label="Validation R2")
    axes[2].set_ylabel("R2 Score")
    axes[2].set_title("R2 Score")
    axes[2].legend()

    axes[3].plot(epochs, history["train_f1"], marker="o", label="Train F1")
    axes[3].plot(epochs, history["val_f1"], marker="o", label="Validation F1")
    axes[3].set_ylabel("F1 Score")
    axes[3].set_xlabel("Epoch")
    axes[3].set_title("Macro F1 Score")
    axes[3].legend()

    fig.tight_layout()
    metric_path = output_dir / "metric_history.png"
    fig.savefig(metric_path, dpi=300)
    plt.close(fig)

    print(f"Saved loss history to: {loss_path}")
    print(f"Saved metric history to: {metric_path}")


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path, class_names: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loaders = load_mnist(
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    model = DenseClassifier()
    model.to(device)

    history = train_model(
        model,
        loaders,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    test_metrics, confusion_mtx = evaluate_model(model, loaders["test"], device)

    model_output_path = save_model_state(model, Path(args.model_output))
    loader_output_path = save_serialized_loaders(loaders, Path(args.loader_output))

    print(
        "Test Metrics -> "
        f"Loss: {test_metrics['loss']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}, "
        f"RMSE: {test_metrics['rmse']:.4f}, "
        f"R2: {test_metrics['r2']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}"
    )

    output_dir = Path(args.output_dir)
    plot_history(history, output_dir)
    plot_confusion_matrix(confusion_mtx, output_dir, [str(i) for i in range(confusion_mtx.shape[0])])

    print(f"Artifacts saved: model -> {model_output_path}, loaders -> {loader_output_path}")


if __name__ == "__main__":
    main()

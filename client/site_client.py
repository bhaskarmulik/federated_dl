import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import grpc
import torch
from torch.utils.data import DataLoader
import sys

from proto import fl_pb2_grpc
from proto.fl_pb2 import Ack, ClientInfo, LocalUpdate, ModelBlob, PullGlobalRequest

from .model_io import bytes_to_state_dict, state_dict_to_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from global_training import DenseClassifier, load_serialized_loaders
    IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  
    IMPORT_ERROR = exc
    DenseClassifier = None
    load_serialized_loaders = None

MAX_MB = 256
CHANNEL_OPTS = [
    ("grpc.max_receive_message_length", MAX_MB * 1024 * 1024),
    ("grpc.max_send_message_length", MAX_MB * 1024 * 1024),
]

DEFAULT_MODEL_PATH = Path("models/global_model.pt")
DEFAULT_LOADER_PATH = Path("global_training/artifacts/mnist_loaders.pt")


def grpc_channel(target: str = "localhost:50051", use_tls: bool = False) -> grpc.Channel:
    if use_tls:
        with open("tls/ca.crt", "rb") as f:
            ca = f.read()
        with open("tls/client.crt", "rb") as f:
            cert = f.read()
        with open("tls/client.key", "rb") as f:
            key = f.read()
        creds = grpc.ssl_channel_credentials(
            root_certificates=ca, private_key=key, certificate_chain=cert
        )
        return grpc.secure_channel(target, creds, options=CHANNEL_OPTS)
    return grpc.insecure_channel(target, options=CHANNEL_OPTS)


def pull_global(
    stub: fl_pb2_grpc.FederatedStub,
    client_info: ClientInfo,
    round_id: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
):
    resp = stub.PullGlobal(PullGlobalRequest(client=client_info, round_id=round_id or ""))
    if model is not None:
        bytes_to_state_dict(model, resp.global_model.weights)
    return resp


def push_update(
    stub: fl_pb2_grpc.FederatedStub,
    client_info: ClientInfo,
    round_id: str,
    model: torch.nn.Module,
    num_samples: int,
    metrics: Optional[Dict[str, float]] = None,
) -> Ack:
    weights = state_dict_to_bytes(model)
    blob = ModelBlob(weights=weights, format="torch_state_dict")
    upd = LocalUpdate(
        client=client_info,
        round_id=round_id,
        update=blob,
        num_samples=num_samples,
        metric_json=json.dumps(metrics or {}),
    )
    ack = stub.PushUpdate(upd)
    return ack


def bootstrap_from_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    loader_path: Path = DEFAULT_LOADER_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[torch.nn.Module], Dict[str, DataLoader]]:
    """Attempt to load a pretrained model and serialized DataLoaders."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DenseClassifier is None:
        if IMPORT_ERROR is not None:
            print(
                "[Client] DenseClassifier import failed. Ensure the project root is on PYTHONPATH "
                f"or run via 'python -m client.site_client'. Error: {IMPORT_ERROR}"
            )
        return None, {}

    model: Optional[torch.nn.Module] = None
    loaders: Dict[str, DataLoader] = {}

    try:
        model = DenseClassifier()
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[Client] Loaded model weights from {model_path}")
        else:
            print(f"[Client] No model weights found at {model_path}; will rely on server pull.")
        model.to(device)
    except Exception as exc:  
        print(f"[Client] Warning: failed to prepare DenseClassifier from artifacts: {exc}")
        model = None

    if load_serialized_loaders is not None and loader_path.exists():
        try:
            loaders = load_serialized_loaders(loader_path)
            print(f"[Client] Loaded serialized loaders from {loader_path}")
        except Exception as exc:  
            print(f"[Client] Warning: failed to load serialized loaders: {exc}")
            loaders = {}
    else:
        if load_serialized_loaders is None:
            print("[Client] Serialized loader utility unavailable; skipping loader bootstrap.")
        else:
            print(f"[Client] No serialized loaders found at {loader_path}.")

    return model, loaders
def local_train_one_round(model, dataloader, plan):
    if dataloader is None:
        return model, 0, {}

    device = next(model.parameters()).device
    model.train()

    optimizer_name = (getattr(plan, "optimizer", "") or "sgd").lower()
    lr = getattr(plan, "lr", 0.01) or 0.01
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    epochs = max(int(getattr(plan, "local_epochs", 1) or 1), 1)
    send_metrics = bool(getattr(plan, "send_metrics", True))

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(epochs):
        for batch in dataloader:
            if isinstance(batch, dict):
                inputs = batch.get("x") or batch.get("inputs")
                targets = batch.get("y") or batch.get("labels")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 0:
                    continue
                if len(batch) == 1:
                    inputs, targets = batch[0], None
                else:
                    inputs, targets = batch[0], batch[1]
            else:
                raise TypeError("Unsupported batch type from dataloader")

            if targets is None:
                raise ValueError("Targets are required for training")

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if send_metrics:
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()

    if total_samples == 0:
        return model, 0, {}

    metrics = {}
    if send_metrics:
        metrics["loss"] = total_loss / total_samples
        metrics["acc"] = total_correct / total_samples
    return model, total_samples, metrics


def run_client(target: str = "localhost:50051", use_tls: bool = False):
    print(f"[Client] Starting. Target={target} TLS={use_tls}")
    channel = grpc_channel(target, use_tls)
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        print(f"[Client] Unable to reach server at {target} within timeout.")
        return

    stub = fl_pb2_grpc.FederatedStub(channel)

    client = ClientInfo(
        client_id=os.environ.get("CLIENT_ID", "site-A"),
        dataset="mnist",
        model_name="dense_classifier" if DenseClassifier else "resnet18",
        framework="torch",
        version="1.0.0",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaders = bootstrap_from_artifacts(device=device)
    if model is None:
        print("[Client] DenseClassifier artifacts unavailable; falling back to ResNet18.")
        import torchvision.models as models

        model = models.resnet18(num_classes=2)
        model.to(device)
        loaders = {}

    resp = pull_global(stub, client, model=model)
    round_id = resp.plan.round_id
    print(f"[Client] Pulled global for round {round_id}")

    train_loader = loaders.get("train") if loaders else None
    if train_loader is None:
        print("[Client] No train loader available; skipping local training.")
    else:
        print(
            f"[Client] Starting local training for {len(train_loader.dataset)} samples, "
            f"batch_size={train_loader.batch_size}."
        )
    model, num_samples, metrics = local_train_one_round(model, train_loader, resp.plan)

    if num_samples == 0:
        print("[Client] No local samples were trained; sending empty update.")
    else:
        print(
            f"[Client] Finished local training: samples={num_samples}, "
            f"metrics={json.dumps(metrics)}"
        )

    ack = push_update(stub, client, round_id, model, num_samples, metrics)
    print("[Client] PushUpdate:", ack.status, ack.message)


if __name__ == "__main__":
    run_client()

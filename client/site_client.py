import os, json, grpc, torch
from proto.fl_pb2 import ClientInfo, PullGlobalRequest, LocalUpdate, ModelBlob
from proto import fl_pb2_grpc

from . model_io import bytes_to_state_dict, state_dict_to_bytes

MAX_MB = 256
CHANNEL_OPTS = [
    ('grpc.max_receive_message_length', MAX_MB * 1024 * 1024),
    ('grpc.max_send_message_length', MAX_MB * 1024 * 1024),
]

def grpc_channel(target="localhost:50051", use_tls=False):
    if use_tls:
        with open("tls/ca.crt","rb") as f: ca = f.read()
        with open("tls/client.crt","rb") as f: cert = f.read()
        with open("tls/client.key","rb") as f: key = f.read()
        creds = grpc.ssl_channel_credentials(root_certificates=ca,
                                             private_key=key,
                                             certificate_chain=cert)
        return grpc.secure_channel(target, creds, options=CHANNEL_OPTS)
    return grpc.insecure_channel(target, options=CHANNEL_OPTS)

def pull_global(stub, client_info, round_id=None, model=None):
    resp = stub.PullGlobal(PullGlobalRequest(client=client_info, round_id=round_id or ""))
    if model is not None:
        bytes_to_state_dict(model, resp.global_model.weights)
    return resp

def push_update(stub, client_info, round_id, model, num_samples, metrics=None):
    weights = state_dict_to_bytes(model)
    blob = ModelBlob(weights=weights, format="torch_state_dict")
    upd = LocalUpdate(client=client_info, round_id=round_id, update=blob,
                      num_samples=num_samples,
                      metric_json=json.dumps(metrics or {}))
    ack = stub.PushUpdate(upd)
    return ack

# ------------------------
# Example training routine
# ------------------------
def local_train_one_round(model, dataloader, plan):
    # TODO: replace with your actual pipeline
    model.train()
    # dummy: no-op training for demo
    metrics = {"loss": 0.0, "acc": 0.0}
    num_samples = 100  # set to len(dataset) actually used
    return model, num_samples, metrics

def run_client(target="localhost:50051", use_tls=False):
    channel = grpc_channel(target, use_tls)
    stub = fl_pb2_grpc.FederatedStub(channel)

    client = ClientInfo(
        client_id=os.environ.get("CLIENT_ID","site-A"),
        dataset="breast-cancer",
        model_name="resnet18",
        framework="torch",
        version="1.0.0"
    )

    # 1) Build your local model here
    import torchvision.models as models
    model = models.resnet18(num_classes=2)
    # 2) Pull global
    resp = pull_global(stub, client, model=model)
    round_id = resp.plan.round_id
    print(f"[Client] Pulled global for round {round_id}")

    # 3) Train locally
    dummy_loader = []  # replace with your dataloader
    model, n, metrics = local_train_one_round(model, dummy_loader, resp.plan)

    # 4) Push update
    ack = push_update(stub, client, round_id, model, n, metrics)
    print("[Client] PushUpdate:", ack.status, ack.message)

if __name__ == "__main__":
    run_client()

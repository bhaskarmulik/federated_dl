import os, json, grpc, time, threading
from concurrent import futures
from proto import fl_pb2, fl_pb2_grpc

from .fedavg import fedavg, sd_to_bytes, bytes_to_sd

# In-memory round state (replace with Redis/DB in prod)
ROUND_ID = "r-0001"
GLOBAL_SD_BYTES = None
LOCK = threading.Lock()
PENDING_UPDATES = []   # list[(state_dict, num_samples)]

MAX_MB = 256
SERVER_OPTIONS = [
    ('grpc.max_receive_message_length', MAX_MB * 1024 * 1024),
    ('grpc.max_send_message_length', MAX_MB * 1024 * 1024),
]

def load_initial_global(path="models/global_model.pt"):
    import os, io, torch, torchvision.models as models
    global GLOBAL_SD_BYTES
    if os.path.exists(path):
        with open(path, "rb") as f:
            GLOBAL_SD_BYTES = f.read()
        print(f"[Server] Loaded global weights from {path}")
        return
    # Fallback: init a fresh model so clients can start round 1
    print(f"[Server] {path} not found. Initializing fresh global model.")
    model = models.resnet18(num_classes=2)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    GLOBAL_SD_BYTES = buf.getvalue()


class FederatedService(fl_pb2_grpc.FederatedServicer):
    def PullGlobal(self, request, context):
        plan = fl_pb2.TrainPlan(
            round_id=ROUND_ID, optimizer="adam", lr=1e-4, local_epochs=1,
            batch_size=16, send_metrics=True)
        blob = fl_pb2.ModelBlob(weights=GLOBAL_SD_BYTES, format="torch_state_dict")
        return fl_pb2.PullGlobalResponse(global_model=blob, plan=plan)

    def PushUpdate(self, request, context):
        try:
            sd = bytes_to_sd(request.update.weights)
            with LOCK:
                PENDING_UPDATES.append((sd, request.num_samples))
            status = "OK"
            msg = f"Update accepted from {request.client.client_id} for {request.round_id}"
            return fl_pb2.Ack(round_id=request.round_id, status=status, message=msg)
        except Exception as e:
            return fl_pb2.Ack(round_id=request.round_id, status="ERROR", message=str(e))

    def PullGlobalStream(self, request, context):
        # chunk GLOBAL_SD_BYTES
        CHUNK = 2 * 1024 * 1024
        total = len(GLOBAL_SD_BYTES)
        seq = 0
        for i in range(0, total, CHUNK):
            seq += 1
            yield fl_pb2.Chunk(
                round_id=ROUND_ID, client_id=request.client.client_id,
                seq=seq, data=GLOBAL_SD_BYTES[i:i+CHUNK], last=(i+CHUNK>=total)
            )

    def PushUpdateStream(self, request_iterator, context):
        buf = bytearray()
        client_id, round_id = None, None
        try:
            for chunk in request_iterator:
                client_id = client_id or chunk.client_id
                round_id = round_id or chunk.round_id
                buf += chunk.data
            sd = bytes_to_sd(bytes(buf))
            with LOCK:
                PENDING_UPDATES.append((sd, 0))   # if unknown samples, treat as equal
            return fl_pb2.Ack(round_id=round_id or "", status="OK", message=f"Streamed update from {client_id}")
        except Exception as e:
            return fl_pb2.Ack(round_id=round_id or "", status="ERROR", message=str(e))

    def Heartbeat(self, request, context):
        return fl_pb2.Ack(round_id=ROUND_ID, status="OK", message="alive")

def aggregate_periodically(interval_sec=30):
    global GLOBAL_SD_BYTES, ROUND_ID, PENDING_UPDATES
    while True:
        time.sleep(interval_sec)
        with LOCK:
            if len(PENDING_UPDATES) == 0:
                continue
            try:
                agg_sd = fedavg(PENDING_UPDATES)
                GLOBAL_SD_BYTES = sd_to_bytes(agg_sd)
                PENDING_UPDATES = []
                # increment round
                rid = int(ROUND_ID.split("-")[-1])
                ROUND_ID = f"r-{rid+1:04d}"
                print(f"[Aggregator] Aggregated, new {ROUND_ID}, broadcast-ready.")
            except Exception as e:
                print("[Aggregator] aggregation error:", e)

def serve(bind_addr="0.0.0.0:50051", use_tls=False):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=SERVER_OPTIONS)
    fl_pb2_grpc.add_FederatedServicer_to_server(FederatedService(), server)

    if use_tls:
        with open("tls/server.crt","rb") as f: cert = f.read()
        with open("tls/server.key","rb") as f: key = f.read()
        with open("tls/ca.crt","rb") as f: ca = f.read()
        creds = grpc.ssl_server_credentials(((key, cert),), root_certificates=ca, require_client_auth=True)
        server.add_secure_port(bind_addr, creds)
    else:
        server.add_insecure_port(bind_addr)

    load_initial_global()
    threading.Thread(target=aggregate_periodically, daemon=True).start()
    server.start()
    print(f"[Server] listening on {bind_addr} TLS={use_tls}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()

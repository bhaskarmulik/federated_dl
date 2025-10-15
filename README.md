# flkit — Federated Learning Framework (MVP scaffold)

This is a minimal scaffold for the cross-silo federated learning framework with:
- Centralized (sync + buffered async) **FedAvg**
- **P2P PushSum** (async) mode
- **Two-phase commit** secure aggregation (educational)
- GPU-backed ops using **PyTorch tensors** (autograd default, with option for custom autograd)

> Generated as a starting point. Fill in TODOs and run the `scripts` commands.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Compile protobufs (install protoc separately)
python -m grpc_tools.protoc -I flkit/proto --python_out=flkit/proto --grpc_python_out=flkit/proto flkit/proto/fl.proto

# Centralized FedAvg (synchronous baseline)
python scripts/run_cli.py sim:centralized --clients 10 --alpha 0.5 --sync --epochs 1 --batch-size 64 --optimizer adam --lr 1e-3

# Centralized async + secure agg (2PC) demo flags
python scripts/run_cli.py sim:centralized --clients 10 --alpha 0.5 --async --B 6 --T 3000 --Smax 2 --secure-agg --two-phase-commit

# P2P PushSum demo (neighbors from config)
python scripts/run_cli.py sim:p2p --clients 10 --alpha 0.5 --mix-interval 0.5
```

## Layout
```
flkit/
  core/          # layers, losses, optimizers, vectorization
  data/          # loaders, partitioners
  client/        # client agent, local trainer
  coordinator/   # rounds, aggregator, secure-agg
  p2p/           # PushSum worker
  proto/         # .proto (compile to *_pb2.py)
  security/      # mTLS helpers, PRG for masks
  sim/           # non-IID partitioner, simulator
  dash/          # minimal dashboard (FastAPI + WS)
  store/         # sqlite registry
  scripts/       # certs, cli
  configs/       # YAML configs
```

## Notes
- This scaffold uses PyTorch **tensors** for compute (`torch.Tensor`) with autograd by default.
- The protobuf-generated files are not checked-in; compile from `fl.proto` using `grpc_tools.protoc`.
- Security features (mTLS, secure-aggregation) are simplified for education; do not use as-is in production.

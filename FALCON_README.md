# FALCON

**Federated Adaptive Learning for Collaborative Omni-detection in Networks**

A privacy-preserving federated learning framework for anomaly detection in medical imaging, powered by **picograd** — a from-scratch deep learning framework built with only NumPy.

[![Tests](https://img.shields.io/badge/tests-113%2F113%20passing-brightgreen)](#testing)
[![Dependency](https://img.shields.io/badge/core%20dependency-numpy%20only-orange)](#requirements)

---

## What FALCON Does

```
Hospital A  ──┐
Hospital B  ──┼──► FL Server (FedAvg / FedBN / FedProx)
Hospital C  ──┘         │
                         ▼
                  Global AnomalyAE ← CNN-Autoencoder (253K params)
                         │
              ┌──────────┼──────────┐
         DP-SGD +    Secure Agg   Grad-CAM
         RDP acct    (masks)      Heatmaps
              └──────────┴──────────┘
                         │
              React Dashboard (live WebSocket)
```

**Key features:**
- `picograd` — complete from-scratch DL: Tensor, autograd DAG, Conv2d, BatchNorm, 45+ ops
- CNN-Autoencoder anomaly detection (train on normal only, detect by reconstruction error)
- DP-SGD with Rényi DP accounting (Mironov 2017) — no Opacus dependency
- Masking-based Secure Aggregation (Bonawitz et al. 2017)
- FedAvg / FedProx / FedBN aggregation strategies
- Dirichlet non-IID data partitioning (α-controllable)
- Grad-CAM adapted for autoencoders (clinical explainability)
- Real-time React dashboard with WebSocket live feed
- Docker-compose deployment with scalable clients

---

## Requirements

**Core (everything runs on this alone):**
```
python >= 3.10
numpy >= 1.24
```

**Dashboard (optional):**
```
flask flask-cors flask-socketio eventlet
```

**Real medical data (optional — synthetic fallback included):**
```
medmnist Pillow
```

**Docker (optional):**
```
docker >= 24.0  +  docker-compose >= 2.20
```

---

## Quick Start

```bash
# 1. Install the only required dependency
pip install numpy

# 2. Verify everything works — should show 113/113 PASS
python3 run_tests.py

# 3. Run the FL demo (5 rounds, 3 clients, with DP)
bash scripts/launch_demo.sh --rounds 5 --clients 3 --dp
```

No GPU. No CUDA. No complex setup.

---

## Project Structure

```
FALCON/
├── picograd/                   ← From-scratch DL framework
│   ├── backend/                ← NumpyBackend (swappable to Triton/CUDA)
│   ├── autograd/               ← DAG-based autograd engine
│   ├── ops/                    ← 45+ differentiable ops
│   ├── nn/                     ← Module, layers, fused losses
│   ├── optim/                  ← SGD, Adam, AdamW, schedulers
│   ├── data/                   ← DataLoader + MedMNIST bridge
│   ├── models/                 ← AnomalyAE, DenseClassifier, LeNet5
│   ├── explain/                ← Grad-CAM for autoencoders
│   └── privacy/                ← DPOptimizer + RDPAccountant
│
├── server/                     ← FL server
│   ├── fedavg.py               ← FedAvg (numpy only, no torch)
│   ├── aggregator.py           ← Full aggregator (FedAvg/FedBN/SecAgg/DP)
│   ├── grpc_server.py          ← InProcessFLServer + gRPC wrapper
│   ├── secure_agg.py           ← Pairwise masking protocol
│   └── strategies/             ← FedProxLoss, fedbn_aggregate
│
├── client/
│   └── site_client.py          ← FL client (local train + DP + FedProx)
│
├── falcon/
│   └── data_partition.py       ← Dirichlet / pathological partitioning
│
├── dashboard/
│   ├── backend/app.py          ← Flask REST + Socket.IO
│   ├── backend/services/       ← MetricsStore + FLDashboardBridge
│   └── frontend/src/           ← React SPA (4 pages + live charts)
│
├── docker/docker-compose.yml   ← Full distributed deployment
├── proto/fl.proto              ← gRPC service definition
├── scripts/
│   ├── train_falcon.py         ← Main training entry point
│   └── launch_demo.sh          ← One-command demo
├── tests/                      ← 113 tests (7 suites)
└── run_tests.py                ← Master test runner
```

---

## Running the Project

### Option 1 — One-Command Demo

```bash
bash scripts/launch_demo.sh --rounds 5 --clients 3
bash scripts/launch_demo.sh --rounds 10 --clients 5 --dp --secure-agg
bash scripts/launch_demo.sh --rounds 10 --clients 4 --alpha 0.1   # severe non-IID
```

### Option 2 — Full Training Script

```bash
# Quick smoke test
python3 scripts/train_falcon.py --quick

# Standard FL with DP + SecAgg
python3 scripts/train_falcon.py \
    --rounds 20 --clients 3 \
    --dataset BreastMNIST \
    --strategy fedavg \
    --latent-dim 64 --epochs 2 --lr 1e-3 \
    --alpha 0.5 --dp --secure-agg \
    --save-model ./global_model.pkl

# FedProx on severely non-IID data
python3 scripts/train_falcon.py \
    --rounds 20 --clients 5 --strategy fedprox --alpha 0.1 --dp

# Cross-modality FedBN
python3 scripts/train_falcon.py \
    --rounds 15 --clients 3 --strategy fedbn --dataset PneumoniaMNIST
```

### Option 3 — With Live Dashboard

```bash
# Install dashboard deps
pip install flask flask-cors flask-socketio eventlet

# Terminal 1: Train with dashboard bridge
python3 scripts/train_falcon.py --rounds 20 --clients 3 --dp --dashboard

# Terminal 2: Dashboard server
python3 -m dashboard.backend.app

# Open: http://localhost:5000
```

### Option 4 — Docker (Full Distributed)

```bash
# Build all images
docker-compose -f docker/docker-compose.yml build

# Start: server + 2 clients + dashboard
docker-compose -f docker/docker-compose.yml up

# Scale to 5 hospital clients
docker-compose -f docker/docker-compose.yml up --scale falcon-client=5

# Cross-modality (Breast + Pneumonia + Retina)
docker-compose -f docker/docker-compose.yml --profile cross-modality up

# Dashboard: http://localhost:3000 | gRPC: localhost:50051
docker-compose -f docker/docker-compose.yml down
```

---

## Testing

```bash
# All 113 tests
python3 run_tests.py

# Specific module
python3 run_tests.py --module 1      # Tensor + Autograd
python3 run_tests.py --module 5      # Privacy & Security
python3 run_tests.py --module 6,7    # AnomalyAE + FL

# Individual suites
python3 tests/test_tensor_autograd.py   # 17 tests — autograd, gradcheck
python3 tests/test_ops.py              # 22 tests — conv, norm, activations
python3 tests/test_nn_training.py      # 19 tests — layers, XOR convergence
python3 tests/test_privacy.py          # 11 tests — DP, RDP, SecAgg
python3 tests/test_models_fl.py        # 18 tests — AnomalyAE, FedAvg
python3 tests/test_dashboard.py        # 17 tests — MetricsStore, bridge
python3 tests/test_integration.py      # 9  tests — full pipeline
```

Expected:
```
TOTAL: 113/113 tests passed  ✓ ALL PASS
```

---

## Using picograd

```python
import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor

# Autograd
x = Tensor([3.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)   # tensor([6.0])

# Model
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10)
)
optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

loss = criterion(model(Tensor(images)), Tensor(labels))
loss.backward()
optimizer.step()

# Swap backend (future GPU)
picograd.set_backend(picograd.NumpyBackend())   # → picograd.set_backend(TritonBackend())

# Serialize
picograd.save(model.state_dict(), 'model.pkl')
model.load_state_dict(picograd.load('model.pkl'))
```

---

## Dataset Setup

MedMNIST downloads automatically when installed:

```bash
pip install medmnist Pillow

python3 scripts/train_falcon.py --dataset BreastMNIST    # ultrasound
python3 scripts/train_falcon.py --dataset PneumoniaMNIST  # chest X-ray
python3 scripts/train_falcon.py --dataset RetinaMNIST     # fundus
```

Without MedMNIST, all scripts fall back to synthetic 28×28 Gaussian data automatically.

---

## References

| Paper | Where used |
|-------|-----------|
| McMahan et al. (2017) — Communication-Efficient Learning (FedAvg) | `server/fedavg.py` |
| Abadi et al. (2016) — Deep Learning with Differential Privacy | `picograd/privacy/dp.py` |
| Mironov (2017) — Rényi Differential Privacy | `picograd/privacy/accountant.py` |
| Bonawitz et al. (2017) — Practical Secure Aggregation | `server/secure_agg.py` |
| Selvaraju et al. (2017) — Grad-CAM | `picograd/explain/gradcam.py` |
| Li et al. (2020) — FedProx | `server/strategies/fedprox_fedbn.py` |
| Li et al. (2021) — FedBN | `server/strategies/fedprox_fedbn.py` |
| Yang et al. (2023) — MedMNIST v2 | `picograd/data/medmnist_dataset.py` |

---

## Design Decisions

**picograd over PyTorch** — Every gradient is inspectable. Backend abstraction makes GPU support a single-line swap (`set_backend(TritonBackend())`).

**RDP over basic composition** — For 100 rounds, σ=1.0, q=0.01: RDP gives ε≈1.5 vs basic composition's ε≈10. A 6.7× tighter privacy bound.

**Reconstruction-error anomaly detection** — Trains only on normal samples (abundant, unlabeled). No labeled anomalies needed. Produces continuous scores, not hard labels.

**im2col convolution** — Converts convolution to one BLAS matmul. Same approach as PyTorch's CUDA kernels. 50ms/forward on CPU, no GPU needed.

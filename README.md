# FALCON — Federated Adaptive Learning for Collaborative Omni-detection in Networks

**Final Year Major Project** — Privacy-preserving federated framework for secure collaborative anomaly detection in medical imaging, powered by **picograd** (a from-scratch deep learning framework with DAG-based autograd).

---

## Quick Start

```bash
# Install the only dependency
pip install numpy

# Run all tests (87/87 pass)
python3 run_tests.py

# Run a specific module
python3 run_tests.py --module 5     # Privacy tests only
python3 run_tests.py --module 6,7   # AnomalyAE + FL tests
```

---

## Architecture

```
FALCON
├── picograd/               ← From-scratch DL framework (Modules 1–4)
│   ├── backend/            ← Backend ABC + NumPy implementation
│   ├── autograd/           ← DAG-based reverse-mode autograd engine
│   ├── ops/                ← All differentiable operations
│   │   ├── elemwise.py     ← +, -, *, /, pow, exp, log, ...
│   │   ├── matmul.py       ← MatMul (2D + batched)
│   │   ├── convolution.py  ← Conv2d (im2col) + ConvTranspose2d
│   │   ├── pooling.py      ← MaxPool2d, AvgPool2d
│   │   ├── normalization.py← BatchNorm, LayerNorm
│   │   └── activations.py  ← ReLU, GELU, Sigmoid, Softmax, ...
│   ├── nn/                 ← Module system + all layers + losses
│   ├── optim/              ← SGD, Adam, AdamW, RMSprop + LR schedulers
│   ├── data/               ← Dataset, DataLoader, samplers
│   ├── models/             ← AnomalyAE, DenseClassifier, LeNet5
│   ├── explain/            ← Grad-CAM + ExplainabilityReport
│   └── privacy/            ← DPOptimizer + RDPAccountant (Module 5)
│
├── server/                 ← FL server (Module 7)
│   ├── fedavg.py           ← FedAvg + fedavg_delta
│   ├── aggregator.py       ← Aggregator (FedAvg/FedBN + SecAgg + DP)
│   ├── secure_agg.py       ← MaskGenerator + SecureAggregator
│   └── strategies/         ← FedProxLoss, fedbn_aggregate
│
├── client/                 ← FL client (Module 7)
│   └── site_client.py      ← SiteClient (local train + DP + FedProx)
│
├── falcon/                 ← FALCON-specific utilities
│   └── data_partition.py   ← Dirichlet/pathological non-IID partitioning
│
└── tests/                  ← Full test suite (87 tests)
    ├── test_tensor_autograd.py   ← Module 1: 17 tests
    ├── test_ops.py               ← Module 2: 22 tests
    ├── test_nn_training.py       ← Module 3+4: 19 tests
    ├── test_privacy.py           ← Module 5: 11 tests
    └── test_models_fl.py         ← Module 6+7: 18 tests
```

---

## picograd Framework

A complete PyTorch-clone built from scratch using only NumPy.

### Core Design

```python
import picograd
import picograd.nn as nn
import picograd.optim as optim
from picograd import Tensor

# Autograd DAG
x = Tensor([3.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)  # tensor([6.0])

# Neural network
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
optimizer = optim.Adam(list(model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

loss = criterion(model(x), y)
loss.backward()
optimizer.step()

# Save/load
picograd.save(model.state_dict(), 'model.pkl')
model.load_state_dict(picograd.load('model.pkl'))
```

### Swappable Backend

```python
# Default: NumPy
picograd.set_backend(picograd.NumpyBackend())

# Future: drop-in replacement with Triton/CUDA
# picograd.set_backend(TritonBackend())
```

---

## Federated Learning Pipeline

```python
from picograd.models import AnomalyAE
from picograd.privacy import DPOptimizer, RDPAccountant, PrivacyConfig
from server.aggregator import Aggregator
from client.site_client import SiteClient
from falcon.data_partition import dirichlet_partition

# 1. Partition data non-IID across clients
parts = dirichlet_partition(labels, n_clients=5, alpha=0.5)

# 2. Create server aggregator
aggregator = Aggregator(strategy='fedavg', secure_agg=True)

# 3. Each client trains locally with DP
privacy = PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0)
client  = SiteClient('hospital_a', AnomalyAE(), train_loader, privacy)
client.set_global_weights(global_model.state_dict())
local_sd, metrics = client.train_round(epochs=1, lr=1e-3)

# 4. Aggregate
aggregator.receive_update('hospital_a', local_sd, n_samples=1000)
new_global_sd = aggregator.aggregate()
print(f"Round {aggregator.round} | ε = {aggregator.get_privacy_budget()[0]:.3f}")
```

---

## Research Gaps Addressed

| Gap | FALCON Solution |
|-----|-----------------|
| Scalability & Latency | Delta compression + buffered async aggregation |
| Segmentation-focused | CNN-Autoencoder reconstruction-error anomaly detection |
| High resource requirements | picograd NumPy backend — no GPU required |
| Lack of clinical interpretability | Grad-CAM heatmaps over anomalous regions |

---

## Test Results

```
Module 1 — Tensor + Autograd        17/17  ✓
Module 2 — Operations Library       22/22  ✓
Module 3+4 — nn.Module + Training   19/19  ✓
Module 5 — Privacy & Security       11/11  ✓
Module 6+7 — AnomalyAE + FL         18/18  ✓
─────────────────────────────────────────
TOTAL                                87/87  ✓ ALL PASS
```

---

## Dataset

Primary: **MedMNIST v2** (BreastMNIST, PneumoniaMNIST, RetinaMNIST)

```bash
pip install medmnist
python3 -c "import medmnist; medmnist.BreastMNIST(split='train', download=True)"
```

---

## Key Technical Decisions

1. **picograd as unified framework** — all DL computation through picograd, no torch dependency
2. **Backend abstraction** — NumpyBackend now; TritonBackend drops in with same interface
3. **im2col convolution** — maps conv to matmul, fast on NumPy (BLAS)
4. **Fused CrossEntropyLoss** — numerically stable (matches PyTorch behaviour)
5. **RDP accountant** — tighter privacy bounds than basic composition (O(σ²T) vs O(σ√T))
6. **Masking-based SecAgg** — server sees only aggregate, not individual updates
7. **Grad-CAM for autoencoders** — gradients of reconstruction error highlight anomalous regions
8. **Dirichlet non-IID** — realistic hospital data distribution simulation

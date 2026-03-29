"""
tests/test_dashboard.py
========================
Tests for Module 8: Dashboard backend.
Tests MetricsStore, FLDashboardBridge, and REST API (without Flask dependency).
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dashboard.backend.services.metrics_store import MetricsStore
from dashboard.backend.services.fl_bridge import FLDashboardBridge


# ── helpers ──────────────────────────────────────────────────────────────────

def assert_close(a, b, atol=1e-6, msg=""):
    diff = abs(float(a) - float(b))
    assert diff < atol, f"{msg} | diff={diff:.2e}"


# ── 1. MetricsStore ───────────────────────────────────────────────────────────

def test_store_server_status():
    store = MetricsStore()
    store.set_server_status("running", strategy="fedavg")
    assert store.server["status"] == "running"
    assert store.server["strategy"] == "fedavg"
    assert store.server["start_time"] is not None
    print("  ✓ MetricsStore: set_server_status works")


def test_store_client_registration():
    store = MetricsStore()
    store.register_client("hospital_a", n_samples=1000)
    clients = store.get_clients()
    assert len(clients) == 1
    assert clients[0]["client_id"] == "hospital_a"
    assert clients[0]["n_samples"] == 1000
    print("  ✓ MetricsStore: client registration and retrieval")


def test_store_client_online_timeout():
    store = MetricsStore()
    store.register_client("recent_client")
    store._store if False else None
    clients = store.get_clients()
    # Just registered → should be online
    assert clients[0]["online"] is True
    print("  ✓ MetricsStore: recently-seen client marked online")


def test_store_round_recording():
    store = MetricsStore()
    store.record_round(1, {"global_loss": 0.25, "n_clients": 3})
    store.record_round(2, {"global_loss": 0.20, "n_clients": 3})
    store.record_round(3, {"global_loss": 0.15, "n_clients": 3})
    rounds = store.get_rounds()
    assert len(rounds) == 3
    assert rounds[-1]["round"] == 3
    assert rounds[-1]["global_loss"] == 0.15
    # last_n
    last2 = store.get_rounds(last_n=2)
    assert len(last2) == 2
    assert last2[0]["round"] == 2
    print("  ✓ MetricsStore: round recording and retrieval (incl last_n)")


def test_store_weight_deltas():
    store = MetricsStore()
    for r in range(5):
        store.record_weight_delta(r, {
            "encoder.conv1.weight": float(np.random.rand()),
            "decoder.fc.weight":    float(np.random.rand()),
        })
    heatmap = store.get_weight_heatmap()
    assert "encoder.conv1.weight" in heatmap
    assert len(heatmap["encoder.conv1.weight"]) == 5
    print("  ✓ MetricsStore: weight delta heatmap data stored")


def test_store_privacy_timeline():
    store = MetricsStore()
    epsilons = [0.5, 1.0, 1.5, 2.0]
    for r, eps in enumerate(epsilons):
        store.record_privacy(r, eps)
    pdata = store.get_privacy()
    assert len(pdata["timeline"]) == 4
    assert_close(pdata["current_eps"], 2.0, msg="current_eps")
    print("  ✓ MetricsStore: privacy timeline recorded")


def test_store_gradcam():
    store = MetricsStore()
    store.update_gradcam("hospital_a", {
        "heatmap_b64":   "base64encodedstring",
        "anomaly_score": 0.042,
    })
    data = store.get_gradcam("hospital_a")
    assert data is not None
    assert data["anomaly_score"] == 0.042
    all_gcam = store.get_gradcam()
    assert "hospital_a" in all_gcam
    print("  ✓ MetricsStore: Grad-CAM data stored and retrieved")


def test_store_containers():
    store = MetricsStore()
    store.update_container("container_abc", name="falcon-client-1",
                           client_id="hospital_b", status="running",
                           image="falcon-client:latest")
    containers = store.get_containers()
    assert len(containers) == 1
    assert containers[0]["container_id"] == "container_abc"
    store.remove_container("container_abc")
    assert len(store.get_containers()) == 0
    print("  ✓ MetricsStore: container lifecycle (add/remove)")


def test_store_event_log():
    store = MetricsStore()
    store.set_server_status("running")
    store.register_client("c1")
    store.record_round(1, {"global_loss": 0.5})
    log = store.get_log()
    assert len(log) >= 3
    assert all("message" in e for e in log)
    print(f"  ✓ MetricsStore: event log ({len(log)} entries)")


def test_store_snapshot():
    store = MetricsStore()
    store.set_server_status("running", strategy="fedprox")
    store.register_client("c1")
    store.record_round(1, {"global_loss": 0.3})
    snap = store.snapshot()
    assert "server"    in snap
    assert "clients"   in snap
    assert "rounds"    in snap
    assert "privacy"   in snap
    assert snap["server"]["strategy"] == "fedprox"
    print("  ✓ MetricsStore: snapshot() returns complete state")


def test_store_thread_safety():
    """Concurrent reads/writes must not deadlock or corrupt state."""
    import threading
    store = MetricsStore()
    errors = []

    def writer(n):
        try:
            for i in range(20):
                store.record_round(n * 100 + i, {"global_loss": float(i) * 0.01})
                store.register_client(f"client_{n}_{i}")
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                _ = store.snapshot()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    print("  ✓ MetricsStore: thread-safe under concurrent read/write")


# ── 2. FLDashboardBridge ──────────────────────────────────────────────────────

def test_bridge_training_lifecycle():
    from picograd.models.anomaly_ae import AnomalyAE
    bridge = FLDashboardBridge(app=None)

    bridge.on_training_start({"strategy": "fedavg", "total_rounds": 10})
    assert bridge._store.server["status"] == "running"

    # Simulate 3 rounds
    model     = AnomalyAE(in_channels=1, latent_dim=8)
    prev_sd   = None
    for rnd in range(1, 4):
        metrics = [
            {"client_id": f"c{i}", "train_loss": 0.1 * rnd + 0.01*i}
            for i in range(3)
        ]
        bridge.on_round_complete(rnd, model, metrics, prev_sd)
        prev_sd = model.state_dict()

    rounds = bridge._store.get_rounds()
    assert len(rounds) == 3
    assert rounds[-1]["round"] == 3
    assert rounds[-1]["n_clients"] == 3

    bridge.on_training_done()
    assert bridge._store.server["status"] == "done"
    print("  ✓ FLDashboardBridge: full training lifecycle")


def test_bridge_privacy_updates():
    bridge = FLDashboardBridge(app=None)
    for rnd, eps in enumerate([0.5, 1.0, 1.8, 2.3], 1):
        bridge.on_privacy_update(rnd, eps)
    pdata = bridge._store.get_privacy()
    assert len(pdata["timeline"]) == 4
    assert_close(pdata["current_eps"], 2.3, msg="privacy eps")
    print("  ✓ FLDashboardBridge: privacy updates recorded")


def test_bridge_client_events():
    bridge = FLDashboardBridge(app=None)
    bridge.on_client_connect("hospital_x", n_samples=500)
    bridge.on_client_update("hospital_x", train_loss=0.042, round=3)
    clients = bridge._store.get_clients()
    c = next((c for c in clients if c["client_id"] == "hospital_x"), None)
    assert c is not None
    assert_close(c["train_loss"], 0.042)
    bridge.on_client_disconnect("hospital_x")
    clients2 = bridge._store.get_clients()
    c2 = next((c for c in clients2 if c["client_id"] == "hospital_x"), None)
    assert c2["status"] == "disconnected"
    print("  ✓ FLDashboardBridge: client connect/update/disconnect lifecycle")


def test_bridge_gradcam_encoding():
    bridge = FLDashboardBridge(app=None)
    heatmap  = np.random.rand(28, 28).astype(np.float32)
    original = np.random.rand(28, 28).astype(np.float32)
    bridge.on_gradcam("hospital_x", heatmap, anomaly_score=0.085, original=original)
    data = bridge._store.get_gradcam("hospital_x")
    assert data is not None
    assert "heatmap_b64"   in data
    assert "original_b64"  in data
    assert len(data["heatmap_b64"]) > 10
    print("  ✓ FLDashboardBridge: Grad-CAM PNG encoding to base64")


def test_bridge_weight_deltas():
    """Bridge computes per-layer weight deltas between rounds."""
    from picograd.models.anomaly_ae import AnomalyAE
    import picograd.optim as optim, picograd.nn as nn
    from picograd import Tensor

    bridge = FLDashboardBridge(app=None)
    model  = AnomalyAE(in_channels=1, latent_dim=8)

    # Round 1
    prev_sd = model.state_dict()
    opt     = optim.Adam(list(model.parameters()), lr=0.01)
    x       = Tensor(np.random.rand(2, 1, 28, 28).astype(np.float32))
    loss    = nn.MSELoss()(model(x), x)
    model.zero_grad(); loss.backward(); opt.step()

    bridge.on_round_complete(1, model, [{"client_id":"c0","train_loss":0.05}], prev_sd)
    heatmap = bridge._store.get_weight_heatmap()
    assert len(heatmap) > 0
    print(f"  ✓ FLDashboardBridge: {len(heatmap)} layer deltas tracked")


# ── 3. Integration: store + bridge + FL loop ──────────────────────────────────

def test_full_fl_dashboard_integration():
    """Simulate 5 FL rounds fully piped through the dashboard bridge."""
    import picograd, picograd.nn as nn, picograd.optim as optim
    from picograd import Tensor, manual_seed
    from picograd.models.anomaly_ae import AnomalyAE
    from server.fedavg import fedavg
    from picograd.privacy import RDPAccountant
    from dashboard.backend.services.metrics_store import MetricsStore

    manual_seed(99)
    # Use isolated store (not global singleton) for test isolation
    bridge     = FLDashboardBridge(app=None)
    bridge._store = MetricsStore()   # fresh isolated store
    accountant = RDPAccountant()
    bridge.on_training_start({"strategy": "fedavg", "total_rounds": 5})

    global_model = AnomalyAE(in_channels=1, latent_dim=8)
    X            = np.random.rand(16, 1, 28, 28).astype(np.float32)

    for rnd in range(1, 6):
        prev_sd        = global_model.state_dict()
        client_updates = []
        client_metrics = []

        for c_id in range(2):
            local = AnomalyAE(in_channels=1, latent_dim=8)
            local.load_state_dict(prev_sd)
            opt  = optim.Adam(list(local.parameters()), lr=1e-3)
            crit = nn.MSELoss()
            x    = Tensor(X[c_id*8:(c_id+1)*8])
            local.zero_grad()
            loss = crit(local(x), x)
            loss.backward(); opt.step()
            client_updates.append(local.state_dict())
            client_metrics.append({"client_id": f"c{c_id}", "train_loss": loss.item()})
            bridge.on_client_update(f"c{c_id}", train_loss=loss.item(), round=rnd)

        agg_sd = fedavg(client_updates, [8, 8])
        global_model.load_state_dict(agg_sd)
        bridge.on_round_complete(rnd, global_model, client_metrics, prev_sd)

        accountant.step(1.0, 0.01)
        eps = accountant.get_epsilon(1e-5)
        bridge.on_privacy_update(rnd, eps)

    bridge.on_training_done()
    snap = bridge._store.snapshot()

    assert snap["server"]["status"] == "done"
    assert len(snap["rounds"]) == 5
    assert len(snap["privacy"]["timeline"]) == 5
    assert len(snap["clients"]) == 2
    assert len(bridge._store.get_weight_heatmap()) > 0
    print(f"  ✓ Full FL→dashboard integration: 5 rounds, "
          f"ε={snap['privacy']['current_eps']:.3f}")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_store_server_status,
        test_store_client_registration,
        test_store_client_online_timeout,
        test_store_round_recording,
        test_store_weight_deltas,
        test_store_privacy_timeline,
        test_store_gradcam,
        test_store_containers,
        test_store_event_log,
        test_store_snapshot,
        test_store_thread_safety,
        test_bridge_training_lifecycle,
        test_bridge_privacy_updates,
        test_bridge_client_events,
        test_bridge_gradcam_encoding,
        test_bridge_weight_deltas,
        test_full_fl_dashboard_integration,
    ]
    print(f"\n{'='*60}\nMODULE 8: Dashboard Backend Tests\n{'='*60}")
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed  "
          f"{'✓ ALL PASS' if failed==0 else f'✗ {failed} FAILED'}")
    print('='*60)
    if failed: sys.exit(1)

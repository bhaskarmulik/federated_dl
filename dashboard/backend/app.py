"""
dashboard/backend/app.py
=========================
Flask + Flask-SocketIO dashboard backend.

REST endpoints:
  GET  /api/status           -- server + round status
  GET  /api/clients          -- connected client list
  GET  /api/rounds           -- training metrics history
  GET  /api/rounds/<n>       -- last N rounds
  GET  /api/weights          -- weight delta heatmap data
  GET  /api/privacy          -- privacy budget timeline
  GET  /api/gradcam/<id>     -- Grad-CAM image for client
  GET  /api/docker/containers-- Docker container list
  POST /api/docker/launch    -- Launch a new client container
  POST /api/docker/stop/<id> -- Stop a container
  GET  /api/config           -- Current FL config
  POST /api/config           -- Update FL config
  GET  /api/log              -- Recent event log

WebSocket events (server -> client):
  round_complete    -- new round metrics available
  client_update     -- client status changed
  weight_update     -- weight delta snapshot
  privacy_update    -- epsilon updated
  gradcam_update    -- new Grad-CAM image
  server_status     -- server state changed
  log_entry         -- new log message
"""

from __future__ import annotations
import json
import time
import threading
from typing import Any, Dict

try:
    from flask import Flask, jsonify, request, abort
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from flask_socketio import SocketIO, emit, join_room
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

from dashboard.backend.services.metrics_store import get_store


def create_app(config: Dict = None) -> Any:
    """Application factory. Returns (app, socketio) or stub if Flask unavailable."""

    if not HAS_FLASK:
        print("[dashboard] Flask not installed -- dashboard backend unavailable.")
        print("[dashboard] Install with: pip install flask flask-cors flask-socketio")
        return None, None

    app     = Flask(__name__)
    app.config["SECRET_KEY"] = "falcon-dashboard-secret"
    CORS(app, origins="*")

    socketio = SocketIO(app, cors_allowed_origins="*",
                        async_mode="threading") if HAS_SOCKETIO else None
    store    = get_store()

    # --- REST: Status --------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "server":   store.server,
            "uptime":   time.time() - (store.server["start_time"] or time.time()),
            "n_clients": len(store.clients),
        })

    @app.route("/api/snapshot")
    def api_snapshot():
        """Full initial state for page load."""
        return jsonify(store.snapshot())

    # --- REST: Clients -------------------------------------------------------

    @app.route("/api/clients")
    def api_clients():
        return jsonify({"clients": store.get_clients()})

    @app.route("/api/clients/<client_id>")
    def api_client_detail(client_id):
        clients = {c["client_id"]: c for c in store.get_clients()}
        if client_id not in clients:
            abort(404)
        return jsonify(clients[client_id])

    # --- REST: Rounds --------------------------------------------------------

    @app.route("/api/rounds")
    def api_rounds():
        n = request.args.get("n", default=50, type=int)
        return jsonify({"rounds": store.get_rounds(last_n=n)})

    @app.route("/api/rounds/latest")
    def api_latest_round():
        rounds = store.get_rounds(last_n=1)
        return jsonify(rounds[0] if rounds else {})

    # --- REST: Weights -------------------------------------------------------

    @app.route("/api/weights")
    def api_weights():
        return jsonify({"heatmap": store.get_weight_heatmap()})

    # --- REST: Privacy -------------------------------------------------------

    @app.route("/api/privacy")
    def api_privacy():
        return jsonify(store.get_privacy())

    # --- REST: GradCAM -------------------------------------------------------

    @app.route("/api/gradcam")
    def api_gradcam_all():
        return jsonify({"gradcam": store.get_gradcam()})

    @app.route("/api/gradcam/<client_id>")
    def api_gradcam_client(client_id):
        data = store.get_gradcam(client_id)
        if data is None:
            abort(404)
        return jsonify(data)

    # --- REST: Docker --------------------------------------------------------

    @app.route("/api/docker/containers")
    def api_containers():
        return jsonify({"containers": store.get_containers()})

    @app.route("/api/docker/launch", methods=["POST"])
    def api_launch_container():
        data = request.get_json() or {}
        client_id = data.get("client_id", f"client_{int(time.time())}")
        dataset   = data.get("dataset", "BreastMNIST")
        alpha     = data.get("alpha", 0.5)

        # Mock: in production this calls docker.from_env().containers.run(...)
        container_id = f"falcon-client-{client_id[:8]}"
        store.update_container(container_id,
                               name=f"falcon-{client_id}",
                               client_id=client_id,
                               status="running",
                               image="falcon-client:latest",
                               dataset=dataset,
                               alpha=alpha)
        store.register_client(client_id, container_id=container_id)

        if socketio:
            socketio.emit("client_update", store.get_clients())

        return jsonify({"container_id": container_id, "client_id": client_id})

    @app.route("/api/docker/stop/<container_id>", methods=["POST"])
    def api_stop_container(container_id):
        store.remove_container(container_id)
        if socketio:
            socketio.emit("client_update", store.get_clients())
        return jsonify({"stopped": container_id})

    # --- REST: Config --------------------------------------------------------

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        return jsonify({
            "strategy":     store.server.get("strategy", "fedavg"),
            "secure_agg":   store.server.get("secure_agg", False),
            "dp_enabled":   store.server.get("dp_enabled", False),
            "total_rounds": store.server.get("total_rounds", 10),
        })

    @app.route("/api/config", methods=["POST"])
    def api_set_config():
        data = request.get_json() or {}
        store.set_config(
            strategy    = data.get("strategy", "fedavg"),
            secure_agg  = data.get("secure_agg", False),
            dp_enabled  = data.get("dp_enabled", False),
            total_rounds= data.get("total_rounds", 10),
        )
        return jsonify({"ok": True})

    # --- REST: Log -----------------------------------------------------------

    @app.route("/api/log")
    def api_log():
        n = request.args.get("n", default=50, type=int)
        return jsonify({"log": store.get_log(last_n=n)})

    # --- WebSocket events ----------------------------------------------------

    if socketio:
        @socketio.on("connect")
        def ws_connect():
            emit("snapshot", store.snapshot())

        @socketio.on("subscribe")
        def ws_subscribe(data):
            room = data.get("room", "default")
            join_room(room)
            emit("subscribed", {"room": room})

    # --- Public broadcast helpers (called by FL server) ----------------------

    def broadcast_round(round_num: int, metrics: Dict) -> None:
        store.record_round(round_num, metrics)
        if socketio:
            socketio.emit("round_complete", {
                "round": round_num, **metrics
            })

    def broadcast_weight_delta(round_num: int, deltas: Dict) -> None:
        store.record_weight_delta(round_num, deltas)
        if socketio:
            socketio.emit("weight_update", {
                "round":  round_num,
                "deltas": deltas,
            })

    def broadcast_privacy(round_num: int, epsilon: float) -> None:
        store.record_privacy(round_num, epsilon)
        if socketio:
            socketio.emit("privacy_update", {
                "round":   round_num,
                "epsilon": epsilon,
            })

    def broadcast_gradcam(client_id: str, data: Dict) -> None:
        store.update_gradcam(client_id, data)
        if socketio:
            socketio.emit("gradcam_update", {
                "client_id": client_id, **data
            })

    # Attach broadcast helpers to app for use by FL integration
    app.broadcast_round        = broadcast_round
    app.broadcast_weight_delta = broadcast_weight_delta
    app.broadcast_privacy      = broadcast_privacy
    app.broadcast_gradcam      = broadcast_gradcam
    app.store                  = store

    return app, socketio


def run_dashboard(host: str = "0.0.0.0", port: int = 5000,
                  debug: bool = False) -> None:
    """Start the dashboard server."""
    app, socketio = create_app()
    if app is None:
        return
    if socketio:
        socketio.run(app, host=host, port=port, debug=debug)
    else:
        app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)

"""
Flask-SocketIO dashboard app.
Run this in a background thread while training runs in the main thread.

Usage:
    from dashboard.app import start_dashboard
    start_dashboard()   # non-blocking, starts in a daemon thread
"""

import threading
from flask import Flask, render_template
from flask_socketio import SocketIO

from .data_collector import get_data_collector

app    = Flask(__name__)
io     = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Map env_name → panel template
_PANEL_TEMPLATES = {
    "CartPole-v1":    "panels/cartpole.html",
    "Pendulum-v1":    "panels/pendulum.html",
    "LunarLander-v3": "panels/lunarlander.html",
}


@app.route("/")
def index():
    dc       = get_data_collector()
    env_name = dc._env_name or "—"
    template = _PANEL_TEMPLATES.get(env_name, "panels/cartpole.html")
    return render_template(template, env_name=env_name)


@io.on("connect")
def on_connect():
    """Push recent history to a newly connected browser."""
    dc = get_data_collector()
    for item in dc.get_latest("episode", 300):
        io.emit("episode", item)
    for item in dc.get_latest("agent", 300):
        io.emit("agent", item)


def _emit_loop():
    """Background thread: drain queues and emit to all clients."""
    dc = get_data_collector()
    while True:
        # env steps
        while not dc.env_queue.empty():
            try:
                io.emit("env", dc.env_queue.get_nowait())
            except Exception:
                break
        # agent snapshots
        while not dc.agent_queue.empty():
            try:
                io.emit("agent", dc.agent_queue.get_nowait())
            except Exception:
                break
        # episode metrics
        while not dc.metrics_queue.empty():
            try:
                io.emit("episode", dc.metrics_queue.get_nowait())
            except Exception:
                break

        io.sleep(0.05)   # 20 Hz


def start_dashboard(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Start dashboard in a daemon thread — non-blocking."""
    io.start_background_task(_emit_loop)
    t = threading.Thread(
        target=lambda: io.run(app, host=host, port=port, log_output=False),
        daemon=True,
    )
    t.start()
    print(f"Dashboard → http://{host}:{port}")

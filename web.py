"""Quantyze browser UI.

Module Description
==================
This module contains the Flask browser UI and API wiring for Quantyze. It
serves the browser frontend, exposes simulation and training endpoints, manages
background web jobs, and returns artifact, chart, and order-book state to the
frontend.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, TextIO, cast

from flask import Flask, jsonify, request, send_from_directory

from api_payloads import (
    book_depth_payload,
    book_summary_payload,
    clamp_int,
    execution_log_payload,
    metrics_payload,
    open_orders_payload,
    trades_payload,
)
from config import (
    ACTIVE_MODEL_MODES,
    DATASET_PACKAGE_PATH,
    DEFAULT_TRADE_LIMIT,
    DEFAULT_DEPTH_LEVELS,
    LOG_PATH,
    LATEST_TRAINING_METRICS_PATH,
    MAX_DEPTH_LEVELS,
    MAX_TRADE_LIMIT,
    SAMPLE_DATASET_PATH,
    SCENARIO_CHOICES,
    TRAINING_METRICS_PATH,
    artifact_paths,
    format_optional_path,
    overlay_mode_text,
    resolve_active_model_status,
    set_active_model_selection,
    training_preset_map,
    ui_config_payload,
)
from simulation import (
    build_system,
    make_run_args,
    print_summary,
    simulation_source_label,
)
from training import train_model


class _SimulationState:
    """Mutable browser-side simulation state for one app instance.

    Instance Attributes:
    - state: the current job state label
    - progress: the current job progress percentage
    - log: the captured stdout lines for the job
    - results: the final simulation results payload, or None while unavailable
    - error: the error message for a failed job, or None if there is no error
    - runtime: the live in-memory runtime objects for the latest simulation

    Representation Invariants:
    - self.state in {'idle', 'running', 'done', 'error'}
    """

    state: str
    progress: float
    log: list[str]
    results: dict[str, Any] | None
    error: str | None
    runtime: dict[str, Any]

    def __init__(self) -> None:
        """Initialize an empty simulation state snapshot."""
        self.state = "idle"
        self.progress = 0.0
        self.log = []
        self.results = None
        self.error = None
        self.runtime = {
            "book": None,
            "engine": None,
            "agent": None,
            "stream": None,
        }


class _TrainingState:
    """Mutable browser-side training state for one app instance.

    Instance Attributes:
    - state: the current job state label
    - progress: the current job progress percentage
    - log: the captured stdout lines for the job
    - results: the final training results payload, or None while unavailable
    - error: the error message for a failed job, or None if there is no error

    Representation Invariants:
    - self.state in {'idle', 'running', 'done', 'error'}
    """

    state: str
    progress: float
    log: list[str]
    results: dict[str, Any] | None
    error: str | None

    def __init__(self) -> None:
        """Initialize an empty training state snapshot."""
        self.state = "idle"
        self.progress = 0.0
        self.log = []
        self.results = None
        self.error = None


class _TeeStream:
    """Mirror stdout while capturing complete log lines for the web UI.

    Instance Attributes:
    - _lines: the captured complete log lines
    - _real: the underlying real stdout stream
    - _buf: the partial current line buffer

    Representation Invariants:
    - len(self._buf.split('\\n')) >= 1
    """

    _lines: list[str]
    _real: TextIO
    _buf: str

    def __init__(self, lines: list[str]) -> None:
        real_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
        self._lines = lines
        self._real = cast(TextIO, real_stdout)
        self._buf = ""

    def write(self, text: str) -> int:
        """Write ``text`` to stdout and append completed lines to ``self._lines``."""
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._lines.append(line)
        return len(text)

    def flush(self) -> None:
        """Flush the underlying real stdout."""
        self._real.flush()


def _run_web_simulation(
    params: dict[str, Any],
    web_lock: threading.Lock,
    web_sim: _SimulationState,
) -> None:
    """Run one browser-triggered simulation in a background thread."""
    log_lines: list[str] = []

    with web_lock:
        web_sim.state = "running"
        web_sim.progress = 0.0
        web_sim.log = log_lines
        web_sim.results = None
        web_sim.error = None
        web_sim.runtime = {
            "book": None,
            "engine": None,
            "agent": None,
            "stream": None,
        }

    tee = _TeeStream(log_lines)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        requested_mode = params.get("model")
        if requested_mode in ACTIVE_MODEL_MODES:
            set_active_model_selection(requested_mode)

        active_model_status = resolve_active_model_status()
        raw_model_path = active_model_status.get("model_path")
        model_path = (
            str(raw_model_path)
            if isinstance(raw_model_path, str) and active_model_status.get("checkpoint_exists")
            else None
        )

        args = make_run_args(
            data=params.get("data") or None,
            scenario=params.get("scenario") or None,
            speed=float(params.get("speed", 0.0)),
        )

        book, engine, stream, agent, loader = build_system(args, model_path)

        print("Quantyze Run Configuration")
        print("=" * 30)
        print(f"Event Source: {simulation_source_label(args)}")
        print(f"Simulation Overlay Mode: {overlay_mode_text(active_model_status['mode'])}")
        print(f"Simulation Overlay Path: {format_optional_path(active_model_status['model_path'])}")
        print(f"Overlay Provenance: {active_model_status['dataset_label']}")
        print("Overlay Role: optional classifier inference")
        if active_model_status["note"]:
            print(f"Note: {active_model_status['note']}")
        print("=" * 30)

        if agent is None:
            print("Running without a model overlay.")
        else:
            print(f"Loaded simulation overlay checkpoint from {active_model_status['model_path']}.")

        if args.data is not None:
            print(f"Replay dataset path: {args.data}")
            print(f"Detected source format: {loader.source_format}")
            if loader.source_format == "lobster":
                print(
                    "Recognized "
                    f"{len(loader.special_events)} non-replayable LOBSTER annotations."
                )
        elif args.scenario is not None:
            print(f"Running synthetic scenario: {args.scenario}")
        else:
            print("Running default synthetic scenario: balanced")

        total_events = len(stream.source)
        stream.running = True
        with web_lock:
            web_sim.runtime["stream"] = stream

        for index, event in enumerate(stream.source):
            if not stream.running:
                break
            fills = stream.emit(event)
            if agent and fills:
                agent.step(book, fills[-1]["exec_price"])
            with web_lock:
                web_sim.progress = (index + 1) / max(total_events, 1) * 100

        stream.running = False
        print_summary(engine, agent)
        if agent is None:
            print("Simulation used no model overlay.")
        else:
            print(f"Simulation used the {active_model_status['mode']} overlay.")

        book.flush_log(LOG_PATH)
        print(f"Execution log written to {LOG_PATH}.")

        metrics = engine.compute_metrics()
        results: dict[str, Any] = {
            **metrics,
            "spread": book.spread(),
            "mid_price": book.mid_price(),
            "agent_pnl": agent.current_pnl() if agent else None,
            "total_events": total_events,
            "source_label": simulation_source_label(args),
            "overlay_mode": active_model_status["mode"],
        }

        with web_lock:
            web_sim.state = "done"
            web_sim.progress = 100.0
            web_sim.results = results
            web_sim.runtime["book"] = book
            web_sim.runtime["engine"] = engine
            web_sim.runtime["agent"] = agent
            web_sim.runtime["stream"] = None

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        tb = traceback.format_exc()
        print(f"ERROR: {exc}")
        with web_lock:
            web_sim.state = "error"
            web_sim.error = str(exc)
            web_sim.runtime["stream"] = None
            web_sim.log.extend(tb.splitlines())
    finally:
        sys.stdout = old_stdout


def _run_web_training(
    params: dict[str, Any],
    web_lock: threading.Lock,
    web_train: _TrainingState,
) -> None:
    """Run one browser-triggered training job in a background thread."""
    log_lines: list[str] = []

    with web_lock:
        web_train.state = "running"
        web_train.progress = 0.0
        web_train.log = log_lines
        web_train.results = None
        web_train.error = None

    tee = _TeeStream(log_lines)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        data_path = str(params["data_path"])
        print(f"Starting training on: {data_path}")
        print("Loading dataset and building features...")
        print("(Training runs 50 epochs; this may take a minute.)")
        results = train_model(data_path)

        with web_lock:
            web_train.state = "done"
            web_train.progress = 100.0
            web_train.results = results
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        tb = traceback.format_exc()
        print(f"ERROR: {exc}")
        with web_lock:
            web_train.state = "error"
            web_train.error = str(exc)
            web_train.log.extend(tb.splitlines())
    finally:
        sys.stdout = old_stdout


def create_web_app() -> Flask:
    """Build the browser-based Quantyze web application."""
    app = Flask(__name__, static_folder=".", static_url_path="")
    web_lock = threading.Lock()
    web_sim = _SimulationState()
    web_train = _TrainingState()

    def _sim_components() -> tuple[Any, Any, Any]:
        with web_lock:
            return (
                web_sim.runtime["book"],
                web_sim.runtime["engine"],
                web_sim.runtime["agent"],
            )

    @app.get("/")
    def index() -> Any:
        return send_from_directory(".", "index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "quantyze-web"})

    @app.get("/api/config")
    def app_config() -> Any:
        return jsonify(ui_config_payload())

    @app.get("/api/model-status")
    def get_model_status() -> Any:
        status = resolve_active_model_status()
        return jsonify({
            "mode": status["mode"],
            "model_path": status.get("model_path"),
            "metrics_path": status.get("metrics_path"),
            "dataset_label": status.get("dataset_label"),
            "checkpoint_exists": bool(status.get("checkpoint_exists")),
            "note": status.get("note", ""),
        })

    @app.post("/api/model-status")
    def set_model_status() -> Any:
        body = request.get_json(force=True, silent=True) or {}
        mode = body.get("mode", "")
        if mode not in ACTIVE_MODEL_MODES:
            return jsonify({"error": f"mode must be one of {list(ACTIVE_MODEL_MODES)}"}), 400
        status = set_active_model_selection(mode)
        return jsonify({
            "mode": status["mode"],
            "checkpoint_exists": bool(status.get("checkpoint_exists")),
        })

    @app.post("/api/simulate")
    def start_simulation() -> Any:
        with web_lock:
            if web_sim.state == "running":
                return jsonify({"error": "Simulation already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "synthetic")

        if source == "synthetic":
            scenario = body.get("scenario", "balanced")
            if scenario not in SCENARIO_CHOICES:
                return jsonify({"error": f"Unknown scenario: {scenario}"}), 400
            params: dict[str, Any] = {"scenario": scenario}
        else:
            data = str(body.get("data") or "").strip()
            if not data:
                return jsonify({"error": "data path is required for CSV source"}), 400
            if not Path(data).exists():
                return jsonify({"error": f"File not found: {data}"}), 400
            params = {"data": data}

        params["speed"] = float(body.get("speed", 0.0))
        if body.get("model") in ACTIVE_MODEL_MODES:
            params["model"] = body["model"]

        threading.Thread(
            target=_run_web_simulation,
            args=(params, web_lock, web_sim),
            daemon=True,
        ).start()
        return jsonify({"started": True}), 202

    @app.get("/api/simulate/status")
    def simulation_status() -> Any:
        with web_lock:
            snapshot = {
                "state": web_sim.state,
                "progress": web_sim.progress,
                "log": list(web_sim.log),
                "results": web_sim.results,
                "error": web_sim.error,
            }
        return jsonify(snapshot)

    @app.post("/api/train")
    def start_training() -> Any:
        with web_lock:
            if web_train.state == "running":
                return jsonify({"error": "Training already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "sample")
        if source == "custom":
            data_path = str(body.get("data_path") or "").strip()
            if not data_path:
                return jsonify({"error": "data_path is required for custom source"}), 400
            if not Path(data_path).exists():
                return jsonify({"error": f"File not found: {data_path}"}), 400
        else:
            data_path = training_preset_map().get(source, SAMPLE_DATASET_PATH)
            if not Path(data_path).exists():
                return jsonify({
                    "error": (
                        f"Dataset not found: {data_path}. "
                        f"Extract {DATASET_PACKAGE_PATH} beside main.py first."
                    )
                }), 400

        threading.Thread(
            target=_run_web_training,
            args=({"data_path": data_path}, web_lock, web_train),
            daemon=True,
        ).start()
        return jsonify({"started": True}), 202

    @app.get("/api/train/status")
    def training_status() -> Any:
        with web_lock:
            snapshot = {
                "state": web_train.state,
                "progress": web_train.progress,
                "log": list(web_train.log),
                "results": web_train.results,
                "error": web_train.error,
            }
        return jsonify(snapshot)

    @app.get("/api/book/summary")
    def web_book_summary() -> Any:
        book = _sim_components()[0]
        agent = _sim_components()[2]
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(book_summary_payload(book, agent))

    @app.get("/api/book/depth")
    def web_book_depth() -> Any:
        book = _sim_components()[0]
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        levels = clamp_int(request.args.get("levels"), DEFAULT_DEPTH_LEVELS, 1, MAX_DEPTH_LEVELS)
        return jsonify(book_depth_payload(book, levels))

    @app.get("/api/metrics")
    def web_metrics() -> Any:
        engine = _sim_components()[1]
        if engine is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(metrics_payload(engine))

    @app.get("/api/trades")
    def web_trades() -> Any:
        book = _sim_components()[0]
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        limit = clamp_int(request.args.get("limit"), DEFAULT_TRADE_LIMIT, 1, MAX_TRADE_LIMIT)
        offset = clamp_int(request.args.get("offset"), 0, 0, MAX_TRADE_LIMIT)
        return jsonify(trades_payload(book, limit, offset, LOG_PATH))

    @app.get("/api/execution-log")
    def web_execution_log() -> Any:
        engine = _sim_components()[1]
        if engine is None:
            return jsonify({"error": "No simulation results yet"}), 404
        limit = clamp_int(request.args.get("limit"), DEFAULT_TRADE_LIMIT, 1, MAX_TRADE_LIMIT)
        offset = clamp_int(request.args.get("offset"), 0, 0, MAX_TRADE_LIMIT)
        return jsonify(execution_log_payload(engine, limit, offset))

    @app.get("/api/orders/open")
    def web_open_orders() -> Any:
        book = _sim_components()[0]
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(open_orders_payload(book))

    def _artifact_info(path: str) -> dict[str, Any]:
        artifact_path = Path(path)
        return {
            "path": path,
            "exists": artifact_path.exists(),
            "size": artifact_path.stat().st_size if artifact_path.exists() else None,
        }

    @app.get("/api/artifacts")
    def artifacts() -> Any:
        status = resolve_active_model_status()
        return jsonify({
            "active_model": {
                "mode": status["mode"],
                "dataset_label": status.get("dataset_label"),
                "checkpoint_exists": bool(status.get("checkpoint_exists")),
                "note": status.get("note", ""),
            },
            "files": {
                name: _artifact_info(path)
                for name, path in artifact_paths()
            },
        })

    @app.get("/api/metrics/baseline")
    def baseline_metrics() -> Any:
        path = Path(TRAINING_METRICS_PATH)
        if not path.exists():
            return jsonify({"error": "Baseline metrics not found. Extract the dataset package first."}), 404
        try:
            return jsonify(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/metrics/latest")
    def latest_metrics() -> Any:
        path = Path(LATEST_TRAINING_METRICS_PATH)
        if not path.exists():
            return jsonify({"error": "No latest metrics yet. Run training first."}), 404
        try:
            return jsonify(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/log-summary")
    def log_summary() -> Any:
        path = Path(LOG_PATH)
        if not path.exists():
            return jsonify({"count": 0, "first": None, "last": None})
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                return jsonify({"count": 0, "first": None, "last": None})
            return jsonify({
                "count": len(records),
                "first": records[0] if records else None,
                "last": records[-1] if records else None,
            })
        except (OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 500

    return app


def run_browser_ui(port: int = 9000) -> None:
    """Start the browser UI on the local development server."""
    web_app = create_web_app()
    print(f"Quantyze web UI -> http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    web_app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'json', 'sys', 'threading', 'traceback', 'pathlib', 'typing',
            'flask', 'api_payloads', 'config', 'simulation', 'training',
            'doctest', 'python_ta'
        ],
        'allowed-io': [
            '_TeeStream.write',
            '_run_web_simulation',
            '_run_web_training',
            'run_browser_ui'
        ],
        'disable': ['R0912', 'R0914', 'R0915'],
        'max-line-length': 120
    })

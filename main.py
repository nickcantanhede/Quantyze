"""Quantyze main entry point.

Module Description
==================
This module contains the top-level program entry point for Quantyze. It parses
command-line arguments, constructs the core system objects, resolves the active
classifier overlay, chooses between synthetic scenarios and dataset replay,
optionally trains a new classifier checkpoint, and prints simulation summaries.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request, send_from_directory
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from cli_menu import (
    MenuCallbacks,
    MenuConfig,
    MenuDatasets,
    MenuPaths,
    interactive_menu as run_interactive_menu,
)
from data_loader import DataLoader
from event_stream import EventStream
from matching_engine import MatchingEngine
from neural_net import Agent, Trainer, OrderBookNet, load_agent
from order_book import OrderBook
from price_level import PriceLevel

CLASS_NAMES = ["buy", "sell", "hold"]
MODEL_PATH = "model.pt"
TRAINING_METRICS_PATH = "training_metrics.json"
LATEST_MODEL_PATH = "latest_model.pt"
LATEST_TRAINING_METRICS_PATH = "latest_training_metrics.json"
LATEST_TRAINING_DATA_PATH = "latest_training_data.csv"
LOG_PATH = "log.json"
ACTIVE_MODEL_STATE_PATH = "active_model.json"
DATASET_PACKAGE_PATH = "quantyze_datasets.zip"
SAMPLE_DATASET_PATH = "sample_internal.csv"
HUGE_DATASET_PATH = "huge_internal.csv"
LOBSTER_SAMPLE_MESSAGE_PATH = "aapl_lobster_2012-06-21_message_5level_sample.csv"
SCENARIO_CHOICES = ("balanced", "low_liquidity", "high_volatility")
ACTIVE_MODEL_MODES = ("baseline", "latest", "none")
BASELINE_MODEL_LABEL = "packaged baseline checkpoint"
_DEFAULT_DEPTH_LEVELS = 10
_MAX_DEPTH_LEVELS = 500
_DEFAULT_TRADE_LIMIT = 200
_MAX_TRADE_LIMIT = 10_000

_web_lock = threading.Lock()
_web_sim: dict[str, Any] = {
    "state": "idle",
    "progress": 0.0,
    "log": [],
    "results": None,
    "error": None,
    "book": None,
    "engine": None,
    "agent": None,
    "stream": None,
}
_web_train: dict[str, Any] = {
    "state": "idle",
    "progress": 0.0,
    "log": [],
    "results": None,
    "error": None,
}


@dataclass
class _EvalStats:
    """Accumulate classifier evaluation counts over validation batches."""

    total: int
    correct: int
    true_counts: list[int]
    pred_counts: list[int]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class _PreparedTrainingData:
    """Normalized training tensors and index splits for one training run."""

    feature_dim: int
    normalized_features: torch.Tensor
    label_tensor: torch.Tensor
    train_indices: torch.Tensor
    val_indices: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def train_features(self) -> torch.Tensor:
        """Return the normalized training feature tensor."""
        return self.normalized_features[self.train_indices]

    def train_labels(self) -> torch.Tensor:
        """Return the training labels."""
        return self.label_tensor[self.train_indices]

    def val_features(self) -> torch.Tensor:
        """Return the normalized validation feature tensor."""
        return self.normalized_features[self.val_indices]

    def val_labels(self) -> torch.Tensor:
        """Return the validation labels."""
        return self.label_tensor[self.val_indices]


@dataclass(frozen=True)
class _TrainingOutputPaths:
    """Filesystem destinations produced by one training run."""

    model_path: str
    metrics_path: str
    training_data_path: str


def parse_args() -> argparse.Namespace:
    """Return the parsed command-line arguments for running Quantyze.

    The supported arguments determine whether Quantyze loads event data from a
    dataset file or generates a synthetic scenario, whether the optional ML
    path is enabled, and how the simulation should be configured.
    """

    parser = argparse.ArgumentParser(
        prog="Quantyze", description="Run the Quantyze limit order book simulator"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a CSV dataset file containing market events"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIO_CHOICES),
        default=None,
        help="Synthetic scenario to generate if no dataset is provided."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Replay speed for the event stream (0.0 = as fast as possible)"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the optional ML training flow instead of the simulation"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port used for the browser UI or post-simulation API server"
    )

    runtime_group = parser.add_mutually_exclusive_group()
    runtime_group.add_argument(
        "--ui",
        "--web",
        dest="ui",
        action="store_true",
        help="Start the browser-based web UI at http://127.0.0.1:<port>"
    )
    runtime_group.add_argument(
        "--cli",
        action="store_true",
        help="Open the interactive terminal menu"
    )
    runtime_group.add_argument(
        "--no-ui",
        action="store_true",
        help="Run directly without starting the browser UI"
    )

    args = parser.parse_args()
    direct_requested = (
        args.train
        or args.data is not None
        or args.scenario is not None
        or args.speed != 0.0
    )

    if args.train and not args.no_ui:
        parser.error("Training mode requires --no-ui.")

    if args.train and args.data is None:
        parser.error("Training mode requires --data <csv_path>.")

    if args.train and args.scenario is not None:
        parser.error("--scenario cannot be combined with --train.")

    if args.cli and direct_requested:
        parser.error("Direct simulation/training arguments cannot be combined with --cli.")

    if args.ui and direct_requested:
        parser.error("Direct simulation/training arguments require --no-ui, not --ui.")

    if not args.ui and not args.cli and not args.no_ui and direct_requested:
        parser.error("Direct simulation/training arguments require --no-ui.")

    return args


def _checkpoint_exists(path: str | None) -> bool:
    """Return whether ``path`` points to a non-empty checkpoint file."""
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def _format_decimal(value: float) -> str:
    """Return ``value`` formatted to two decimal places."""
    return f"{value:.2f}"


def _format_metric(value: object) -> str:
    """Return a short printable representation for numeric metrics."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_decimal(value)
    return str(value)


def _format_optional_path(path: object) -> str:
    """Return an absolute path for strings, or ``None`` when not available."""
    if not isinstance(path, str) or path == "":
        return "None"
    return os.path.abspath(path)


def _overlay_mode_text(mode: object) -> str:
    """Return a human-readable overlay mode label."""
    labels = {"baseline": "Baseline", "latest": "Latest", "none": "None"}
    if isinstance(mode, str):
        return labels.get(mode, mode.title())
    return str(mode)


def _default_active_model_mode() -> str:
    """Return the shipped default overlay mode for this environment."""
    return "baseline" if _checkpoint_exists(MODEL_PATH) else "none"


def _default_no_model_note() -> str:
    """Return the clearest note for runs without an available default checkpoint."""
    if os.path.exists(DATASET_PACKAGE_PATH):
        return (
            "No packaged checkpoint is available yet. Extract "
            f"{DATASET_PACKAGE_PATH} beside main.py to enable the baseline overlay."
        )
    return "No packaged checkpoint is available; running without a model."


def _dataset_label_for_path(dataset_path: str | None) -> str:
    """Return a short human-readable label for a dataset path."""
    if not dataset_path:
        return "unknown"

    dataset_name = os.path.basename(dataset_path)
    if dataset_name in {SAMPLE_DATASET_PATH, HUGE_DATASET_PATH, LOBSTER_SAMPLE_MESSAGE_PATH}:
        return dataset_name

    return f"custom ({dataset_name})"


def _dataset_label_from_metrics(metrics_path: str, fallback: str) -> str:
    """Return a dataset label inferred from ``metrics_path`` if possible."""
    try:
        with open(metrics_path, encoding="utf-8") as file:
            metrics = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return fallback

    dataset_path = metrics.get("dataset_path")
    if not isinstance(dataset_path, str):
        return fallback

    return _dataset_label_for_path(dataset_path)


def _active_model_payload(
    mode: str,
    dataset_label: str | None = None,
    user_selected: bool = False,
) -> dict[str, object]:
    """Return the persisted payload for the selected active-model mode."""
    if mode == "baseline":
        return {
            "mode": "baseline",
            "model_path": MODEL_PATH,
            "metrics_path": TRAINING_METRICS_PATH,
            "dataset_label": dataset_label or BASELINE_MODEL_LABEL,
            "user_selected": user_selected,
        }
    if mode == "latest":
        latest_label = dataset_label or _dataset_label_from_metrics(
            LATEST_TRAINING_METRICS_PATH,
            "latest training",
        )
        return {
            "mode": "latest",
            "model_path": LATEST_MODEL_PATH,
            "metrics_path": LATEST_TRAINING_METRICS_PATH,
            "dataset_label": latest_label,
            "user_selected": user_selected,
        }
    if mode == "none":
        return {
            "mode": "none",
            "model_path": None,
            "metrics_path": None,
            "dataset_label": dataset_label or "No active model",
            "user_selected": user_selected,
        }

    raise ValueError(f"Unsupported active model mode: {mode}")


def _load_active_model_payload() -> dict[str, object] | None:
    """Return the saved active-model payload, or None if unavailable/invalid."""
    try:
        with open(ACTIVE_MODEL_STATE_PATH, encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    mode = payload.get("mode")
    if mode not in ACTIVE_MODEL_MODES:
        return None

    user_selected = payload.get("user_selected")
    if user_selected is not None and not isinstance(user_selected, bool):
        return None

    return payload


def _saved_dataset_label(payload: dict[str, object] | None) -> str | None:
    """Return the saved dataset label from ``payload`` when it is a string."""
    if payload is None:
        return None

    dataset_label = payload.get("dataset_label")
    if isinstance(dataset_label, str):
        return dataset_label
    return None


def save_active_model_selection(
    mode: str,
    dataset_label: str | None = None,
    user_selected: bool = True,
) -> dict[str, object]:
    """Persist and return the selected active-model payload."""
    payload = _active_model_payload(mode, dataset_label, user_selected=user_selected)
    with open(ACTIVE_MODEL_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def resolve_active_model_status() -> dict[str, object]:
    """Return the resolved active-model status, applying safe fallbacks."""
    raw_payload = _load_active_model_payload()
    note = ""
    default_mode = _default_active_model_mode()
    explicit_selection = False

    if raw_payload is None:
        requested_mode = default_mode
        if default_mode == "baseline":
            note = "Active model state was missing or invalid; using the packaged baseline model."
        else:
            note = _default_no_model_note()
    else:
        explicit_selection = bool(raw_payload.get("user_selected"))
        if explicit_selection:
            requested_mode = str(raw_payload["mode"])
        else:
            requested_mode = default_mode
            if str(raw_payload["mode"]) != default_mode:
                if default_mode == "baseline":
                    note = "No explicit overlay selection was found; using the packaged baseline model."
                else:
                    note = _default_no_model_note()

    if requested_mode == "latest":
        if _checkpoint_exists(LATEST_MODEL_PATH):
            resolved_payload = _active_model_payload(
                "latest",
                _saved_dataset_label(raw_payload),
                user_selected=explicit_selection,
            )
        elif _checkpoint_exists(MODEL_PATH):
            resolved_payload = _active_model_payload("baseline", user_selected=explicit_selection)
            note = "Latest checkpoint was unavailable; fell back to the baseline model."
        else:
            resolved_payload = _active_model_payload("none", user_selected=explicit_selection)
            note = "Latest checkpoint was unavailable; running without a model."
    elif requested_mode == "baseline":
        if _checkpoint_exists(MODEL_PATH):
            resolved_payload = _active_model_payload("baseline", user_selected=explicit_selection)
        else:
            resolved_payload = _active_model_payload("none", user_selected=explicit_selection)
            note = "Baseline checkpoint was unavailable; running without a model."
    else:
        resolved_payload = _active_model_payload("none", user_selected=explicit_selection)

    should_persist_resolved_state = (
        raw_payload is None
        or explicit_selection
        or (raw_payload is not None and "user_selected" not in raw_payload)
    )
    if raw_payload != resolved_payload and should_persist_resolved_state:
        save_active_model_selection(
            str(resolved_payload["mode"]),
            _saved_dataset_label(resolved_payload),
            user_selected=bool(resolved_payload.get("user_selected")),
        )

    model_path = resolved_payload["model_path"]
    return {
        **resolved_payload,
        "requested_mode": requested_mode,
        "checkpoint_exists": _checkpoint_exists(model_path if isinstance(model_path, str) else None),
        "state_path": ACTIVE_MODEL_STATE_PATH,
        "note": note,
    }


def set_active_model_selection(mode: str) -> dict[str, object]:
    """Persist a requested mode and return the resolved active-model status."""
    save_active_model_selection(mode, user_selected=True)
    return resolve_active_model_status()


def _simulation_source_label(args: argparse.Namespace) -> str:
    """Return a concise label for the selected simulation source."""
    if args.data is not None:
        return args.data
    if args.scenario is not None:
        return f"synthetic {args.scenario}"
    return "synthetic balanced"


def build_system(
    args: argparse.Namespace,
    model_path: str | None = None,
) -> tuple[OrderBook, MatchingEngine, EventStream, Agent | None, DataLoader]:
    """Construct and return the core Quantyze system objects.

    This function creates the order book, matching engine, event stream, and
    optional agent needed to run the program from the parsed command-line
    arguments. It also returns the loader used to provide event data so callers
    can report detected source-format metadata.
    """

    book = OrderBook()
    engine = MatchingEngine(book)
    loader = DataLoader()

    if args.data is not None:
        loader.filepath = args.data
        events = loader.load_csv()
    elif args.scenario is not None:
        events = loader.generate_synthetic(args.scenario)
    else:
        events = loader.generate_synthetic("balanced")  # default scenario

    stream = EventStream(events, engine, args.speed)

    if args.train or model_path is None:
        agent = None
    else:
        agent = load_agent(model_path)

    return book, engine, stream, agent, loader


def run_simulation(stream: EventStream, agent: Agent | None, book: OrderBook) -> None:
    """Run the main Quantyze simulation flow.

    This function processes the event stream through the matching engine and
    allows any optional agent logic to observe or react to the evolving order
    book state.
    """

    if agent is None:
        stream.run_all()
        return

    for event in stream.source:
        fills = stream.emit(event)

        if fills:
            last_fill_price = fills[-1]["exec_price"]
            agent.step(book, last_fill_price)


class _TeeStream:
    """Write to the real stdout and capture complete log lines for the web UI."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._real = sys.__stdout__
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


def _price_level_top(level: PriceLevel | None) -> dict[str, float] | None:
    """Serialize one best bid or ask level for JSON."""
    if level is None:
        return None
    return {"price": float(level.price), "volume": float(level.volume)}


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Parse ``value`` as an int and clamp it into ``[low, high]``."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _load_trade_records(log_path: str | None) -> list[dict[str, Any]]:
    """Return trade records loaded from ``log_path``, or an empty list on failure."""
    if log_path is None:
        return []

    path = Path(log_path)
    if not path.is_file():
        return []

    try:
        with path.open(encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(loaded, list):
        return loaded
    return []


def _api_book_summary_payload(book: OrderBook, agent: Agent | None) -> dict[str, Any]:
    """Return summary JSON for the current order book and optional overlay agent."""
    payload: dict[str, Any] = {
        "best_bid": _price_level_top(book.best_bid()),
        "best_ask": _price_level_top(book.best_ask()),
        "spread": book.spread(),
        "mid_price": book.mid_price(),
    }
    payload["agent"] = {"current_pnl": agent.current_pnl()} if agent is not None else None
    return payload


def _api_book_depth_payload(book: OrderBook, levels: int) -> dict[str, Any]:
    """Return a depth-of-book payload for ``levels`` rows per side."""
    raw = book.depth_snapshot(levels)
    bids = [{"price": price, "volume": volume} for price, volume in raw["bids"]]
    asks = [{"price": price, "volume": volume} for price, volume in raw["asks"]]
    return {"levels": levels, "bids": bids, "asks": asks}


def _api_metrics_payload(engine: MatchingEngine) -> dict[str, Any]:
    """Return aggregate matching-engine metrics."""
    return engine.compute_metrics()


def _api_trades_payload(
    book: OrderBook,
    limit: int,
    offset: int,
    log_path: str | None,
) -> dict[str, Any]:
    """Return paginated trade records from memory or the flushed log file."""
    records: list[dict[str, Any]] = list(book.trade_log)
    if not records and log_path:
        records = _load_trade_records(log_path)

    total = len(records)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": records[offset: offset + limit],
    }


def _api_execution_log_payload(
    engine: MatchingEngine,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Return paginated matching-engine execution entries."""
    total = len(engine.execution_log)
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": engine.execution_log[offset: offset + limit],
    }


def _api_open_orders_payload(book: OrderBook) -> dict[str, Any]:
    """Return resting open orders still present in the order index."""
    return {
        "count": len(book.order_index),
        "orders": [order.to_dict() for order in book.order_index.values()],
    }


def create_post_simulation_app(
    book: OrderBook,
    engine: MatchingEngine,
    *,
    agent: Agent | None = None,
    log_path: str | None = None,
) -> Flask:
    """Build the post-simulation API app used by the CLI flow."""
    app = Flask(__name__)
    app.config["QUANTYZE_LOG_PATH"] = log_path

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "quantyze"})

    @app.get("/api/book/summary")
    def book_summary() -> Any:
        return jsonify(_api_book_summary_payload(book, agent))

    @app.get("/api/book/depth")
    def book_depth() -> Any:
        levels = _clamp_int(
            request.args.get("levels"),
            _DEFAULT_DEPTH_LEVELS,
            1,
            _MAX_DEPTH_LEVELS,
        )
        return jsonify(_api_book_depth_payload(book, levels))

    @app.get("/api/metrics")
    def metrics() -> Any:
        return jsonify(_api_metrics_payload(engine))

    @app.get("/api/trades")
    def trades() -> Any:
        limit = _clamp_int(
            request.args.get("limit"),
            _DEFAULT_TRADE_LIMIT,
            1,
            _MAX_TRADE_LIMIT,
        )
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        path = app.config.get("QUANTYZE_LOG_PATH")
        return jsonify(_api_trades_payload(book, limit, offset, str(path) if path else None))

    @app.get("/api/execution-log")
    def execution_log() -> Any:
        limit = _clamp_int(
            request.args.get("limit"),
            _DEFAULT_TRADE_LIMIT,
            1,
            _MAX_TRADE_LIMIT,
        )
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        return jsonify(_api_execution_log_payload(engine, limit, offset))

    @app.get("/api/orders/open")
    def open_orders() -> Any:
        return jsonify(_api_open_orders_payload(book))

    return app


def run_post_simulation_server(app: Flask, port: int, host: str = "127.0.0.1") -> None:
    """Start the post-simulation Flask server."""
    app.run(host=host, port=port, debug=False, threaded=True)


def _compute_class_weights(train_labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Return inverse-frequency class weights normalized to mean 1.0.

    >>> class_weights = _compute_class_weights(torch.tensor([0, 0, 1, 2]))
    >>> [round(float(class_weights[0]), 1), round(float(class_weights[1]), 1), round(float(class_weights[2]), 1)]
    [0.6, 1.2, 1.2]
    """
    counts = torch.bincount(train_labels.long(), minlength=num_classes).float()
    positive_mask = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32)

    if positive_mask.any():
        weights[positive_mask] = 1.0 / counts[positive_mask]
        weights[positive_mask] /= weights[positive_mask].mean()

    return weights


def _compute_feature_normalization(train_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-feature mean and safe standard deviation from the training split."""
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False)
    safe_std = torch.where(feature_std < 1e-8, torch.ones_like(feature_std), feature_std)
    return feature_mean, safe_std


def _export_training_csv(path: str, features: torch.Tensor, labels: torch.Tensor) -> None:
    """Write normalized model-ready features and labels to a CSV file."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(list(DataLoader.feature_names) + ["label"])
        for feature_row, label in zip(features.tolist(), labels.tolist()):
            writer.writerow(feature_row + [int(label)])


def _new_eval_stats() -> _EvalStats:
    """Return zeroed evaluation counters for the classifier metrics pass."""
    return _EvalStats(
        total=0,
        correct=0,
        true_counts=[0] * len(CLASS_NAMES),
        pred_counts=[0] * len(CLASS_NAMES),
        confusion_matrix=[[0] * len(CLASS_NAMES) for _ in CLASS_NAMES],
    )


def _update_eval_stats(
    stats: _EvalStats,
    actual_labels: list[int],
    predicted_labels: list[int]
) -> None:
    """Update ``stats`` using one evaluated batch."""
    stats.total += len(actual_labels)
    for actual_label, predicted_label in zip(actual_labels, predicted_labels):
        stats.correct += int(predicted_label == actual_label)
        stats.true_counts[actual_label] += 1
        stats.pred_counts[predicted_label] += 1
        stats.confusion_matrix[actual_label][predicted_label] += 1


def _per_class_recall(stats: _EvalStats) -> list[float]:
    """Return recall values derived from the accumulated evaluation stats."""
    recall_values = []
    for class_index, class_total in enumerate(stats.true_counts):
        if class_total == 0:
            recall_values.append(0.0)
        else:
            recall_values.append(stats.confusion_matrix[class_index][class_index] / class_total)
    return recall_values


def _evaluate_classifier(model: OrderBookNet, loader: TorchDataLoader) -> dict[str, object]:
    """Return validation metrics for the trained classifier."""
    device = next(model.parameters()).device
    model.eval()
    stats = _new_eval_stats()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            preds = torch.argmax(model(x), dim=1)
            actual_labels = y.cpu().tolist()
            predicted_labels = preds.cpu().tolist()
            _update_eval_stats(stats, actual_labels, predicted_labels)

    majority_baseline = max(stats.true_counts) / stats.total if stats.total > 0 else 0.0

    return {
        "val_accuracy": stats.correct / stats.total if stats.total > 0 else 0.0,
        "majority_baseline_accuracy": majority_baseline,
        "class_names": CLASS_NAMES,
        "val_true_counts": stats.true_counts,
        "val_pred_counts": stats.pred_counts,
        "per_class_recall": _per_class_recall(stats),
        "confusion_matrix": stats.confusion_matrix,
    }


def _load_training_arrays(data_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw feature and label tensors loaded from ``data_path``."""
    data_loader = DataLoader(data_path)
    data_loader.load_csv()
    features, labels = data_loader.build_training_dataset()

    if len(features) < 2:
        raise ValueError("Need at least two training examples to build train/validation splits.")

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _prepare_training_data(
    raw_feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor
) -> _PreparedTrainingData:
    """Return normalized tensors, index splits, and normalization stats."""
    split_generator = torch.Generator().manual_seed(111)
    permutation = torch.randperm(len(label_tensor), generator=split_generator)
    train_size = int(0.8 * len(label_tensor))
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    feature_mean, feature_std = _compute_feature_normalization(
        raw_feature_tensor[train_indices]
    )
    normalized_features = (raw_feature_tensor - feature_mean) / feature_std
    return _PreparedTrainingData(
        feature_dim=raw_feature_tensor.shape[1],
        normalized_features=normalized_features,
        label_tensor=label_tensor,
        train_indices=train_indices,
        val_indices=val_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _build_training_loaders(
    prepared_data: _PreparedTrainingData
) -> tuple[TorchDataLoader, TorchDataLoader]:
    """Return the train/validation loaders for ``prepared_data``."""
    train_dataset = TensorDataset(
        prepared_data.train_features(),
        prepared_data.train_labels(),
    )
    val_dataset = TensorDataset(
        prepared_data.val_features(),
        prepared_data.val_labels(),
    )
    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


def _build_training_metrics(
    trainer: Trainer,
    metrics: dict[str, object],
    data_path: str,
    output_paths: _TrainingOutputPaths
) -> dict[str, object]:
    """Return the persisted training metrics payload."""
    return {
        "train_loss_history": trainer.history["train_loss"],
        "val_loss_history": trainer.history["val_loss"],
        "val_accuracy": metrics["val_accuracy"],
        "majority_baseline_accuracy": metrics["majority_baseline_accuracy"],
        "class_names": metrics["class_names"],
        "val_true_counts": metrics["val_true_counts"],
        "val_pred_counts": metrics["val_pred_counts"],
        "per_class_recall": metrics["per_class_recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "dataset_path": data_path,
        "model_output_path": output_paths.model_path,
        "metrics_output_path": output_paths.metrics_path,
        "training_data_output_path": output_paths.training_data_path,
    }


def train_model(
    data_path: str,
    model_path: str = LATEST_MODEL_PATH,
    metrics_path: str = LATEST_TRAINING_METRICS_PATH,
    training_data_path: str = LATEST_TRAINING_DATA_PATH
) -> dict[str, object]:
    """Train the optional OrderBookNet model using data from <data_path>.

    This function loads training data, constructs the neural-network training
    objects, exports the derived training dataset, and saves model weights for
    later inference.
    """
    output_paths = _TrainingOutputPaths(
        model_path=model_path,
        metrics_path=metrics_path,
        training_data_path=training_data_path,
    )
    raw_feature_tensor, label_tensor = _load_training_arrays(data_path)
    prepared_data = _prepare_training_data(raw_feature_tensor, label_tensor)
    _export_training_csv(
        output_paths.training_data_path,
        prepared_data.normalized_features,
        prepared_data.label_tensor,
    )

    train_loader, val_loader = _build_training_loaders(prepared_data)
    model = OrderBookNet(feature_dim=prepared_data.feature_dim)
    trainer = Trainer(
        model,
        class_weights=_compute_class_weights(prepared_data.train_labels())
    )
    trainer.fit(train_loader, val_loader)
    trainer.save(
        output_paths.model_path,
        feature_mean=prepared_data.feature_mean.cpu().numpy(),
        feature_std=prepared_data.feature_std.cpu().numpy()
    )

    metrics = _evaluate_classifier(model, val_loader)
    training_metrics = _build_training_metrics(
        trainer,
        metrics,
        data_path,
        output_paths,
    )

    with open(output_paths.metrics_path, "w", encoding="utf-8") as file:
        json.dump(training_metrics, file, indent=2)

    print("Classifier Evaluation Metrics")
    print("=" * 30)
    print(f"Validation Accuracy: {training_metrics['val_accuracy']:.6f}")
    print(f"Majority Baseline Accuracy: {training_metrics['majority_baseline_accuracy']:.6f}")
    print(f"Validation True Counts: {training_metrics['val_true_counts']}")
    print(f"Validation Pred Counts: {training_metrics['val_pred_counts']}")
    print(f"Per-Class Recall: {training_metrics['per_class_recall']}")
    print(f"Confusion Matrix: {training_metrics['confusion_matrix']}")
    print(f"Dataset Path: {data_path}")
    print(f"Checkpoint Output: {output_paths.model_path}")
    print(f"Metrics Output: {output_paths.metrics_path}")
    print(f"Training Data Output: {output_paths.training_data_path}")
    print("=" * 30)
    return training_metrics


def run_simulation_from_args(args: argparse.Namespace) -> None:
    """Run one simulation from a prepared argument namespace."""
    active_model_status = resolve_active_model_status()
    model_path = active_model_status["model_path"]
    book, engine, stream, agent, loader = build_system(
        args,
        model_path if isinstance(model_path, str) else None,
    )

    print("Quantyze Run Configuration")
    print("=" * 30)
    print(f"Event Source: {_simulation_source_label(args)}")
    print(f"Simulation Overlay Mode: {_overlay_mode_text(active_model_status['mode'])}")
    print(f"Simulation Overlay Path: {_format_optional_path(active_model_status['model_path'])}")
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

    run_simulation(stream, agent, book)
    print_summary(engine, agent)
    if agent is None:
        print("Simulation used no model overlay.")
    else:
        print(f"Simulation used the {active_model_status['mode']} overlay.")

    if args.no_ui:
        book.flush_log(LOG_PATH)
        print(f"Execution log written to {LOG_PATH}.")
        return

    app = create_post_simulation_app(book, engine, agent=agent, log_path=LOG_PATH)
    print(f"Starting Quantyze API at http://127.0.0.1:{args.port} (Ctrl+C to stop, then log flushes).")
    try:
        run_post_simulation_server(app, args.port)
    finally:
        book.flush_log(LOG_PATH)
        print(f"Execution log written to {LOG_PATH}.")


def _run_web_simulation(params: dict[str, Any]) -> None:
    """Run one browser-triggered simulation in a background thread."""
    log_lines: list[str] = []

    with _web_lock:
        _web_sim.update(
            state="running",
            progress=0.0,
            log=log_lines,
            results=None,
            error=None,
            book=None,
            engine=None,
            agent=None,
            stream=None,
        )

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

        args = argparse.Namespace(
            data=params.get("data") or None,
            scenario=params.get("scenario") or None,
            speed=float(params.get("speed", 0.0)),
            train=False,
            port=9000,
            no_ui=True,
            web=False,
        )

        book, engine, stream, agent, loader = build_system(args, model_path)

        print("Quantyze Run Configuration")
        print("=" * 30)
        print(f"Event Source: {_simulation_source_label(args)}")
        print(f"Simulation Overlay Mode: {_overlay_mode_text(active_model_status['mode'])}")
        print(f"Simulation Overlay Path: {_format_optional_path(active_model_status['model_path'])}")
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
        with _web_lock:
            _web_sim["stream"] = stream

        for index, event in enumerate(stream.source):
            if not stream.running:
                break
            fills = stream.emit(event)
            if agent and fills:
                agent.step(book, fills[-1]["exec_price"])
            with _web_lock:
                _web_sim["progress"] = (index + 1) / max(total_events, 1) * 100

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
            "source_label": _simulation_source_label(args),
            "overlay_mode": active_model_status["mode"],
        }

        with _web_lock:
            cancelled = not stream.running and _web_sim["state"] == "idle"
            if cancelled:
                _web_sim["results"] = results
                _web_sim["book"] = book
                _web_sim["engine"] = engine
                _web_sim["agent"] = agent
                _web_sim["stream"] = None
            else:
                _web_sim["state"] = "done"
                _web_sim["progress"] = 100.0
                _web_sim["results"] = results
                _web_sim["book"] = book
                _web_sim["engine"] = engine
                _web_sim["agent"] = agent
                _web_sim["stream"] = None

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"ERROR: {exc}")
        with _web_lock:
            _web_sim["state"] = "error"
            _web_sim["error"] = str(exc)
            _web_sim["stream"] = None
            _web_sim["log"].extend(tb.splitlines())
    finally:
        sys.stdout = old_stdout


def _run_web_training(params: dict[str, Any]) -> None:
    """Run one browser-triggered training job in a background thread."""
    log_lines: list[str] = []

    with _web_lock:
        _web_train.update(
            state="running",
            progress=0.0,
            log=log_lines,
            results=None,
            error=None,
        )

    tee = _TeeStream(log_lines)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        data_path = str(params["data_path"])
        print(f"Starting training on: {data_path}")
        print("Loading dataset and building features...")
        print("(Training runs 50 epochs; this may take a minute.)")
        results = train_model(data_path)

        with _web_lock:
            _web_train["state"] = "done"
            _web_train["progress"] = 100.0
            _web_train["results"] = results
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"ERROR: {exc}")
        with _web_lock:
            _web_train["state"] = "error"
            _web_train["error"] = str(exc)
            _web_train["log"].extend(tb.splitlines())
    finally:
        sys.stdout = old_stdout


def create_web_app() -> Flask:
    """Build the browser-based Quantyze web application backed entirely by main.py."""
    app = Flask(__name__, static_folder=".", static_url_path="")

    @app.get("/")
    def index() -> Any:
        return send_from_directory(".", "index.html")

    @app.get("/frontend/<path:asset>")
    def frontend_asset(asset: str) -> Any:
        """Serve legacy frontend asset URLs used by older cached HTML."""
        return send_from_directory(".", asset)

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "quantyze-web"})

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
        with _web_lock:
            if _web_sim["state"] == "running":
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

        threading.Thread(target=_run_web_simulation, args=(params,), daemon=True).start()
        return jsonify({"started": True}), 202

    @app.get("/api/simulate/status")
    def simulation_status() -> Any:
        with _web_lock:
            snapshot = {
                "state": _web_sim["state"],
                "progress": _web_sim["progress"],
                "log": list(_web_sim["log"]),
                "results": _web_sim["results"],
                "error": _web_sim["error"],
            }
        return jsonify(snapshot)

    @app.delete("/api/simulate")
    def cancel_simulation() -> Any:
        with _web_lock:
            stream = _web_sim.get("stream")
            if _web_sim["state"] == "running" and stream is not None:
                stream.running = False
                _web_sim["state"] = "idle"
                _web_sim["stream"] = None
        return jsonify({"cancelled": True})

    @app.post("/api/train")
    def start_training() -> Any:
        with _web_lock:
            if _web_train["state"] == "running":
                return jsonify({"error": "Training already running"}), 409

        body = request.get_json(force=True, silent=True) or {}
        source = body.get("source", "sample")
        preset_map = {
            "sample": SAMPLE_DATASET_PATH,
            "huge": HUGE_DATASET_PATH,
            "lobster": LOBSTER_SAMPLE_MESSAGE_PATH,
        }

        if source == "custom":
            data_path = str(body.get("data_path") or "").strip()
            if not data_path:
                return jsonify({"error": "data_path is required for custom source"}), 400
            if not Path(data_path).exists():
                return jsonify({"error": f"File not found: {data_path}"}), 400
        else:
            data_path = preset_map.get(source, SAMPLE_DATASET_PATH)
            if not Path(data_path).exists():
                return jsonify({
                    "error": (
                        f"Dataset not found: {data_path}. "
                        f"Extract {DATASET_PACKAGE_PATH} beside main.py first."
                    )
                }), 400

        threading.Thread(target=_run_web_training, args=({"data_path": data_path},), daemon=True).start()
        return jsonify({"started": True}), 202

    @app.get("/api/train/status")
    def training_status() -> Any:
        with _web_lock:
            snapshot = {
                "state": _web_train["state"],
                "progress": _web_train["progress"],
                "log": list(_web_train["log"]),
                "results": _web_train["results"],
                "error": _web_train["error"],
            }
        return jsonify(snapshot)

    def _sim_components() -> tuple[Any, Any, Any]:
        with _web_lock:
            return _web_sim.get("book"), _web_sim.get("engine"), _web_sim.get("agent")

    @app.get("/api/book/summary")
    def web_book_summary() -> Any:
        book, _, agent = _sim_components()
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(_api_book_summary_payload(book, agent))

    @app.get("/api/book/depth")
    def web_book_depth() -> Any:
        book, _, _ = _sim_components()
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        levels = _clamp_int(request.args.get("levels"), _DEFAULT_DEPTH_LEVELS, 1, _MAX_DEPTH_LEVELS)
        return jsonify(_api_book_depth_payload(book, levels))

    @app.get("/api/metrics")
    def web_metrics() -> Any:
        _, engine, _ = _sim_components()
        if engine is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(_api_metrics_payload(engine))

    @app.get("/api/trades")
    def web_trades() -> Any:
        book, _, _ = _sim_components()
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        limit = _clamp_int(request.args.get("limit"), _DEFAULT_TRADE_LIMIT, 1, _MAX_TRADE_LIMIT)
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        return jsonify(_api_trades_payload(book, limit, offset, LOG_PATH))

    @app.get("/api/execution-log")
    def web_execution_log() -> Any:
        _, engine, _ = _sim_components()
        if engine is None:
            return jsonify({"error": "No simulation results yet"}), 404
        limit = _clamp_int(request.args.get("limit"), _DEFAULT_TRADE_LIMIT, 1, _MAX_TRADE_LIMIT)
        offset = _clamp_int(request.args.get("offset"), 0, 0, _MAX_TRADE_LIMIT)
        return jsonify(_api_execution_log_payload(engine, limit, offset))

    @app.get("/api/orders/open")
    def web_open_orders() -> Any:
        book, _, _ = _sim_components()
        if book is None:
            return jsonify({"error": "No simulation results yet"}), 404
        return jsonify(_api_open_orders_payload(book))

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
                "baseline_model": _artifact_info(MODEL_PATH),
                "baseline_metrics": _artifact_info(TRAINING_METRICS_PATH),
                "latest_model": _artifact_info(LATEST_MODEL_PATH),
                "latest_metrics": _artifact_info(LATEST_TRAINING_METRICS_PATH),
                "latest_training_data": _artifact_info(LATEST_TRAINING_DATA_PATH),
                "log": _artifact_info(LOG_PATH),
                "active_model_state": _artifact_info(ACTIVE_MODEL_STATE_PATH),
                "dataset_package": _artifact_info(DATASET_PACKAGE_PATH),
                "sample_dataset": _artifact_info(SAMPLE_DATASET_PATH),
                "huge_dataset": _artifact_info(HUGE_DATASET_PATH),
                "lobster_dataset": _artifact_info(LOBSTER_SAMPLE_MESSAGE_PATH),
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


def _build_menu_config() -> MenuConfig:
    """Return the interactive menu configuration for this project."""
    return MenuConfig(
        paths=MenuPaths(
            model_path=MODEL_PATH,
            training_metrics_path=TRAINING_METRICS_PATH,
            latest_model_path=LATEST_MODEL_PATH,
            latest_training_metrics_path=LATEST_TRAINING_METRICS_PATH,
            latest_training_data_path=LATEST_TRAINING_DATA_PATH,
            log_path=LOG_PATH,
            active_model_state_path=ACTIVE_MODEL_STATE_PATH,
        ),
        datasets=MenuDatasets(
            sample_dataset_path=SAMPLE_DATASET_PATH,
            huge_dataset_path=HUGE_DATASET_PATH,
            lobster_sample_message_path=LOBSTER_SAMPLE_MESSAGE_PATH,
            scenario_choices=SCENARIO_CHOICES,
        ),
        callbacks=MenuCallbacks(
            run_simulation=run_simulation_from_args,
            train_model=train_model,
            get_active_model_status=resolve_active_model_status,
            set_active_model=set_active_model_selection,
        ),
    )


def print_summary(engine: MatchingEngine, agent: Agent | None) -> None:
    """Print a summary of the completed Quantyze run.

    The summary may include execution metrics from the matching engine and any
    available ML-related results, such as the agent's current mark-to-market P&L.
    """

    metrics = engine.compute_metrics()
    spread = engine.book.spread()
    mid_price = engine.book.mid_price()

    print("Quantyze Summary")
    print("=" * 30)
    print("Simulation Metrics")
    print("-" * 30)
    print(f"Total Filled: {_format_metric(metrics['total_filled'])}")
    print(f"Fill Count: {_format_metric(metrics['fill_count'])}")
    print(f"Cancel Count: {_format_metric(metrics['cancel_count'])}")
    print(f"Average Slippage: {_format_metric(metrics.get('average_slippage', 0.0))}")

    if spread is None:
        print("Spread: Unavailable")
    else:
        print(f"Spread: {_format_decimal(spread)}")

    if mid_price is None:
        print("Mid Price: Unavailable")
    else:
        print(f"Mid Price: {_format_decimal(mid_price)}")

    if agent is not None:
        print("-" * 30)
        print("Agent Overlay")
        print("-" * 30)
        print(f"Agent Overlay Mark-to-Market P&L: {_format_decimal(agent.current_pnl())}")

    print("=" * 30)


def main() -> None:
    """Run the Quantyze program.

    This function parses command-line arguments, builds the system, chooses the
    requested runtime mode, executes the simulation or training flow, and
    prints final summary information.
    """

    args = parse_args()

    if args.cli:
        run_interactive_menu(_build_menu_config())
        return

    if args.ui or not args.no_ui:
        web_app = create_web_app()
        print(f"Quantyze web UI -> http://127.0.0.1:{args.port}")
        print("Press Ctrl+C to stop.")
        web_app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
        return

    if args.train:
        if args.data is None:
            raise ValueError("Training mode requires --data <csv_path>.")

        train_model(args.data)
        return

    run_simulation_from_args(args)


if __name__ == "__main__":
    main()

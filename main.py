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
import importlib
import json
import os
import sys
from dataclasses import dataclass

import torch
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

CLASS_NAMES = ["buy", "sell", "hold"]
MODEL_PATH = "model.pt"
TRAINING_METRICS_PATH = "training_metrics.json"
LATEST_MODEL_PATH = "latest_model.pt"
LATEST_TRAINING_METRICS_PATH = "latest_training_metrics.json"
LATEST_TRAINING_DATA_PATH = "latest_training_data.csv"
LOG_PATH = "log.json"
ACTIVE_MODEL_STATE_PATH = "active_model.json"
SAMPLE_DATASET_PATH = "sample_internal.csv"
HUGE_DATASET_PATH = "huge_internal.csv"
SCENARIO_CHOICES = ("balanced", "low_liquidity", "high_volatility")
ACTIVE_MODEL_MODES = ("baseline", "latest", "none")
BASELINE_MODEL_LABEL = "packaged baseline checkpoint"


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
        help="Port used for the Flask or UI server"
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Skip the Flask or UI server"
    )

    return parser.parse_args()


def _checkpoint_exists(path: str | None) -> bool:
    """Return whether ``path`` points to a non-empty checkpoint file."""
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def _dataset_label_for_path(dataset_path: str | None) -> str:
    """Return a short human-readable label for a dataset path."""
    if not dataset_path:
        return "unknown"

    dataset_name = os.path.basename(dataset_path)
    if dataset_name in {SAMPLE_DATASET_PATH, HUGE_DATASET_PATH}:
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


def _active_model_payload(mode: str, dataset_label: str | None = None) -> dict[str, object]:
    """Return the persisted payload for the selected active-model mode."""
    if mode == "baseline":
        return {
            "mode": "baseline",
            "model_path": MODEL_PATH,
            "metrics_path": TRAINING_METRICS_PATH,
            "dataset_label": dataset_label or BASELINE_MODEL_LABEL,
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
        }
    if mode == "none":
        return {
            "mode": "none",
            "model_path": None,
            "metrics_path": None,
            "dataset_label": dataset_label or "No active model",
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

    return payload


def _saved_dataset_label(payload: dict[str, object] | None) -> str | None:
    """Return the saved dataset label from ``payload`` when it is a string."""
    if payload is None:
        return None

    dataset_label = payload.get("dataset_label")
    if isinstance(dataset_label, str):
        return dataset_label
    return None


def save_active_model_selection(mode: str, dataset_label: str | None = None) -> dict[str, object]:
    """Persist and return the selected active-model payload."""
    payload = _active_model_payload(mode, dataset_label)
    with open(ACTIVE_MODEL_STATE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def resolve_active_model_status() -> dict[str, object]:
    """Return the resolved active-model status, applying safe fallbacks."""
    raw_payload = _load_active_model_payload()
    note = ""

    if raw_payload is None:
        requested_mode = "baseline" if _checkpoint_exists(MODEL_PATH) else "none"
        note = "Active model state was missing or invalid; using the default selection."
    else:
        requested_mode = str(raw_payload["mode"])

    if requested_mode == "latest":
        if _checkpoint_exists(LATEST_MODEL_PATH):
            resolved_payload = _active_model_payload("latest", _saved_dataset_label(raw_payload))
        elif _checkpoint_exists(MODEL_PATH):
            resolved_payload = _active_model_payload("baseline")
            note = "Latest checkpoint was unavailable; fell back to the baseline model."
        else:
            resolved_payload = _active_model_payload("none")
            note = "Latest checkpoint was unavailable; running without a model."
    elif requested_mode == "baseline":
        if _checkpoint_exists(MODEL_PATH):
            resolved_payload = _active_model_payload("baseline")
        else:
            resolved_payload = _active_model_payload("none")
            note = "Baseline checkpoint was unavailable; running without a model."
    else:
        resolved_payload = _active_model_payload("none")

    if raw_payload != resolved_payload:
        save_active_model_selection(
            str(resolved_payload["mode"]),
            _saved_dataset_label(resolved_payload),
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
    save_active_model_selection(mode)
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
    print(f"Simulation Overlay Mode: {active_model_status['mode']}")
    print(f"Simulation Overlay Path: {active_model_status['model_path'] or 'None'}")
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

    server_module = importlib.import_module("server")
    create_app = server_module.create_app
    run_server = server_module.run_server
    app = create_app(book, engine, agent=agent, log_path=LOG_PATH)
    print(f"Starting Quantyze API at http://127.0.0.1:{args.port} (Ctrl+C to stop, then log flushes).")
    try:
        run_server(app, args.port)
    finally:
        book.flush_log(LOG_PATH)
        print(f"Execution log written to {LOG_PATH}.")


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
    print(f"Total Filled: {metrics['total_filled']}")
    print(f"Fill Count: {metrics['fill_count']}")
    print(f"Cancel Count: {metrics['cancel_count']}")
    print(f"Average Slippage: {metrics.get('average_slippage', 0.0)}")

    if spread is None:
        print("Spread: Unavailable")
    else:
        print(f"Spread: {spread}")

    if mid_price is None:
        print("Mid Price: Unavailable")
    else:
        print(f"Mid Price: {mid_price}")

    if agent is not None:
        print("-" * 30)
        print("Agent Overlay")
        print("-" * 30)
        print(f"Agent Overlay Mark-to-Market P&L: {agent.current_pnl()}")

    print("=" * 30)


def main() -> None:
    """Run the Quantyze program.

    This function parses command-line arguments, builds the system, chooses the
    requested runtime mode, executes the simulation or training flow, and
    prints final summary information.
    """

    if len(sys.argv) == 1:
        run_interactive_menu(_build_menu_config())
        return

    args = parse_args()

    if args.train:
        if args.data is None:
            raise ValueError("Training mode requires --data <csv_path>.")

        train_model(args.data)
        return

    run_simulation_from_args(args)


if __name__ == "__main__":
    main()

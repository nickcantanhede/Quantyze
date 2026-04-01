"""Quantyze main entry point.

Module Description
==================
This module contains the top-level architecture for Quantyze, responsible for building the
system components, loading or generating data, running the event stream through the matching
engine and print summary of the simulation results.

The main module is responsible for:
- parsing command-line arguments
- constructing the core Quantyze system objects
- selecting between dataset replay and synthetic scenarions
- running the simulation loop
- trigerring optional ML training or inference flows
- printing final summary information and flushing logs

This module does not implement BST logic, order matching, event validation, or
visualization logic itself. It coordinates the other modules and serves as the
program entry point.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from cli_menu import MenuConfig, interactive_menu as run_interactive_menu
from data_loader import DataLoader
from event_stream import EventStream
from matching_engine import MatchingEngine
from neural_net import Agent, Trainer, OrderBookNet, load_agent
from order_book import OrderBook

CLASS_NAMES = ["buy", "sell", "hold"]
MODEL_PATH = "model.pt"
TRAINING_METRICS_PATH = "training_metrics.json"
TRAINING_DATA_PATH = "training_data.csv"
LATEST_MODEL_PATH = "latest_model.pt"
LATEST_TRAINING_METRICS_PATH = "latest_training_metrics.json"
LATEST_TRAINING_DATA_PATH = "latest_training_data.csv"
LOG_PATH = "log.json"
SAMPLE_DATASET_PATH = "sample_internal.csv"
HUGE_DATASET_PATH = "huge_internal.csv"
SCENARIO_CHOICES = ("balanced", "low_liquidity", "high_volatility")


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


def build_system(
    args: argparse.Namespace
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

    if args.train:
        agent = None
    else:
        agent = load_agent(MODEL_PATH)

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
    """Return inverse-frequency class weights normalized to mean 1.0."""
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
        writer.writerow(list(DataLoader.FEATURE_NAMES) + ["label"])
        for feature_row, label in zip(features.tolist(), labels.tolist()):
            writer.writerow(feature_row + [int(label)])


def _evaluate_classifier(model: OrderBookNet, loader: TorchDataLoader) -> dict[str, object]:
    """Return validation metrics for the trained classifier."""
    device = next(model.parameters()).device
    model.eval()

    total = 0
    correct = 0
    true_counts = [0] * len(CLASS_NAMES)
    pred_counts = [0] * len(CLASS_NAMES)
    confusion_matrix = [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES]

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            preds = torch.argmax(model(x), dim=1)

            y_cpu = y.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            total += len(y_cpu)
            correct += sum(int(pred == actual) for pred, actual in zip(preds_cpu, y_cpu))

            for actual, pred in zip(y_cpu, preds_cpu):
                true_counts[actual] += 1
                pred_counts[pred] += 1
                confusion_matrix[actual][pred] += 1

    majority_baseline = max(true_counts) / total if total > 0 else 0.0
    per_class_recall = []
    for class_index, class_total in enumerate(true_counts):
        if class_total == 0:
            per_class_recall.append(0.0)
        else:
            per_class_recall.append(confusion_matrix[class_index][class_index] / class_total)

    return {
        "val_accuracy": correct / total if total > 0 else 0.0,
        "majority_baseline_accuracy": majority_baseline,
        "class_names": CLASS_NAMES,
        "val_true_counts": true_counts,
        "val_pred_counts": pred_counts,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion_matrix,
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

    data_loader = DataLoader(data_path)
    data_loader.load_csv()
    features, labels = data_loader.build_training_dataset()

    if len(features) < 2:
        raise ValueError("Need at least two training examples to build train/validation splits.")

    raw_feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    split_generator = torch.Generator().manual_seed(111)
    permutation = torch.randperm(len(label_tensor), generator=split_generator)
    train_size = int(0.8 * len(label_tensor))
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_features = raw_feature_tensor[train_indices]
    feature_mean, feature_std = _compute_feature_normalization(train_features)
    normalized_feature_tensor = (raw_feature_tensor - feature_mean) / feature_std
    _export_training_csv(training_data_path, normalized_feature_tensor, label_tensor)

    train_labels = label_tensor[train_indices]
    class_weights = _compute_class_weights(train_labels)

    train_dataset = TensorDataset(normalized_feature_tensor[train_indices], train_labels)
    val_dataset = TensorDataset(normalized_feature_tensor[val_indices], label_tensor[val_indices])
    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)

    model = OrderBookNet(feature_dim=features.shape[1])
    trainer = Trainer(model, class_weights=class_weights)
    trainer.fit(train_loader, val_loader)
    trainer.save(
        model_path,
        feature_mean=feature_mean.cpu().numpy(),
        feature_std=feature_std.cpu().numpy()
    )

    metrics = _evaluate_classifier(model, val_loader)
    training_metrics = {
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
        "model_output_path": model_path,
        "training_data_output_path": training_data_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(training_metrics, file, indent=2)

    print("Training Evaluation Metrics")
    print("=" * 30)
    print(f"Validation Accuracy: {training_metrics['val_accuracy']:.6f}")
    print(f"Majority Baseline Accuracy: {training_metrics['majority_baseline_accuracy']:.6f}")
    print(f"Validation True Counts: {training_metrics['val_true_counts']}")
    print(f"Validation Pred Counts: {training_metrics['val_pred_counts']}")
    print(f"Per-Class Recall: {training_metrics['per_class_recall']}")
    print(f"Confusion Matrix: {training_metrics['confusion_matrix']}")
    print(f"Dataset Path: {data_path}")
    print(f"Checkpoint Output: {model_path}")
    print(f"Metrics Output: {metrics_path}")
    print(f"Training Data Output: {training_data_path}")
    print("=" * 30)
    return training_metrics


def run_simulation_from_args(args: argparse.Namespace) -> None:
    """Run one simulation from a prepared argument namespace."""
    book, engine, stream, agent, loader = build_system(args)

    if agent is None:
        print(f"Running without ML checkpoint; {MODEL_PATH} is missing or invalid.")
    else:
        print(f"Loaded saved model checkpoint from {MODEL_PATH}.")

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

    if args.no_ui:
        book.flush_log(LOG_PATH)
        print(f"Execution log written to {LOG_PATH}.")
        return

    from server import create_app, run_server

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
        model_path=MODEL_PATH,
        training_metrics_path=TRAINING_METRICS_PATH,
        training_data_path=TRAINING_DATA_PATH,
        latest_model_path=LATEST_MODEL_PATH,
        latest_training_metrics_path=LATEST_TRAINING_METRICS_PATH,
        latest_training_data_path=LATEST_TRAINING_DATA_PATH,
        log_path=LOG_PATH,
        sample_dataset_path=SAMPLE_DATASET_PATH,
        huge_dataset_path=HUGE_DATASET_PATH,
        scenario_choices=SCENARIO_CHOICES,
        run_simulation=run_simulation_from_args,
        train_model=train_model
    )


def print_summary(engine: MatchingEngine, agent: Agent | None) -> None:
    """Print a summary of the completed Quantyze run.

    The summary may include execution metrics from the matching engine and any
    available ML-related results, such as the agent's current mark-to-market P&L.
    """

    metrics = engine.compute_metrics()
    spread = engine.book.spread()
    mid_price = engine.book.mid_price()

    print("Quantyze Summary: Matching Engine Metrics")
    print("="*30)

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
        print(f"Current Mark-to-Market P&L: {agent.current_pnl()}")

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

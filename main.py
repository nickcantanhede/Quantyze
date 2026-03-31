"""Quantyze main entry point.

Module Description
==================
This module contains the top-level architecture for Quantyze, responsible for building the
system components, loading or generating data, running the event stream through the matching
engine and print summary of the simulation results.

The main module is responsible for:
- parsing command-line arguments
- constructing the core Quantyze system objects
- selecting between datast replay and synthetic scenarions
- runnig the simulation loop
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
import os
import sys
import zipfile

import torch
from flask import Flask
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

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

        if fills != []:
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
    book.flush_log(LOG_PATH)
    print(f"Execution log written to {LOG_PATH}.")


def _print_metrics_file(path: str, heading: str) -> bool:
    """Print one metrics artifact; return whether the file was displayed."""
    try:
        with open(path, encoding="utf-8") as file:
            metrics = json.load(file)
    except FileNotFoundError:
        return False
    except json.JSONDecodeError:
        print(f"Could not read {path}; the file is not valid JSON.")
        return True
    except OSError as exc:
        print(f"Could not open {path}: {exc}")
        return True

    print(heading)
    print("=" * 30)
    print(f"Validation Accuracy: {metrics.get('val_accuracy', 'Unavailable')}")
    print(f"Majority Baseline Accuracy: {metrics.get('majority_baseline_accuracy', 'Unavailable')}")
    print(f"Per-Class Recall: {metrics.get('per_class_recall', 'Unavailable')}")
    print(f"Confusion Matrix: {metrics.get('confusion_matrix', 'Unavailable')}")
    if 'dataset_path' in metrics:
        print(f"Dataset Path: {metrics['dataset_path']}")
    if 'model_output_path' in metrics:
        print(f"Checkpoint Output: {metrics['model_output_path']}")
    print("=" * 30)
    return True


def print_saved_training_metrics() -> None:
    """Print the packaged metrics and any newer training metrics if available."""
    displayed_any = False

    displayed_any = _print_metrics_file(
        TRAINING_METRICS_PATH, "Saved Baseline Metrics"
    ) or displayed_any
    displayed_any = _print_metrics_file(
        LATEST_TRAINING_METRICS_PATH, "Latest Training Metrics"
    ) or displayed_any

    if not displayed_any:
        print("No saved metrics found. Train a model first.")


def print_baseline_training_metrics() -> None:
    """Print only the packaged baseline metrics artifact."""
    if not _print_metrics_file(TRAINING_METRICS_PATH, "Saved Baseline Metrics"):
        print(f"Baseline metrics not found at {TRAINING_METRICS_PATH}.")


def print_latest_training_metrics() -> None:
    """Print only the newest training metrics artifact, if present."""
    if not _print_metrics_file(LATEST_TRAINING_METRICS_PATH, "Latest Training Metrics"):
        print(f"Latest metrics not found at {LATEST_TRAINING_METRICS_PATH}.")


def _prompt_text(prompt: str) -> str | None:
    """Return stripped terminal input, or None if the prompt is cancelled."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return None


def _prompt_scenario() -> str | None:
    """Prompt until a valid synthetic scenario is chosen or cancelled."""
    scenario_text = ", ".join(SCENARIO_CHOICES)
    while True:
        scenario = _prompt_text(
            f"Choose a scenario ({scenario_text}) or press Enter to cancel: "
        )

        if scenario is None or scenario == "":
            print("Scenario selection cancelled.")
            return None

        if scenario in SCENARIO_CHOICES:
            return scenario

        print(f"Invalid scenario. Please choose one of: {scenario_text}.")


def _prompt_replay_speed() -> float | None:
    """Prompt for a non-negative replay speed; blank returns the default 0.0."""
    while True:
        speed_text = _prompt_text("Enter replay speed (blank uses 0.0): ")
        if speed_text is None:
            return None

        if speed_text == "":
            return 0.0

        try:
            speed = float(speed_text)
        except ValueError:
            print("Replay speed must be numeric.")
            continue

        if speed < 0:
            print("Replay speed must be non-negative.")
            continue

        return speed


def _prompt_dataset_path(action: str) -> str | None:
    """Prompt for a dataset path or return None when cancelled."""
    data_path = _prompt_text(f"Enter the CSV path to {action} (blank cancels): ")
    if data_path is None or data_path == "":
        print("Path entry cancelled.")
        return None
    return data_path


def _build_runtime_args(
    data: str | None = None,
    scenario: str | None = None,
    speed: float = 0.0,
    train: bool = False
) -> argparse.Namespace:
    """Return a namespace shaped like the CLI parser output."""
    return argparse.Namespace(
        data=data,
        scenario=scenario,
        speed=speed,
        train=train,
        port=9000,
        no_ui=True
    )


def _print_packaged_dataset_hint(dataset_name: str) -> None:
    """Explain how to make packaged datasets available to the menu."""
    print(
        f"Could not find packaged dataset at {dataset_name}. "
        "Extract quantyze_datasets.zip beside main.py and try again."
    )


def _run_simulation_menu(args: argparse.Namespace) -> None:
    """Run one simulation from the interactive menu with friendly error handling."""
    try:
        run_simulation_from_args(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Simulation failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive menu guard
        print(f"Simulation failed unexpectedly: {type(exc).__name__}: {exc}")


def _run_training_menu(data_path: str, packaged: bool = False) -> None:
    """Run the training flow from the interactive menu and report any failures."""
    if not os.path.exists(data_path):
        if packaged:
            _print_packaged_dataset_hint(data_path)
        else:
            print(f"Could not find dataset at {data_path}.")
        return

    try:
        train_model(data_path)
        print(
            f"Saved baseline checkpoint remains at {MODEL_PATH}. "
            f"New training outputs were written to {LATEST_MODEL_PATH} and "
            f"{LATEST_TRAINING_METRICS_PATH}."
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Training failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive menu guard
        print(f"Training failed unexpectedly: {type(exc).__name__}: {exc}")


def _print_training_output_targets() -> None:
    """Print baseline and latest training artifact destinations."""
    print("Baseline saved-state artifacts")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(MODEL_PATH)}")
    print(f"Metrics: {os.path.abspath(TRAINING_METRICS_PATH)}")
    print(f"Training Data: {os.path.abspath(TRAINING_DATA_PATH)}")
    print("=" * 30)
    print("Latest training outputs")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(LATEST_MODEL_PATH)}")
    print(f"Metrics: {os.path.abspath(LATEST_TRAINING_METRICS_PATH)}")
    print(f"Training Data: {os.path.abspath(LATEST_TRAINING_DATA_PATH)}")
    print("Interactive and direct retraining write only to the latest_* artifacts.")
    print("=" * 30)


def _print_simulation_configuration() -> None:
    """Print the current baseline simulation configuration."""
    checkpoint_loaded = load_agent(MODEL_PATH) is not None
    print("Simulation Configuration")
    print("=" * 30)
    print(f"Baseline checkpoint path: {os.path.abspath(MODEL_PATH)}")
    print(f"Baseline checkpoint available: {checkpoint_loaded}")
    print(f"Default execution log path: {os.path.abspath(LOG_PATH)}")
    print(f"Supported synthetic scenarios: {', '.join(SCENARIO_CHOICES)}")
    print("=" * 30)


def _print_artifact_status() -> None:
    """Print existence and size details for important project artifacts."""
    artifact_names = [
        MODEL_PATH,
        TRAINING_METRICS_PATH,
        TRAINING_DATA_PATH,
        LATEST_MODEL_PATH,
        LATEST_TRAINING_METRICS_PATH,
        LATEST_TRAINING_DATA_PATH,
        "quantyze_datasets.zip"
    ]

    print("Artifact Status")
    print("=" * 30)
    for artifact_name in artifact_names:
        artifact_path = os.path.abspath(artifact_name)
        exists = os.path.exists(artifact_name)
        size_text = f"{os.path.getsize(artifact_name)} bytes" if exists else "Missing"
        print(f"{artifact_name}: {'Present' if exists else 'Missing'}")
        print(f"  Path: {artifact_path}")
        print(f"  Size: {size_text}")
    print("=" * 30)


def _print_log_summary() -> None:
    """Print a short summary of the current execution log."""
    if not os.path.exists(LOG_PATH):
        print(f"No execution log found at {LOG_PATH}. Run a simulation first.")
        return

    try:
        with open(LOG_PATH, encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError:
        print(f"Could not read {LOG_PATH}; the file is not valid JSON.")
        return
    except OSError as exc:
        print(f"Could not open {LOG_PATH}: {exc}")
        return

    if not isinstance(records, list):
        print(f"{LOG_PATH} does not contain a list of execution records.")
        return

    print("Execution Log Summary")
    print("=" * 30)
    print(f"Record Count: {len(records)}")
    if records:
        print(f"First Record: {json.dumps(records[0])}")
        print(f"Last Record: {json.dumps(records[-1])}")
    print("=" * 30)


def _print_dataset_package_contents() -> None:
    """List the contents of quantyze_datasets.zip if it exists."""
    package_path = "quantyze_datasets.zip"
    if not os.path.exists(package_path):
        print(f"Dataset package not found at {package_path}.")
        return

    try:
        with zipfile.ZipFile(package_path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"Could not read {package_path}: {exc}")
        return

    print("Dataset Package Contents")
    print("=" * 30)
    for member in members:
        print(f"{member.filename} ({member.file_size} bytes)")
    print("=" * 30)


def _print_help_reference() -> None:
    """Print a concise command reference matching the README."""
    print("Quantyze Command Reference")
    print("=" * 30)
    print("Interactive menu:")
    print("  Mac/Linux: python3 main.py")
    print("  Windows:   py -3 main.py")
    print("Direct simulation:")
    print("  Mac/Linux: python3 main.py --no-ui")
    print("  Windows:   py -3 main.py --no-ui")
    print("Train on packaged sample dataset:")
    print(f"  Mac/Linux: python3 main.py --train --data {SAMPLE_DATASET_PATH}")
    print(f"  Windows:   py -3 main.py --train --data {SAMPLE_DATASET_PATH}")
    print("Train on packaged huge dataset:")
    print(f"  Mac/Linux: python3 main.py --train --data {HUGE_DATASET_PATH}")
    print(f"  Windows:   py -3 main.py --train --data {HUGE_DATASET_PATH}")
    print("Train on a custom dataset:")
    print("  Mac/Linux: python3 main.py --train --data <csv_path>")
    print("  Windows:   py -3 main.py --train --data <csv_path>")
    print("Dataset package note:")
    print(
        "  Extract quantyze_datasets.zip beside main.py before using packaged "
        "sample or huge training options."
    )
    print("Supported dataset types:")
    print("  - internal Quantyze CSV")
    print("  - raw LOBSTER message CSV; the paired orderbook path is inferred by filename")
    print("=" * 30)


def _simulation_menu() -> None:
    """Run the simulation submenu until the user chooses to go back."""
    while True:
        print("\nSimulation Menu")
        print("=" * 30)
        print("1. Run default saved-model simulation")
        print("2. Run synthetic scenario")
        print("3. Replay from dataset path")
        print("4. View current simulation configuration")
        print("5. Back")

        choice = _prompt_text("Select an option (1-5): ")
        if choice is None or choice == "5":
            return

        if choice == "1":
            _run_simulation_menu(_build_runtime_args())
        elif choice == "2":
            scenario = _prompt_scenario()
            if scenario is None:
                continue

            speed = _prompt_replay_speed()
            if speed is None:
                continue

            _run_simulation_menu(_build_runtime_args(scenario=scenario, speed=speed))
        elif choice == "3":
            data_path = _prompt_dataset_path("replay")
            if data_path is None:
                continue

            speed = _prompt_replay_speed()
            if speed is None:
                continue

            _run_simulation_menu(_build_runtime_args(data=data_path, speed=speed))
        elif choice == "4":
            _print_simulation_configuration()
        else:
            print("Invalid option. Please enter a number from 1 to 5.")


def _training_menu() -> None:
    """Run the training submenu until the user chooses to go back."""
    while True:
        print("\nTraining Menu")
        print("=" * 30)
        print(f"1. Train on packaged {SAMPLE_DATASET_PATH}")
        print(f"2. Train on packaged {HUGE_DATASET_PATH}")
        print("3. Train on a custom dataset path")
        print("4. View training output targets")
        print("5. Back")

        choice = _prompt_text("Select an option (1-5): ")
        if choice is None or choice == "5":
            return

        if choice == "1":
            _run_training_menu(SAMPLE_DATASET_PATH, packaged=True)
        elif choice == "2":
            _run_training_menu(HUGE_DATASET_PATH, packaged=True)
        elif choice == "3":
            data_path = _prompt_dataset_path("train on")
            if data_path is None:
                continue
            _run_training_menu(data_path)
        elif choice == "4":
            _print_training_output_targets()
        else:
            print("Invalid option. Please enter a number from 1 to 5.")


def _artifacts_menu() -> None:
    """Run the artifacts and metrics submenu until the user chooses to go back."""
    while True:
        print("\nArtifacts & Metrics Menu")
        print("=" * 30)
        print("1. View baseline metrics")
        print("2. View latest metrics")
        print("3. View both baseline and latest metrics")
        print("4. View artifact status")
        print("5. View log summary")
        print("6. View dataset package contents")
        print("7. Back")

        choice = _prompt_text("Select an option (1-7): ")
        if choice is None or choice == "7":
            return

        if choice == "1":
            print_baseline_training_metrics()
        elif choice == "2":
            print_latest_training_metrics()
        elif choice == "3":
            print_saved_training_metrics()
        elif choice == "4":
            _print_artifact_status()
        elif choice == "5":
            _print_log_summary()
        elif choice == "6":
            _print_dataset_package_contents()
        else:
            print("Invalid option. Please enter a number from 1 to 7.")


def _help_menu() -> None:
    """Run the help submenu until the user chooses to go back."""
    while True:
        print("\nHelp / Command Reference")
        print("=" * 30)
        print("1. View command reference")
        print("2. Back")

        choice = _prompt_text("Select an option (1-2): ")
        if choice is None or choice == "2":
            return

        if choice == "1":
            _print_help_reference()
        else:
            print("Invalid option. Please enter 1 or 2.")


def interactive_menu() -> None:
    """Run the TA-facing interactive menu until the user chooses to exit."""
    while True:
        print("\nQuantyze Interactive Menu")
        print("=" * 30)
        print("1. Simulation")
        print("2. Training")
        print("3. Artifacts & Metrics")
        print("4. Help / Command Reference")
        print("5. Exit")

        choice = _prompt_text("Select an option (1-5): ")
        if choice is None or choice == "5":
            print("Exiting Quantyze.")
            return

        if choice == "1":
            _simulation_menu()
        elif choice == "2":
            _training_menu()
        elif choice == "3":
            _artifacts_menu()
        elif choice == "4":
            _help_menu()
        else:
            print("Invalid option. Please enter a number from 1 to 5.")


def start_flask(app: Flask, port: int) -> None:
    """Start the Quantyze Flask application on the given port.

    This function is responsible only for launching the API or UI server layer
    after the core system has been initialized.
    """

    app.run(host='127.0.0.1', port=port, debug=False)


def print_summary(engine: MatchingEngine, agent: Agent | None) -> None:
    """Print a summary of the completed Quantyze run.

    The summary may include execution metrics from the matching engine and any
    available ML-related results, such as the agent's total P&L.
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
        print(f"Total P&L: {agent.total_pnl()}")

    print("=" * 30)


def main() -> None:
    """Run the Quantyze program.

    This function parses command-line arguments, builds the system, chooses the
    requested runtime mode, executes the simulation or training flow, and
    prints final summary information.
    """

    if len(sys.argv) == 1:
        interactive_menu()
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

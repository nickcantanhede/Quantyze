"""Interactive terminal menu for Quantyze.

Module Description
===============================
This module contains the CLI menu and the small read-only artifact
inspection helpers that support it.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

from neural_net import load_agent

DATASET_PACKAGE_PATH = "quantyze_datasets.zip"


@dataclass(frozen=True)
class MenuConfig:
    """Configuration and callbacks used by the interactive menu."""

    model_path: str
    training_metrics_path: str
    training_data_path: str
    latest_model_path: str
    latest_training_metrics_path: str
    latest_training_data_path: str
    log_path: str
    sample_dataset_path: str
    huge_dataset_path: str
    scenario_choices: tuple[str, ...]
    run_simulation: Callable[[argparse.Namespace], None]
    train_model: Callable[[str], dict[str, object]]


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


def print_saved_training_metrics(config: MenuConfig) -> None:
    """Print the packaged metrics and any newer training metrics if available."""
    displayed_any = False

    displayed_any = _print_metrics_file(
        config.training_metrics_path, "Saved Baseline Metrics"
    ) or displayed_any
    displayed_any = _print_metrics_file(
        config.latest_training_metrics_path, "Latest Training Metrics"
    ) or displayed_any

    if not displayed_any:
        print("No saved metrics found. Train a model first.")


def print_baseline_training_metrics(config: MenuConfig) -> None:
    """Print only the packaged baseline metrics artifact."""
    if not _print_metrics_file(config.training_metrics_path, "Saved Baseline Metrics"):
        print(f"Baseline metrics not found at {config.training_metrics_path}.")


def print_latest_training_metrics(config: MenuConfig) -> None:
    """Print only the newest training metrics artifact, if present."""
    if not _print_metrics_file(config.latest_training_metrics_path, "Latest Training Metrics"):
        print(f"Latest metrics not found at {config.latest_training_metrics_path}.")


def _prompt_text(prompt: str) -> str | None:
    """Return stripped terminal input, or None if the prompt is cancelled."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return None


def _prompt_scenario(config: MenuConfig) -> str | None:
    """Prompt until a valid synthetic scenario is chosen or cancelled."""
    scenario_text = ", ".join(config.scenario_choices)
    while True:
        scenario = _prompt_text(
            f"Choose a scenario ({scenario_text}) or press Enter to cancel: "
        )

        if scenario is None or scenario == "":
            print("Scenario selection cancelled.")
            return None

        if scenario in config.scenario_choices:
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
        f"Extract {DATASET_PACKAGE_PATH} beside main.py and try again."
    )


def _run_simulation_menu(config: MenuConfig, args: argparse.Namespace) -> None:
    """Run one simulation from the interactive menu with friendly error handling."""
    try:
        config.run_simulation(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Simulation failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive menu guard
        print(f"Simulation failed unexpectedly: {type(exc).__name__}: {exc}")


def _run_training_menu(config: MenuConfig, data_path: str, packaged: bool = False) -> None:
    """Run the training flow from the interactive menu and report any failures."""
    if not os.path.exists(data_path):
        if packaged:
            _print_packaged_dataset_hint(data_path)
        else:
            print(f"Could not find dataset at {data_path}.")
        return

    try:
        config.train_model(data_path)
        print(
            f"Saved baseline checkpoint remains at {config.model_path}. "
            f"New training outputs were written to {config.latest_model_path} and "
            f"{config.latest_training_metrics_path}."
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Training failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive menu guard
        print(f"Training failed unexpectedly: {type(exc).__name__}: {exc}")


def _print_training_output_targets(config: MenuConfig) -> None:
    """Print baseline and latest training artifact destinations."""
    print("Baseline saved-state artifacts")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(config.model_path)}")
    print(f"Metrics: {os.path.abspath(config.training_metrics_path)}")
    print(f"Training Data: {os.path.abspath(config.training_data_path)}")
    print("=" * 30)
    print("Latest training outputs")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(config.latest_model_path)}")
    print(f"Metrics: {os.path.abspath(config.latest_training_metrics_path)}")
    print(f"Training Data: {os.path.abspath(config.latest_training_data_path)}")
    print("Interactive and direct retraining write only to the latest_* artifacts.")
    print("=" * 30)


def _print_simulation_configuration(config: MenuConfig) -> None:
    """Print the current baseline simulation configuration."""
    checkpoint_loaded = load_agent(config.model_path) is not None
    print("Simulation Configuration")
    print("=" * 30)
    print(f"Baseline checkpoint path: {os.path.abspath(config.model_path)}")
    print(f"Baseline checkpoint available: {checkpoint_loaded}")
    print(f"Default execution log path: {os.path.abspath(config.log_path)}")
    print(f"Supported synthetic scenarios: {', '.join(config.scenario_choices)}")
    print("=" * 30)


def _print_artifact_status(config: MenuConfig) -> None:
    """Print existence and size details for important project artifacts."""
    artifact_names = [
        config.model_path,
        config.training_metrics_path,
        config.training_data_path,
        config.latest_model_path,
        config.latest_training_metrics_path,
        config.latest_training_data_path,
        DATASET_PACKAGE_PATH
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


def _print_log_summary(config: MenuConfig) -> None:
    """Print a short summary of the current execution log."""
    if not os.path.exists(config.log_path):
        print(f"No execution log found at {config.log_path}. Run a simulation first.")
        return

    try:
        with open(config.log_path, encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError:
        print(f"Could not read {config.log_path}; the file is not valid JSON.")
        return
    except OSError as exc:
        print(f"Could not open {config.log_path}: {exc}")
        return

    if not isinstance(records, list):
        print(f"{config.log_path} does not contain a list of execution records.")
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
    if not os.path.exists(DATASET_PACKAGE_PATH):
        print(f"Dataset package not found at {DATASET_PACKAGE_PATH}.")
        return

    try:
        with zipfile.ZipFile(DATASET_PACKAGE_PATH) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"Could not read {DATASET_PACKAGE_PATH}: {exc}")
        return

    print("Dataset Package Contents")
    print("=" * 30)
    for member in members:
        print(f"{member.filename} ({member.file_size} bytes)")
    print("=" * 30)


def _print_help_reference(config: MenuConfig) -> None:
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
    print(f"  Mac/Linux: python3 main.py --train --data {config.sample_dataset_path}")
    print(f"  Windows:   py -3 main.py --train --data {config.sample_dataset_path}")
    print("Train on packaged huge dataset:")
    print(f"  Mac/Linux: python3 main.py --train --data {config.huge_dataset_path}")
    print(f"  Windows:   py -3 main.py --train --data {config.huge_dataset_path}")
    print("Train on a custom dataset:")
    print("  Mac/Linux: python3 main.py --train --data <csv_path>")
    print("  Windows:   py -3 main.py --train --data <csv_path>")
    print("Dataset package note:")
    print(
        f"  Extract {DATASET_PACKAGE_PATH} beside main.py before using packaged "
        "sample or huge training options."
    )
    print("Supported dataset types:")
    print("  - internal Quantyze CSV")
    print("  - raw LOBSTER message CSV; the paired orderbook path is inferred by filename")
    print("=" * 30)


def _simulation_menu(config: MenuConfig) -> None:
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
            _run_simulation_menu(config, _build_runtime_args())
        elif choice == "2":
            scenario = _prompt_scenario(config)
            if scenario is None:
                continue

            speed = _prompt_replay_speed()
            if speed is None:
                continue

            _run_simulation_menu(config, _build_runtime_args(scenario=scenario, speed=speed))
        elif choice == "3":
            data_path = _prompt_dataset_path("replay")
            if data_path is None:
                continue

            speed = _prompt_replay_speed()
            if speed is None:
                continue

            _run_simulation_menu(config, _build_runtime_args(data=data_path, speed=speed))
        elif choice == "4":
            _print_simulation_configuration(config)
        else:
            print("Invalid option. Please enter a number from 1 to 5.")


def _training_menu(config: MenuConfig) -> None:
    """Run the training submenu until the user chooses to go back."""
    while True:
        print("\nTraining Menu")
        print("=" * 30)
        print(f"1. Train on packaged {config.sample_dataset_path}")
        print(f"2. Train on packaged {config.huge_dataset_path}")
        print("3. Train on a custom dataset path")
        print("4. View training output targets")
        print("5. Back")

        choice = _prompt_text("Select an option (1-5): ")
        if choice is None or choice == "5":
            return

        if choice == "1":
            _run_training_menu(config, config.sample_dataset_path, packaged=True)
        elif choice == "2":
            _run_training_menu(config, config.huge_dataset_path, packaged=True)
        elif choice == "3":
            data_path = _prompt_dataset_path("train on")
            if data_path is None:
                continue
            _run_training_menu(config, data_path)
        elif choice == "4":
            _print_training_output_targets(config)
        else:
            print("Invalid option. Please enter a number from 1 to 5.")


def _artifacts_menu(config: MenuConfig) -> None:
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
            print_baseline_training_metrics(config)
        elif choice == "2":
            print_latest_training_metrics(config)
        elif choice == "3":
            print_saved_training_metrics(config)
        elif choice == "4":
            _print_artifact_status(config)
        elif choice == "5":
            _print_log_summary(config)
        elif choice == "6":
            _print_dataset_package_contents()
        else:
            print("Invalid option. Please enter a number from 1 to 7.")


def _help_menu(config: MenuConfig) -> None:
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
            _print_help_reference(config)
        else:
            print("Invalid option. Please enter 1 or 2.")


def interactive_menu(config: MenuConfig) -> None:
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
            _simulation_menu(config)
        elif choice == "2":
            _training_menu(config)
        elif choice == "3":
            _artifacts_menu(config)
        elif choice == "4":
            _help_menu(config)
        else:
            print("Invalid option. Please enter a number from 1 to 5.")

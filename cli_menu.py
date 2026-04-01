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

DATASET_PACKAGE_PATH = "quantyze_datasets.zip"


@dataclass(frozen=True)
class MenuConfig:
    """Configuration and callbacks used by the interactive menu."""

    model_path: str
    training_metrics_path: str
    latest_model_path: str
    latest_training_metrics_path: str
    latest_training_data_path: str
    log_path: str
    sample_dataset_path: str
    huge_dataset_path: str
    scenario_choices: tuple[str, ...]
    active_model_state_path: str
    run_simulation: Callable[[argparse.Namespace], None]
    train_model: Callable[[str], dict[str, object]]
    get_active_model_status: Callable[[], dict[str, object]]
    set_active_model: Callable[[str], dict[str, object]]


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


def _prompt_yes_no(prompt: str) -> bool | None:
    """Prompt for a yes/no answer; blank defaults to no."""
    while True:
        response = _prompt_text(prompt)
        if response is None:
            return None

        normalized = response.lower()
        if normalized == "":
            return False
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"n", "no"}:
            return False

        print("Please answer yes or no.")


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


def _checkpoint_available(path: str | None) -> bool:
    """Return whether ``path`` points to a non-empty checkpoint artifact."""
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def _path_text(path: object) -> str:
    """Return a printable path string or ``None`` for missing values."""
    if not isinstance(path, str) or path == "":
        return "None"
    return os.path.abspath(path)


def _print_active_model_status(config: MenuConfig) -> dict[str, object]:
    """Print the resolved active-model status and return it."""
    status = config.get_active_model_status()
    print("Active Model Status")
    print("=" * 30)
    print(f"Current Mode: {status['mode']}")
    print(f"Checkpoint Path: {_path_text(status.get('model_path'))}")
    print(f"Metrics Path: {_path_text(status.get('metrics_path'))}")
    print(f"Provenance Label: {status.get('dataset_label', 'Unavailable')}")
    print(f"Checkpoint Available: {status.get('checkpoint_exists', False)}")
    print(f"State File: {_path_text(status.get('state_path'))}")
    if status.get("note"):
        print(f"Note: {status['note']}")
    print("=" * 30)
    return status


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
        training_result = config.train_model(data_path)
        print("Training completed.")
        print("=" * 30)
        print(f"Dataset Used: {training_result.get('dataset_path', data_path)}")
        print(f"Latest Checkpoint Output: {training_result.get('model_output_path', config.latest_model_path)}")
        print(
            "Latest Metrics Output: "
            f"{training_result.get('metrics_output_path', config.latest_training_metrics_path)}"
        )
        print(
            "Latest Training Data Output: "
            f"{training_result.get('training_data_output_path', config.latest_training_data_path)}"
        )
        print(f"Validation Accuracy: {training_result.get('val_accuracy', 'Unavailable')}")
        print(
            "Majority Baseline Accuracy: "
            f"{training_result.get('majority_baseline_accuracy', 'Unavailable')}"
        )
        print(f"Baseline checkpoint remains at {config.model_path}.")
        print("Interactive and direct retraining still write only to the latest_* artifacts.")
        print("=" * 30)

        activate_latest = _prompt_yes_no(
            "Use this newly trained model for future simulations? [y/N]: "
        )
        if activate_latest is None:
            print("Active model selection cancelled; keeping the current setting.")
            return

        if activate_latest:
            status = config.set_active_model("latest")
            print("Latest trained model is now active for future simulations.")
            if status.get("note"):
                print(f"Note: {status['note']}")
        else:
            status = config.get_active_model_status()
            print(f"Active model remains set to {status['mode']}.")
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
    print("=" * 30)
    print("Latest training outputs")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(config.latest_model_path)}")
    print(f"Metrics: {os.path.abspath(config.latest_training_metrics_path)}")
    print(f"Training Data: {os.path.abspath(config.latest_training_data_path)}")
    print("Interactive and direct retraining write only to the latest_* artifacts.")
    print(f"Persistent active-model state: {os.path.abspath(config.active_model_state_path)}")
    print("=" * 30)


def _print_simulation_configuration(config: MenuConfig) -> None:
    """Print the current baseline simulation configuration."""
    status = config.get_active_model_status()
    print("Simulation Configuration")
    print("=" * 30)
    print("Default simulation source: synthetic balanced")
    print(f"Active model mode: {status['mode']}")
    print(f"Active checkpoint path: {_path_text(status.get('model_path'))}")
    print(f"Active model provenance: {status.get('dataset_label', 'Unavailable')}")
    print(f"Default execution log path: {os.path.abspath(config.log_path)}")
    print(f"Supported synthetic scenarios: {', '.join(config.scenario_choices)}")
    if status.get("note"):
        print(f"Note: {status['note']}")
    print("=" * 30)


def _print_artifact_status(config: MenuConfig) -> None:
    """Print existence and size details for important project artifacts."""
    artifact_names = [
        config.model_path,
        config.training_metrics_path,
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
    print("Workflow roles:")
    print("  - synthetic balanced = default simulation demo")
    print(f"  - {config.sample_dataset_path} = quick training demo")
    print(f"  - {config.huge_dataset_path} = larger packaged retraining demo")
    print("  - baseline model = packaged saved checkpoint from the dataset zip")
    print("Training and simulation are separate modes connected by the active model.")
    print(
        f"The active model persists across runs through {config.active_model_state_path} "
        "and can be baseline, latest, or none."
    )
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


def _choose_active_model_menu(config: MenuConfig) -> None:
    """Run the active-model selector submenu until the user goes back."""
    while True:
        latest_available = _checkpoint_available(config.latest_model_path)
        print("\nChoose Active Model")
        print("=" * 30)
        print(f"1. Use baseline model ({config.model_path})")
        latest_label = f"Use latest trained model ({config.latest_model_path})"
        if not latest_available:
            latest_label += " [Unavailable]"
        print(f"2. {latest_label}")
        print("3. Run without model")
        print("4. Back")

        choice = _prompt_text("Select an option (1-4): ")
        if choice is None or choice == "4":
            return

        if choice == "1":
            status = config.set_active_model("baseline")
            print("Baseline model selected.")
            if status.get("note"):
                print(f"Note: {status['note']}")
            return
        elif choice == "2":
            if not latest_available:
                print(f"Latest trained checkpoint not found at {config.latest_model_path}.")
                continue
            status = config.set_active_model("latest")
            print("Latest trained model selected.")
            if status.get("note"):
                print(f"Note: {status['note']}")
            return
        elif choice == "3":
            config.set_active_model("none")
            print("Future simulations will run without a model.")
            return
        else:
            print("Invalid option. Please enter a number from 1 to 4.")


def _simulation_menu(config: MenuConfig) -> None:
    """Run the simulation submenu until the user chooses to go back."""
    while True:
        print("\nSimulation Menu")
        print("=" * 30)
        print("1. Run default synthetic demo (balanced)")
        print("2. Run synthetic scenario")
        print("3. Replay from dataset path")
        print("4. Choose active model")
        print("5. View current simulation configuration")
        print("6. Back")

        choice = _prompt_text("Select an option (1-6): ")
        if choice is None or choice == "6":
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
            _choose_active_model_menu(config)
        elif choice == "5":
            _print_simulation_configuration(config)
        else:
            print("Invalid option. Please enter a number from 1 to 6.")


def _training_menu(config: MenuConfig) -> None:
    """Run the training submenu until the user chooses to go back."""
    while True:
        print("\nTraining Menu")
        print("=" * 30)
        print(f"1. Quick retraining demo on {config.sample_dataset_path}")
        print(f"2. Retrain baseline-scale model on {config.huge_dataset_path}")
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
        print("1. View active model status")
        print("2. View baseline metrics")
        print("3. View latest metrics")
        print("4. View both baseline and latest metrics")
        print("5. View artifact status")
        print("6. View log summary")
        print("7. View dataset package contents")
        print("8. Back")

        choice = _prompt_text("Select an option (1-8): ")
        if choice is None or choice == "8":
            return

        if choice == "1":
            _print_active_model_status(config)
        elif choice == "2":
            print_baseline_training_metrics(config)
        elif choice == "3":
            print_latest_training_metrics(config)
        elif choice == "4":
            print_saved_training_metrics(config)
        elif choice == "5":
            _print_artifact_status(config)
        elif choice == "6":
            _print_log_summary(config)
        elif choice == "7":
            _print_dataset_package_contents()
        else:
            print("Invalid option. Please enter a number from 1 to 8.")


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


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'argparse', 'json', 'os', 'zipfile', 'collections.abc', 'dataclasses',
            'doctest', 'python_ta'
        ],
        'allowed-io': [
            '_print_metrics_file', 'print_saved_training_metrics', 'print_baseline_training_metrics',
            'print_latest_training_metrics', '_prompt_text', '_prompt_scenario', '_prompt_yes_no',
            '_prompt_replay_speed', '_prompt_dataset_path', '_print_packaged_dataset_hint',
            '_print_active_model_status', '_run_simulation_menu', '_run_training_menu',
            '_print_training_output_targets', '_print_simulation_configuration',
            '_print_artifact_status', '_print_log_summary', '_print_dataset_package_contents',
            '_print_help_reference', '_choose_active_model_menu', '_simulation_menu',
            '_training_menu', '_artifacts_menu', '_help_menu', 'interactive_menu'
        ],
        'max-line-length': 120
    })

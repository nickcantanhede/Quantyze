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
class MenuPaths:
    """Runtime file paths used by the interactive menu."""

    model_path: str
    training_metrics_path: str
    latest_model_path: str
    latest_training_metrics_path: str
    latest_training_data_path: str
    log_path: str
    active_model_state_path: str


@dataclass(frozen=True)
class MenuDatasets:
    """Packaged dataset names and supported synthetic scenarios."""

    sample_dataset_path: str
    huge_dataset_path: str
    lobster_sample_message_path: str
    scenario_choices: tuple[str, ...]


@dataclass(frozen=True)
class MenuCallbacks:
    """Callables used by the interactive menu to run project workflows."""

    run_simulation: Callable[[argparse.Namespace], None]
    train_model: Callable[[str], dict[str, object]]
    get_active_model_status: Callable[[], dict[str, object]]
    set_active_model: Callable[[str], dict[str, object]]


@dataclass(frozen=True)
class MenuConfig:
    """Configuration and callbacks used by the interactive menu."""

    paths: MenuPaths
    datasets: MenuDatasets
    callbacks: MenuCallbacks

    @property
    def model_path(self) -> str:
        """Return the packaged baseline checkpoint path."""
        return self.paths.model_path

    @property
    def training_metrics_path(self) -> str:
        """Return the packaged baseline metrics path."""
        return self.paths.training_metrics_path

    @property
    def latest_model_path(self) -> str:
        """Return the latest trained checkpoint path."""
        return self.paths.latest_model_path

    @property
    def latest_training_metrics_path(self) -> str:
        """Return the latest classifier metrics path."""
        return self.paths.latest_training_metrics_path

    @property
    def latest_training_data_path(self) -> str:
        """Return the latest exported training-data path."""
        return self.paths.latest_training_data_path

    @property
    def log_path(self) -> str:
        """Return the execution-log path."""
        return self.paths.log_path

    @property
    def active_model_state_path(self) -> str:
        """Return the persisted overlay-state path."""
        return self.paths.active_model_state_path

    @property
    def sample_dataset_path(self) -> str:
        """Return the packaged sample dataset path."""
        return self.datasets.sample_dataset_path

    @property
    def huge_dataset_path(self) -> str:
        """Return the packaged larger dataset path."""
        return self.datasets.huge_dataset_path

    @property
    def lobster_sample_message_path(self) -> str:
        """Return the packaged LOBSTER message-file path."""
        return self.datasets.lobster_sample_message_path

    @property
    def scenario_choices(self) -> tuple[str, ...]:
        """Return the supported synthetic scenario names."""
        return self.datasets.scenario_choices

    @property
    def run_simulation(self) -> Callable[[argparse.Namespace], None]:
        """Return the simulation callback."""
        return self.callbacks.run_simulation

    @property
    def train_model(self) -> Callable[[str], dict[str, object]]:
        """Return the training callback."""
        return self.callbacks.train_model

    @property
    def get_active_model_status(self) -> Callable[[], dict[str, object]]:
        """Return the active-overlay status callback."""
        return self.callbacks.get_active_model_status

    @property
    def set_active_model(self) -> Callable[[str], dict[str, object]]:
        """Return the active-overlay mutation callback."""
        return self.callbacks.set_active_model


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
        config.training_metrics_path, "Packaged Baseline Classifier Metrics"
    ) or displayed_any
    displayed_any = _print_metrics_file(
        config.latest_training_metrics_path, "Latest Classifier Metrics"
    ) or displayed_any

    if not displayed_any:
        print("No saved metrics found. Train a model first.")


def print_baseline_metrics(config: MenuConfig) -> None:
    """Print only the packaged baseline metrics artifact."""
    if not _print_metrics_file(
        config.training_metrics_path,
        "Packaged Baseline Classifier Metrics"
    ):
        print(f"Baseline metrics not found at {config.training_metrics_path}.")


def print_latest_training_metrics(config: MenuConfig) -> None:
    """Print only the newest training metrics artifact, if present."""
    if not _print_metrics_file(config.latest_training_metrics_path, "Latest Classifier Metrics"):
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

    return None  # Unreachable but included for PythonTA Check


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

    return None  # Unreachable but included for PythonTA Check


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

    raise AssertionError("Unreachable: replay speed prompt loop always returns.")


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
    """Return a namespace shaped like the CLI parser output.

    >>> args = _build_runtime_args(scenario='balanced', speed=2.0)
    >>> (args.scenario, args.speed, args.no_ui)
    ('balanced', 2.0, True)
    """
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
    print("Simulation Overlay Status")
    print("=" * 30)
    print(f"Current Overlay Mode: {status['mode']}")
    print(f"Overlay Checkpoint Path: {_path_text(status.get('model_path'))}")
    print(f"Overlay Metrics Path: {_path_text(status.get('metrics_path'))}")
    print(f"Overlay Provenance: {status.get('dataset_label', 'Unavailable')}")
    print(f"Overlay Checkpoint Available: {status.get('checkpoint_exists', False)}")
    print(f"Overlay State File: {_path_text(status.get('state_path'))}")
    if status.get("note"):
        print(f"Note: {status['note']}")
    print("=" * 30)
    return status


def _print_quick_ta_demo_intro() -> None:
    """Print the simulator-first framing for the guided TA demo."""
    print("Quick TA Demo")
    print("=" * 30)
    print("Quantyze is primarily a tree-based limit order book simulator.")
    print("Event replay and matching-engine metrics are the main result.")
    print("Event source and simulation overlay source are separate.")
    print("The classifier is optional and does not drive the market.")
    print("=" * 30)


def _print_quick_ta_demo_followup() -> None:
    """Print the interpretation block shown after the guided TA demo run."""
    print("Quick Demo Interpretation")
    print("=" * 30)
    print("The balanced scenario is the recommended first demonstration.")
    print("Simulation metrics describe the replay and matching engine.")
    print("The overlay metric is secondary and not a trading-strategy claim.")
    print("Next recommended actions:")
    print("  - Run low_liquidity or high_volatility from Simulation")
    print("  - Optionally train on sample_internal.csv from Training")
    print("=" * 30)


def _quick_ta_demo(config: MenuConfig) -> None:
    """Run the guided simulator-first TA demo path."""
    _print_quick_ta_demo_intro()
    _print_active_model_status(config)
    _run_simulation_menu(config, _build_runtime_args())
    _print_quick_ta_demo_followup()


def _run_simulation_menu(config: MenuConfig, args: argparse.Namespace) -> None:
    """Run one simulation from the interactive menu with friendly error handling."""
    try:
        config.run_simulation(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Simulation failed: {exc}")


def _run_training_menu(config: MenuConfig, data_path: str, packaged: bool = False) -> None:
    """Run the training flow from the interactive menu and report any failures."""
    if not os.path.exists(data_path):
        if packaged:
            _print_packaged_dataset_hint(data_path)
        else:
            print(f"Could not find dataset at {data_path}.")
        return

    if data_path == config.huge_dataset_path:
        print("Note: this packaged internal dataset is very large and may take a while to load and train.")
    elif data_path == config.lobster_sample_message_path:
        print("Note: this packaged LOBSTER dataset is very large and may take a while to load and build.")

    try:
        training_result = config.train_model(data_path)
        print("Training Mode: Classifier Outputs")
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
        print("Training writes only to the latest_* artifacts.")
        print("Training does not change replay behavior by itself.")
        print("Simulation only uses the new checkpoint if latest is activated.")
        print("=" * 30)

        activate_latest = _prompt_yes_no(
            "Use this newly trained checkpoint for future simulations? [y/N]: "
        )
        if activate_latest is None:
            print("Overlay selection cancelled; simulation will continue using the current overlay setting.")
            return

        if activate_latest:
            status = config.set_active_model("latest")
            print("Latest checkpoint activated for future simulations.")
            if status.get("note"):
                print(f"Note: {status['note']}")
        else:
            print("Simulation will continue using the current overlay setting.")
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Training failed: {exc}")


def _print_training_output_targets(config: MenuConfig) -> None:
    """Print baseline and latest training artifact destinations."""
    print("Packaged baseline classifier artifacts")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(config.model_path)}")
    print(f"Metrics: {os.path.abspath(config.training_metrics_path)}")
    print("=" * 30)
    print("Latest classifier outputs")
    print("=" * 30)
    print(f"Checkpoint: {os.path.abspath(config.latest_model_path)}")
    print(f"Metrics: {os.path.abspath(config.latest_training_metrics_path)}")
    print(f"Training Data: {os.path.abspath(config.latest_training_data_path)}")
    print("Interactive and direct retraining write only to the latest_* artifacts.")
    print(f"Persistent overlay state: {os.path.abspath(config.active_model_state_path)}")
    print("=" * 30)


def _print_sim_config(config: MenuConfig) -> None:
    """Print the current baseline simulation configuration."""
    status = config.get_active_model_status()
    print("Simulation Configuration")
    print("=" * 30)
    print("Default event source: synthetic balanced")
    print(f"Simulation overlay mode: {status['mode']}")
    print(f"Simulation overlay path: {_path_text(status.get('model_path'))}")
    print(f"Overlay provenance: {status.get('dataset_label', 'Unavailable')}")
    print(f"Default execution log path: {os.path.abspath(config.log_path)}")
    print(f"Supported synthetic scenarios: {', '.join(config.scenario_choices)}")
    print("Overlay role: optional classifier inference")
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


def _print_dataset_contents() -> None:
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
    print(f"  - {config.huge_dataset_path} = larger packaged retraining demo (large; slower to load)")
    print(
        f"  - {config.lobster_sample_message_path} = packaged raw LOBSTER message file "
        "(paired orderbook file included; large; slower to load)"
    )
    print("  - model.pt = packaged baseline checkpoint")
    print("Simulation is the main workflow; the classifier is an optional overlay.")
    print(
        f"The overlay choice persists across runs through {config.active_model_state_path} "
        "and can be baseline, latest, or none."
    )
    print("Recommended TA path:")
    print("  - Extract quantyze_datasets.zip")
    print("  - Run python3 main.py")
    print("  - Choose Quick TA Demo")
    print("  - Optionally open Training and choose the packaged AAPL LOBSTER message file")
    print("The browser UI is a planned extension and is not part of the TA grading path.")
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
    print("Train on packaged AAPL LOBSTER message file:")
    print(f"  Mac/Linux: python3 main.py --train --data {config.lobster_sample_message_path}")
    print(f"  Windows:   py -3 main.py --train --data {config.lobster_sample_message_path}")
    print("Train on a custom dataset:")
    print("  Mac/Linux: python3 main.py --train --data <csv_path>")
    print("  Windows:   py -3 main.py --train --data <csv_path>")
    print("Dataset package note:")
    print(
        f"  Extract {DATASET_PACKAGE_PATH} beside main.py before using packaged "
        "sample, huge, or AAPL LOBSTER training options."
    )
    print("Supported dataset types:")
    print("  - internal Quantyze CSV")
    print("  - raw LOBSTER message CSV; the paired orderbook path is inferred by filename")
    print("=" * 30)


def _choose_active_model_menu(config: MenuConfig) -> None:
    """Run the active-model selector submenu until the user goes back."""
    while True:
        latest_available = _checkpoint_available(config.latest_model_path)
        print("\nChoose Simulation Overlay")
        print("=" * 30)
        print(f"1. Use packaged baseline checkpoint ({config.model_path})")
        latest_label = f"Use latest trained checkpoint ({config.latest_model_path})"
        if not latest_available:
            latest_label += " [Unavailable]"
        print(f"2. {latest_label}")
        print("3. Run simulation without model overlay")
        print("4. Back")

        choice = _prompt_text("Select an option (1-4): ")
        if choice is None or choice == "4":
            return

        if choice == "1":
            status = config.set_active_model("baseline")
            print("Packaged baseline checkpoint selected.")
            if status.get("note"):
                print(f"Note: {status['note']}")
            return
        elif choice == "2":
            if not latest_available:
                print(f"Latest trained checkpoint not found at {config.latest_model_path}.")
                continue
            status = config.set_active_model("latest")
            print("Latest trained checkpoint selected.")
            if status.get("note"):
                print(f"Note: {status['note']}")
            return
        elif choice == "3":
            config.set_active_model("none")
            print("Future simulations will run without a model overlay.")
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
        print("4. Choose simulation overlay")
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
            _print_sim_config(config)
        else:
            print("Invalid option. Please enter a number from 1 to 6.")


def _training_menu(config: MenuConfig) -> None:
    """Run the training submenu until the user chooses to go back."""
    while True:
        print("\nTraining Menu")
        print("=" * 30)
        print(f"1. Quick retraining demo on {config.sample_dataset_path}")
        print(f"2. Retrain baseline-scale model on {config.huge_dataset_path} [large; slower to load]")
        print(
            "3. Train on packaged AAPL LOBSTER message file "
            f"({config.lobster_sample_message_path}) [large; slower to load]"
        )
        print("4. Train on a custom dataset path")
        print("5. View training output targets")
        print("6. Back")

        choice = _prompt_text("Select an option (1-6): ")
        if choice is None or choice == "6":
            return

        if choice == "1":
            _run_training_menu(config, config.sample_dataset_path, packaged=True)
        elif choice == "2":
            _run_training_menu(config, config.huge_dataset_path, packaged=True)
        elif choice == "3":
            _run_training_menu(config, config.lobster_sample_message_path, packaged=True)
        elif choice == "4":
            data_path = _prompt_dataset_path("train on")
            if data_path is None:
                continue
            _run_training_menu(config, data_path)
        elif choice == "5":
            _print_training_output_targets(config)
        else:
            print("Invalid option. Please enter a number from 1 to 6.")


def _artifacts_menu(config: MenuConfig) -> None:
    """Run the artifacts and metrics submenu until the user chooses to go back."""
    while True:
        print("\nArtifacts & Metrics Menu")
        print("=" * 30)
        print("1. View simulation overlay status")
        print("2. View packaged baseline classifier metrics")
        print("3. View latest classifier metrics")
        print("4. View both baseline and latest classifier metrics")
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
            print_baseline_metrics(config)
        elif choice == "3":
            print_latest_training_metrics(config)
        elif choice == "4":
            print_saved_training_metrics(config)
        elif choice == "5":
            _print_artifact_status(config)
        elif choice == "6":
            _print_log_summary(config)
        elif choice == "7":
            _print_dataset_contents()
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
        print("1. Quick TA Demo")
        print("2. Simulation")
        print("3. Training")
        print("4. Artifacts & Metrics")
        print("5. Help / Command Reference")
        print("6. Exit")

        choice = _prompt_text("Select an option (1-6): ")
        if choice is None or choice == "6":
            print("Exiting Quantyze.")
            return

        if choice == "1":
            _quick_ta_demo(config)
        elif choice == "2":
            _simulation_menu(config)
        elif choice == "3":
            _training_menu(config)
        elif choice == "4":
            _artifacts_menu(config)
        elif choice == "5":
            _help_menu(config)
        else:
            print("Invalid option. Please enter a number from 1 to 6.")


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'argparse', 'json', 'os', 'zipfile', 'collections.abc', 'dataclasses',
            'doctest', 'python_ta'
        ],
        'disable': ['E9998'],
        'max-line-length': 120
    })

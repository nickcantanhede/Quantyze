"""Quantyze main entry point.

Module Description
==================
This file puts together the browser UI, terminal menu, and
shared runtime modules.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from __future__ import annotations

from cli.terminal_menu import (
    MenuCallbacks,
    MenuConfig,
    MenuDatasets,
    MenuPaths,
    interactive_menu as run_interactive_menu,
)
from config import (
    ACTIVE_MODEL_STATE_PATH,
    LOG_PATH,
    LATEST_MODEL_PATH,
    LATEST_TRAINING_DATA_PATH,
    LATEST_TRAINING_METRICS_PATH,
    LOBSTER_SAMPLE_MESSAGE_PATH,
    MODEL_PATH,
    SAMPLE_DATASET_PATH,
    SCENARIO_CHOICES,
    TRAINING_METRICS_PATH,
    HUGE_DATASET_PATH,
    resolve_active_model_status,
    set_active_model_selection,
)
from runtime.simulation import run_simulation_from_config
from ml.training import train_model
from web.app import run_browser_ui


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
            run_simulation=run_simulation_from_config,
            train_model=train_model,
            get_active_model_status=resolve_active_model_status,
            set_active_model=set_active_model_selection,
        ),
    )


def main(run_ui: bool = True, port: int = 9000) -> None:
    """Run Quantyze.

    Running ``main.py`` in PyCharm launches the browser UI by default. Passing
    ``run_ui=False`` switches the entrypoint to the terminal menu without any
    command-line flag parsing.
    """
    if run_ui:
        run_browser_ui(port=port)
    else:
        run_interactive_menu(_build_menu_config())


if __name__ == "__main__":
    main()  # run with main(run_ui=False) for terminal view only.

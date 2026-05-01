"""Quantyze configuration.

Module Description
==================
This module contains the shared configuration values used across Quantyze,
including artifact paths, dataset presets, training constants, formatting
helpers, and active-model overlay-state logic.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from data.data_loader import DataLoader

CLASS_NAMES = ["buy", "sell", "hold"]
MODEL_PATH = "model.pt"
TRAINING_METRICS_PATH = "training_metrics.json"
LATEST_MODEL_PATH = "latest_model.pt"
LATEST_TRAINING_METRICS_PATH = "latest_training_metrics.json"
LATEST_TRAINING_DATA_PATH = "latest_training_data.csv"
LOG_PATH = "log.json"
ACTIVE_MODEL_STATE_PATH = "active_model.json"
DATASET_PACKAGE_PATH = "quantyze_datasets.zip"
INSTRUCTIONS_PATH = "instructions.txt"
SAMPLE_DATASET_PATH = "sample_internal.csv"
HUGE_DATASET_PATH = "huge_internal.csv"
LOBSTER_SAMPLE_MESSAGE_PATH = "aapl_lobster_2012-06-21_message_5level_sample.csv"
LOBSTER_SAMPLE_ORDERBOOK_PATH = "aapl_lobster_2012-06-21_orderbook_5level_sample.csv"
SCENARIO_CHOICES = ("balanced", "low_liquidity", "high_volatility")
ACTIVE_MODEL_MODES = ("baseline", "latest", "none")
BASELINE_MODEL_LABEL = "packaged baseline checkpoint"
DEFAULT_DEPTH_LEVELS = 10
MAX_DEPTH_LEVELS = 500
DEFAULT_TRADE_LIMIT = 200
MAX_TRADE_LIMIT = 10_000
TRAIN_SPLIT_RATIO = 0.8
TRAIN_SPLIT_SEED = 111
TRAIN_MODEL_SEED = 111
TRAIN_SHUFFLE_SEED = 111
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 3e-4
TRAIN_OPTIMIZER_NAME = "Adam"


def checkpoint_exists(path: str | None) -> bool:
    """Return whether ``path`` points to a non-empty checkpoint file."""
    return path is not None and os.path.exists(path) and os.path.getsize(path) > 0


def format_decimal(value: float) -> str:
    """Return ``value`` formatted to two decimal places."""
    return f"{value:.2f}"


def sha256_file(path: str | None) -> str | None:
    """Return the SHA-256 digest for ``path``, or ``None`` when unavailable."""
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_metric(value: object) -> str:
    """Return a short printable representation for numeric metrics."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_decimal(value)
    return str(value)


def format_optional_path(path: object) -> str:
    """Return an absolute path for strings, or ``None`` when not available."""
    if not isinstance(path, str) or path == "":
        return "None"
    return os.path.abspath(path)


def overlay_mode_text(mode: object) -> str:
    """Return a human-readable overlay mode label."""
    labels = {"baseline": "Baseline", "latest": "Latest", "none": "None"}
    if isinstance(mode, str):
        return labels.get(mode, mode.title())
    return str(mode)


def _default_active_model_mode() -> str:
    """Return the shipped default overlay mode for this environment."""
    return "baseline"


def _default_no_model_note() -> str:
    """Return the clearest note for runs without an available default checkpoint."""
    if os.path.exists(DATASET_PACKAGE_PATH):
        return (
            "No packaged checkpoint is available yet. Extract "
            f"{DATASET_PACKAGE_PATH} beside main.py to enable the baseline overlay."
        )
    return "No packaged checkpoint is available; running without a model."


def _baseline_unavailable_note() -> str:
    """Return the clearest note for an unavailable packaged baseline checkpoint."""
    if os.path.exists(DATASET_PACKAGE_PATH):
        return (
            "Baseline overlay is the default packaged option, but its checkpoint is "
            f"not extracted yet. Extract {DATASET_PACKAGE_PATH} beside main.py to enable it."
        )
    return "Baseline overlay is the default packaged option, but its checkpoint is unavailable."


def dataset_label_for_path(dataset_path: str | None) -> str:
    """Return a short human-readable label for a dataset path."""
    if not dataset_path:
        return "unknown"

    dataset_name = os.path.basename(dataset_path)
    if dataset_name in {SAMPLE_DATASET_PATH, HUGE_DATASET_PATH, LOBSTER_SAMPLE_MESSAGE_PATH}:
        return dataset_name
    return f"custom ({dataset_name})"


def dataset_presets() -> tuple[dict[str, object], ...]:
    """Return the shared dataset presets exposed by the backend."""
    return (
        {
            "id": "sample",
            "button_label": "Sample",
            "label": "Sample",
            "path": SAMPLE_DATASET_PATH,
            "meta": SAMPLE_DATASET_PATH,
            "slower": False,
        },
        {
            "id": "huge",
            "button_label": "Huge",
            "label": "Huge",
            "path": HUGE_DATASET_PATH,
            "meta": f"{HUGE_DATASET_PATH} (slower)",
            "slower": True,
        },
        {
            "id": "lobster",
            "button_label": "LOBSTER",
            "label": "LOBSTER",
            "path": LOBSTER_SAMPLE_MESSAGE_PATH,
            "meta": (
                f"{LOBSTER_SAMPLE_MESSAGE_PATH} + "
                f"{LOBSTER_SAMPLE_ORDERBOOK_PATH} (slower)"
            ),
            "slower": True,
        },
    )


def training_preset_map() -> dict[str, str]:
    """Return the backend training preset id -> dataset path mapping."""
    return {
        str(preset["id"]): str(preset["path"])
        for preset in dataset_presets()
    }


def artifact_paths() -> tuple[tuple[str, str], ...]:
    """Return the submission-facing artifact paths reported by the UI."""
    return (
        ("baseline_model", MODEL_PATH),
        ("baseline_metrics", TRAINING_METRICS_PATH),
        ("latest_model", LATEST_MODEL_PATH),
        ("latest_metrics", LATEST_TRAINING_METRICS_PATH),
        ("latest_training_data", LATEST_TRAINING_DATA_PATH),
        ("log", LOG_PATH),
        ("active_model_state", ACTIVE_MODEL_STATE_PATH),
        ("dataset_package", DATASET_PACKAGE_PATH),
        ("package_instructions", INSTRUCTIONS_PATH),
        ("sample_dataset", SAMPLE_DATASET_PATH),
        ("huge_dataset", HUGE_DATASET_PATH),
        ("lobster_message_dataset", LOBSTER_SAMPLE_MESSAGE_PATH),
        ("lobster_orderbook_dataset", LOBSTER_SAMPLE_ORDERBOOK_PATH),
    )


def ui_config_payload() -> dict[str, Any]:
    """Return browser UI configuration derived from backend constants."""
    train_split_percent = int(TRAIN_SPLIT_RATIO * 100)
    return {
        "simulation": {
            "default_scenario": "balanced",
            "csv_presets": list(dataset_presets()),
        },
        "training": {
            "presets": list(dataset_presets()),
            "feature_dim": len(DataLoader.feature_names),
            "label_horizon_events": DataLoader.label_horizon_events,
            "train_split_percent": train_split_percent,
            "val_split_percent": 100 - train_split_percent,
            "split_seed": TRAIN_SPLIT_SEED,
            "epochs": TRAIN_EPOCHS,
            "optimizer": TRAIN_OPTIMIZER_NAME,
            "learning_rate": TRAIN_LEARNING_RATE,
            "output_files": {
                "latest_model": LATEST_MODEL_PATH,
                "latest_metrics": LATEST_TRAINING_METRICS_PATH,
            },
        },
        "ui": {
            "charts_require_network": True,
            "charts_dependency": "Chart.js CDN",
        },
    }


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
    return dataset_label_for_path(dataset_path)


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
    """Return the saved active-model payload, or ``None`` if invalid."""
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


def _requested_mode_from_payload(
    raw_payload: dict[str, object] | None,
    default_mode: str,
) -> tuple[str, bool, str]:
    """Return the requested mode, explicit-selection flag, and any fallback note."""
    if raw_payload is None:
        return (
            default_mode,
            False,
            "Active model state was missing or invalid; using the packaged baseline model.",
        )

    explicit_selection = bool(raw_payload.get("user_selected"))
    if explicit_selection:
        return str(raw_payload["mode"]), True, ""

    if str(raw_payload["mode"]) == default_mode:
        return default_mode, False, ""

    return (
        default_mode,
        False,
        "No explicit overlay selection was found; using the packaged baseline model.",
    )


def _resolved_payload_for_mode(
    requested_mode: str,
    raw_payload: dict[str, object] | None,
    explicit_selection: bool,
) -> tuple[dict[str, object], str]:
    """Return the resolved payload and any mode-resolution note."""
    if requested_mode == "latest":
        if checkpoint_exists(LATEST_MODEL_PATH):
            return (
                _active_model_payload(
                    "latest",
                    _saved_dataset_label(raw_payload),
                    user_selected=explicit_selection,
                ),
                "",
            )
        if checkpoint_exists(MODEL_PATH):
            return (
                _active_model_payload("baseline", user_selected=explicit_selection),
                "Latest checkpoint was unavailable; fell back to the baseline model.",
            )
        return (
            _active_model_payload("baseline", user_selected=explicit_selection),
            (
                "Latest checkpoint was unavailable. "
                + _baseline_unavailable_note()
            ),
        )

    if requested_mode == "baseline":
        if checkpoint_exists(MODEL_PATH):
            return _active_model_payload("baseline", user_selected=explicit_selection), ""
        return (
            _active_model_payload("baseline", user_selected=explicit_selection),
            _baseline_unavailable_note(),
        )

    return _active_model_payload("none", user_selected=explicit_selection), ""


def resolve_active_model_status() -> dict[str, object]:
    """Return the resolved active-model status, applying safe fallbacks."""
    raw_payload = _load_active_model_payload()
    default_mode = _default_active_model_mode()
    requested_mode, explicit_selection, note = _requested_mode_from_payload(
        raw_payload,
        default_mode,
    )
    resolved_payload, resolution_note = _resolved_payload_for_mode(
        requested_mode,
        raw_payload,
        explicit_selection,
    )
    if resolution_note != "":
        note = resolution_note

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
        "checkpoint_exists": checkpoint_exists(model_path if isinstance(model_path, str) else None),
        "state_path": ACTIVE_MODEL_STATE_PATH,
        "note": note,
    }


def set_active_model_selection(mode: str) -> dict[str, object]:
    """Persist a requested mode and return the resolved active-model status."""
    save_active_model_selection(mode, user_selected=True)
    return resolve_active_model_status()


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'hashlib', 'json', 'os', 'pathlib', 'typing',
            'data.data_loader', 'doctest', 'python_ta'
        ],
        'allowed-io': [
            'sha256_file',
            '_dataset_label_from_metrics',
            '_load_active_model_payload',
            'save_active_model_selection'
        ],
        'max-line-length': 120
    })

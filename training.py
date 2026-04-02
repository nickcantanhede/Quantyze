"""Quantyze training helpers.

Module Description
==================
This module contains the classifier training and evaluation helpers used by
Quantyze. It prepares feature tensors, performs the deterministic train/validation
split, trains the neural-network classifier, and writes self-describing metric
artifacts for the packaged baseline and latest retrained checkpoint.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from data_loader import DataLoader
from neural_net import OrderBookNet, Trainer
from config import (
    CLASS_NAMES,
    LATEST_MODEL_PATH,
    LATEST_TRAINING_DATA_PATH,
    LATEST_TRAINING_METRICS_PATH,
    MODEL_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_MODEL_SEED,
    TRAIN_SHUFFLE_SEED,
    TRAIN_SPLIT_RATIO,
    TRAIN_SPLIT_SEED,
    TRAINING_METRICS_PATH,
    dataset_label_for_path,
    sha256_file,
)


@dataclass
class EvalStats:
    """Accumulate classifier evaluation counts over validation batches."""

    total: int
    correct: int
    true_counts: list[int]
    pred_counts: list[int]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class PreparedTrainingData:
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
class TrainingOutputPaths:
    """Filesystem destinations produced by one training run."""

    model_path: str
    metrics_path: str
    training_data_path: str


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


def _new_eval_stats() -> EvalStats:
    """Return zeroed evaluation counters for the classifier metrics pass."""
    return EvalStats(
        total=0,
        correct=0,
        true_counts=[0] * len(CLASS_NAMES),
        pred_counts=[0] * len(CLASS_NAMES),
        confusion_matrix=[[0] * len(CLASS_NAMES) for _ in CLASS_NAMES],
    )


def _update_eval_stats(
    stats: EvalStats,
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


def _per_class_recall(stats: EvalStats) -> list[float]:
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
) -> PreparedTrainingData:
    """Return normalized tensors, index splits, and normalization stats."""
    split_generator = torch.Generator().manual_seed(TRAIN_SPLIT_SEED)
    permutation = torch.randperm(len(label_tensor), generator=split_generator)
    train_size = int(TRAIN_SPLIT_RATIO * len(label_tensor))
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]
    feature_mean, feature_std = _compute_feature_normalization(
        raw_feature_tensor[train_indices]
    )
    normalized_features = (raw_feature_tensor - feature_mean) / feature_std
    return PreparedTrainingData(
        feature_dim=raw_feature_tensor.shape[1],
        normalized_features=normalized_features,
        label_tensor=label_tensor,
        train_indices=train_indices,
        val_indices=val_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _build_training_loaders(
    prepared_data: PreparedTrainingData
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
    train_generator = torch.Generator().manual_seed(TRAIN_SHUFFLE_SEED)
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = TorchDataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def _seed_training_run() -> None:
    """Seed the training path for reproducible checkpoints and metrics."""
    torch.manual_seed(TRAIN_MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_MODEL_SEED)


def _artifact_kind(output_paths: TrainingOutputPaths) -> str:
    """Return a stable label describing the artifact role for ``output_paths``."""
    if (
        output_paths.model_path == MODEL_PATH
        and output_paths.metrics_path == TRAINING_METRICS_PATH
    ):
        return "baseline"
    if (
        output_paths.model_path == LATEST_MODEL_PATH
        and output_paths.metrics_path == LATEST_TRAINING_METRICS_PATH
    ):
        return "latest"
    return "custom"


def _training_metadata(
    data_path: str,
    prepared_data: PreparedTrainingData,
    output_paths: TrainingOutputPaths,
) -> dict[str, object]:
    """Return self-describing metadata for one persisted training artifact."""
    dataset_path = Path(data_path)
    config_metadata = {
        "dataset_path": data_path,
        "dataset_label": dataset_label_for_path(data_path),
        "dataset_size_bytes": dataset_path.stat().st_size if dataset_path.exists() else None,
        "dataset_sha256": sha256_file(data_path),
        "feature_dim": prepared_data.feature_dim,
        "feature_names": list(DataLoader.feature_names),
        "label_horizon_events": DataLoader.label_horizon_events,
        "label_move_threshold": DataLoader.label_move_threshold,
        "train_split_seed": TRAIN_SPLIT_SEED,
        "model_seed": TRAIN_MODEL_SEED,
        "shuffle_seed": TRAIN_SHUFFLE_SEED,
        "batch_size": TRAIN_BATCH_SIZE,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": TRAIN_LEARNING_RATE,
        "train_examples": int(prepared_data.train_indices.numel()),
        "val_examples": int(prepared_data.val_indices.numel()),
        "total_examples": int(prepared_data.label_tensor.numel()),
    }
    metadata = {
        "artifact_kind": _artifact_kind(output_paths),
        **config_metadata,
        "model_output_path": output_paths.model_path,
        "metrics_output_path": output_paths.metrics_path,
        "training_data_output_path": output_paths.training_data_path,
    }
    fingerprint_source = json.dumps(config_metadata, sort_keys=True).encode("utf-8")
    metadata["training_config_fingerprint"] = hashlib.sha256(fingerprint_source).hexdigest()[:12]
    return metadata


def _build_training_metrics(
    trainer: Trainer,
    metrics: dict[str, object],
    data_path: str,
    prepared_data: PreparedTrainingData,
    output_paths: TrainingOutputPaths
) -> dict[str, object]:
    """Return the persisted training metrics payload."""
    metadata = _training_metadata(data_path, prepared_data, output_paths)
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
        **metadata,
    }


def train_model(
    data_path: str,
    model_path: str = LATEST_MODEL_PATH,
    metrics_path: str = LATEST_TRAINING_METRICS_PATH,
    training_data_path: str = LATEST_TRAINING_DATA_PATH
) -> dict[str, object]:
    """Train the optional OrderBookNet model using data from <data_path>."""
    output_paths = TrainingOutputPaths(
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
    _seed_training_run()
    model = OrderBookNet(feature_dim=prepared_data.feature_dim)
    trainer = Trainer(
        model,
        class_weights=_compute_class_weights(prepared_data.train_labels()),
        learning_rate=TRAIN_LEARNING_RATE,
    )
    trainer.fit(train_loader, val_loader, epochs=TRAIN_EPOCHS)
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
        prepared_data,
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


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'csv', 'hashlib', 'json', 'dataclasses', 'pathlib', 'typing',
            'torch', 'torch.utils.data', 'data_loader', 'neural_net',
            'config', 'doctest', 'python_ta'
        ],
        'allowed-io': ['_export_training_csv', 'train_model'],
        'max-line-length': 120
    })

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import Tensor

from hsd.common.labels import IGNORE_INDEX, NUM_CLASSES, NUM_SEVERITY_LEVELS, TARGET_NAMES

TARGET_THRESHOLD: float = 0.5


class MetricAccumulator:
    def __init__(self) -> None:
        self.class_logits: list[np.ndarray] = []
        self.severity_logits: list[np.ndarray] = []
        self.target_logits: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.severities: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.has_targets: list[np.ndarray] = []

    def update(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        self.class_logits.append(outputs["class_logits"].detach().float().cpu().numpy())
        self.severity_logits.append(outputs["severity_logits"].detach().float().cpu().numpy())
        self.target_logits.append(outputs["target_logits"].detach().float().cpu().numpy())
        self.labels.append(batch["label"].detach().cpu().numpy())
        self.severities.append(batch["severity"].detach().cpu().numpy())
        self.targets.append(batch["targets"].detach().cpu().numpy())
        self.has_targets.append(batch["has_targets"].detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        return compute_metrics(
            class_logits=np.concatenate(self.class_logits),
            severity_logits=np.concatenate(self.severity_logits),
            target_logits=np.concatenate(self.target_logits),
            labels=np.concatenate(self.labels),
            severities=np.concatenate(self.severities),
            targets=np.concatenate(self.targets),
            has_targets=np.concatenate(self.has_targets),
        )


def compute_metrics(
    class_logits: np.ndarray,
    severity_logits: np.ndarray,
    target_logits: np.ndarray,
    labels: np.ndarray,
    severities: np.ndarray,
    targets: np.ndarray,
    has_targets: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(compute_class_metrics(class_logits, labels))
    metrics.update(compute_severity_metrics(severity_logits, severities))
    metrics.update(compute_target_metrics(target_logits, targets, has_targets))
    return metrics


def compute_class_metrics(class_logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = class_logits.argmax(axis=1)
    label_ids = list(range(NUM_CLASSES))
    return {
        "val_accuracy_class": float(accuracy_score(labels, predictions)),
        "val_macro_f1_class": float(
            f1_score(labels, predictions, labels=label_ids, average="macro", zero_division=0)
        ),
        "val_weighted_f1_class": float(
            f1_score(labels, predictions, labels=label_ids, average="weighted", zero_division=0)
        ),
    }


def compute_severity_metrics(
    severity_logits: np.ndarray, severities: np.ndarray
) -> dict[str, float]:
    valid = severities != IGNORE_INDEX
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "val_accuracy_severity": 0.0,
            "val_macro_f1_severity": 0.0,
            "val_num_severity_rows": 0.0,
        }
    predictions = severity_logits[valid].argmax(axis=1)
    reference = severities[valid]
    label_ids = list(range(NUM_SEVERITY_LEVELS))
    return {
        "val_accuracy_severity": float(accuracy_score(reference, predictions)),
        "val_macro_f1_severity": float(
            f1_score(reference, predictions, labels=label_ids, average="macro", zero_division=0)
        ),
        "val_num_severity_rows": float(valid_count),
    }


def compute_target_metrics(
    target_logits: np.ndarray,
    targets: np.ndarray,
    has_targets: np.ndarray,
) -> dict[str, float]:
    mask = has_targets.astype(bool)
    row_count = int(mask.sum())
    metrics: dict[str, float] = {"val_num_target_rows": float(row_count)}
    if row_count == 0:
        metrics["val_macro_f1_target"] = 0.0
        metrics["val_macro_precision_target"] = 0.0
        metrics["val_macro_recall_target"] = 0.0
        metrics["val_subset_accuracy_target"] = 0.0
        for name in TARGET_NAMES:
            metrics[f"val_precision_target_{name}"] = 0.0
            metrics[f"val_recall_target_{name}"] = 0.0
            metrics[f"val_f1_target_{name}"] = 0.0
        return metrics

    probabilities = 1.0 / (1.0 + np.exp(-target_logits[mask]))
    predictions = (probabilities >= TARGET_THRESHOLD).astype(int)
    reference = targets[mask].astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        reference, predictions, average=None, zero_division=0
    )
    metrics["val_macro_f1_target"] = float(np.mean(f1))
    metrics["val_macro_precision_target"] = float(np.mean(precision))
    metrics["val_macro_recall_target"] = float(np.mean(recall))
    metrics["val_subset_accuracy_target"] = float(np.mean(np.all(predictions == reference, axis=1)))
    for index, name in enumerate(TARGET_NAMES):
        metrics[f"val_precision_target_{name}"] = float(precision[index])
        metrics[f"val_recall_target_{name}"] = float(recall[index])
        metrics[f"val_f1_target_{name}"] = float(f1[index])
    return metrics


def select_metric(metrics: dict[str, float], selection_metric: str) -> float:
    return metrics[selection_metric]


def tensor_to_numpy(tensor: Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()

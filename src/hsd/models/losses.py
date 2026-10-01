# paste losses.py above
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from hsd.common.labels import IGNORE_INDEX, NUM_CLASSES, NUM_TARGET_LABELS

SEVERITY_POSITIONS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)


def compute_class_weights(labels: Sequence[int], num_classes: int = NUM_CLASSES) -> Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    normalized = inverse_sqrt * num_classes / inverse_sqrt.sum()
    return torch.tensor(normalized, dtype=torch.float32)


def compute_target_label_weights(
    targets: np.ndarray,
    has_targets: np.ndarray,
    num_labels: int = NUM_TARGET_LABELS,
) -> Tensor:
    mask = np.asarray(has_targets, dtype=bool)
    masked_targets = np.asarray(targets)[mask]
    if masked_targets.shape[0] == 0:
        return torch.ones(num_labels, dtype=torch.float32)
    counts = masked_targets.sum(axis=0)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    normalized = inverse_sqrt * num_labels / inverse_sqrt.sum()
    return torch.tensor(normalized, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        class_weights: Tensor,
        target_label_weights: Tensor,
        lambda_severity: float,
        lambda_target: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights.clone().float())
        self.register_buffer("target_label_weights", target_label_weights.clone().float())
        self.register_buffer(
            "severity_positions", torch.tensor(SEVERITY_POSITIONS, dtype=torch.float32)
        )
        self.lambda_severity = lambda_severity
        self.lambda_target = lambda_target
        self.label_smoothing = label_smoothing

    def forward(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        class_loss = functional.cross_entropy(
            outputs["class_logits"].float(),
            batch["label"],
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        severity_loss = self._severity_loss(outputs["severity_logits"].float(), batch["severity"])
        target_loss = self._target_loss(
            outputs["target_logits"].float(),
            batch["targets"].float(),
            batch["has_targets"],
        )
        total = class_loss + self.lambda_severity * severity_loss + self.lambda_target * target_loss
        return {
            "total": total,
            "class": class_loss,
            "severity": severity_loss,
            "target": target_loss,
        }

    def _severity_loss(self, severity_logits: Tensor, severity: Tensor) -> Tensor:
        valid = severity != IGNORE_INDEX
        if not bool(valid.any()):
            return (severity_logits * 0.0).sum()
        probabilities = functional.softmax(severity_logits[valid], dim=1)
        expected_value = (probabilities * self.severity_positions).sum(dim=1)
        true_value = severity[valid].float()
        return functional.mse_loss(expected_value, true_value)

    def _target_loss(self, target_logits: Tensor, targets: Tensor, has_targets: Tensor) -> Tensor:
        per_label = functional.binary_cross_entropy_with_logits(
            target_logits, targets, reduction="none"
        )
        weighted_per_label = per_label * self.target_label_weights.unsqueeze(0)
        per_row = weighted_per_label.mean(dim=1)
        row_mask = has_targets.to(per_row.dtype)
        denominator = row_mask.sum().clamp(min=1.0)
        return (per_row * row_mask).sum() / denominator

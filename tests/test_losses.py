from __future__ import annotations

import torch

from hsd.models.losses import MultiTaskLoss, compute_class_weights


def build_loss(lambda_severity: float = 0.5, lambda_target: float = 0.2) -> MultiTaskLoss:
    weights = compute_class_weights([0] * 300 + [1] * 100 + [2] * 50 + [3] * 10)
    return MultiTaskLoss(
        class_weights=weights,
        lambda_severity=lambda_severity,
        lambda_target=lambda_target,
        label_smoothing=0.05,
    )


def build_batch(has_targets: list[bool], severity: list[int]) -> dict[str, torch.Tensor]:
    size = len(has_targets)
    generator = torch.Generator().manual_seed(0)
    return {
        "label": torch.tensor([index % 4 for index in range(size)]),
        "severity": torch.tensor(severity),
        "has_targets": torch.tensor(has_targets),
        "targets": torch.randint(0, 2, (size, 8), generator=generator).float(),
    }


def build_outputs(size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        "class_logits": torch.randn(size, 4, generator=generator, requires_grad=True),
        "severity_logits": torch.randn(size, 4, generator=generator, requires_grad=True),
        "target_logits": torch.randn(size, 8, generator=generator, requires_grad=True),
    }


def test_class_weights_decrease_with_frequency() -> None:
    weights = compute_class_weights([0] * 300 + [1] * 100 + [2] * 50 + [3] * 10)
    assert weights[3] > weights[2] > weights[1] > weights[0]
    assert abs(float(weights.mean()) - 1.0) < 1e-6


def test_jigsaw_rows_have_exactly_zero_target_loss() -> None:
    loss_fn = build_loss()
    batch = build_batch([False, False, False, False], [-100, 1, 2, 3])
    result = loss_fn(build_outputs(4), batch)
    assert result["target"].item() == 0.0


def test_jigsaw_rows_have_exactly_zero_target_gradient() -> None:
    loss_fn = build_loss()
    has_targets = [True, False, True, False, False, True]
    batch = build_batch(has_targets, [-100, 1, 2, 3, 1, 2])
    outputs = build_outputs(6)
    result = loss_fn(outputs, batch)
    result["target"].backward()
    gradient = outputs["target_logits"].grad
    mask = torch.tensor(has_targets)
    assert torch.all(gradient[~mask] == 0.0)
    assert torch.all(gradient[mask].abs().sum(dim=1) > 0.0)


def test_zero_valid_severity_returns_connected_zero() -> None:
    loss_fn = build_loss()
    batch = build_batch([True, False, True, False], [-100, -100, -100, -100])
    outputs = build_outputs(4)
    result = loss_fn(outputs, batch)
    assert result["severity"].item() == 0.0
    assert result["severity"].requires_grad
    result["total"].backward()
    assert outputs["class_logits"].grad is not None


def test_zero_has_targets_returns_connected_zero() -> None:
    loss_fn = build_loss()
    batch = build_batch([False, False], [1, 2])
    outputs = build_outputs(2)
    result = loss_fn(outputs, batch)
    assert result["target"].requires_grad
    result["total"].backward()
    assert torch.all(outputs["target_logits"].grad == 0.0)


def test_total_combines_terms_with_lambdas() -> None:
    loss_fn = build_loss(lambda_severity=0.5, lambda_target=0.2)
    batch = build_batch([True, True, False, True], [-100, 1, 2, 3])
    result = loss_fn(build_outputs(4), batch)
    expected = result["class"] + 0.5 * result["severity"] + 0.2 * result["target"]
    assert torch.allclose(result["total"], expected)

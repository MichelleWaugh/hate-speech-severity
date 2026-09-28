from __future__ import annotations

import json
import math
import random
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from hsd.common.logging import get_logger
from hsd.models.losses import MultiTaskLoss
from hsd.models.multitask import XLMRMultiTask
from hsd.training.checkpoint import (
    JsonlLogger,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from hsd.training.metrics import MetricAccumulator
from hsd.training.sampler import LengthGroupedSampler

logger = get_logger(__name__)

LOSS_KEYS: tuple[str, ...] = ("total", "class", "severity", "target")
PRECISION_CHOICES: tuple[str, ...] = ("auto", "bf16", "fp16", "fp32")
REQUIRED_SELECTION_METRIC: str = "val_macro_f1_class"


@dataclass
class TrainConfig:
    model_name: str
    max_length: int
    batch_size: int
    grad_accum_steps: int
    encoder_lr: float
    head_lr: float
    weight_decay: float
    layerwise_lr_decay: float
    epochs: int
    warmup_ratio: float
    max_grad_norm: float
    lambda_severity: float
    lambda_target: float
    target_head_detach: bool
    label_smoothing: float
    max_neither_ratio: float
    precision: str
    gradient_checkpointing: bool
    num_workers: int
    tokenize_num_proc: int
    seed: int
    selection_metric: str
    early_stopping_patience: int
    eval_every_steps: int
    log_every_steps: int = 10

    def __post_init__(self) -> None:
        self.encoder_lr = float(self.encoder_lr)
        self.head_lr = float(self.head_lr)
        self.weight_decay = float(self.weight_decay)
        self.layerwise_lr_decay = float(self.layerwise_lr_decay)
        self.warmup_ratio = float(self.warmup_ratio)
        self.max_grad_norm = float(self.max_grad_norm)
        self.lambda_severity = float(self.lambda_severity)
        self.lambda_target = float(self.lambda_target)
        self.label_smoothing = float(self.label_smoothing)
        self.max_neither_ratio = float(self.max_neither_ratio)
        if self.selection_metric != REQUIRED_SELECTION_METRIC:
            raise ValueError(f"selection_metric must be {REQUIRED_SELECTION_METRIC}")
        if self.precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1")


@dataclass(frozen=True)
class PrecisionPlan:
    name: str
    enabled: bool
    dtype: torch.dtype
    use_scaler: bool


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in raw.items() if key in known})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(name: str, device: torch.device) -> PrecisionPlan:
    if device.type != "cuda" or name == "fp32":
        return PrecisionPlan("fp32", False, torch.bfloat16, False)
    supports_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    if name == "bf16" or (name == "auto" and supports_bf16):
        return PrecisionPlan("bf16", True, torch.bfloat16, False)
    return PrecisionPlan("fp16", True, torch.float16, True)


def layer_depth(parameter_name: str, num_layers: int) -> int:
    if parameter_name.startswith("embeddings."):
        return num_layers
    if parameter_name.startswith("encoder.layer."):
        layer_index = int(parameter_name.split(".")[2])
        return num_layers - 1 - layer_index
    return 0


def weight_decay_for(parameter_name: str, weight_decay: float) -> float:
    if parameter_name.endswith("bias") or "LayerNorm" in parameter_name:
        return 0.0
    return weight_decay


def build_optimizer(model: XLMRMultiTask, config: TrainConfig) -> AdamW:
    num_layers = len(model.encoder.encoder.layer)
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        depth = layer_depth(name, num_layers)
        learning_rate = config.encoder_lr * (config.layerwise_lr_decay**depth)
        decay = weight_decay_for(name, config.weight_decay)
        grouped.setdefault((learning_rate, decay), []).append(parameter)
    head_modules = (model.projection, model.class_head, model.severity_head, model.target_head)
    for module in head_modules:
        for name, parameter in module.named_parameters():
            decay = weight_decay_for(name, config.weight_decay)
            grouped.setdefault((config.head_lr, decay), []).append(parameter)
    parameter_groups = [
        {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), parameters in grouped.items()
    ]
    return AdamW(parameter_groups, lr=config.head_lr)


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: XLMRMultiTask,
        loss_fn: MultiTaskLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        device: torch.device,
        resume: bool,
    ) -> None:
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.best_dir = output_dir / "best"
        self.last_path = output_dir / "last.ckpt"
        self.precision = resolve_precision(config.precision, device)
        self.scaler = torch.amp.GradScaler("cuda") if self.precision.use_scaler else None
        self.optimizer = build_optimizer(self.model, config)
        steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
        self.total_steps = steps_per_epoch * config.epochs
        warmup_steps = int(self.total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, self.total_steps
        )
        self.state = TrainingState()
        log_path = output_dir / "train_log.jsonl"
        self.resumed = resume and self.last_path.exists()
        if self.resumed:
            self.state = load_checkpoint(
                self.last_path, self.model, self.optimizer, self.scheduler, self.scaler, device
            )
            logger.info(f"resumed from {self.last_path} at epoch {self.state.epoch}")
        else:
            if resume:
                logger.info("no last.ckpt found, starting fresh")
            log_path.unlink(missing_ok=True)
        self.log = JsonlLogger(log_path)

    def autocast_context(self) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.precision.dtype,
            enabled=self.precision.enabled,
        )

    def run_metadata(self) -> dict[str, object]:
        return {
            "event": "run_start",
            "target_head_detach": self.config.target_head_detach,
            "precision": self.precision.name,
            "device": self.device.type,
            "total_steps": self.total_steps,
            "resumed": self.resumed,
            "config": asdict(self.config),
        }

    def train(self) -> TrainingState:
        self.log.write(self.run_metadata())
        for epoch in range(self.state.epoch, self.config.epochs):
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                break
            sampler = self.train_loader.sampler
            if isinstance(sampler, LengthGroupedSampler):
                sampler.set_epoch(epoch)
            train_summary, epoch_improved = self.train_epoch(epoch)
            metrics = self.evaluate()
            epoch_improved = self.update_best(metrics) or epoch_improved
            if epoch_improved:
                self.state.epochs_without_improvement = 0
            else:
                self.state.epochs_without_improvement += 1
            self.state.epoch = epoch + 1
            self.log.write(
                {
                    "event": "epoch",
                    "epoch": epoch + 1,
                    "step": self.state.step,
                    "target_head_detach": self.config.target_head_detach,
                    "best_metric": self.state.best_metric,
                    "epochs_without_improvement": self.state.epochs_without_improvement,
                    **train_summary,
                    **metrics,
                }
            )
            logger.info(
                f"epoch {epoch + 1} val_macro_f1_class={metrics['val_macro_f1_class']:.4f} "
                f"best={self.state.best_metric:.4f}"
            )
            save_checkpoint(
                self.last_path,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.state,
            )
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info("early stopping triggered")
                break
        self.log.write(
            {
                "event": "run_end",
                "target_head_detach": self.config.target_head_detach,
                "best_metric": self.state.best_metric,
                "step": self.state.step,
            }
        )
        return self.state

    def train_epoch(self, epoch: int) -> tuple[dict[str, float], bool]:
        self.model.train()
        accumulation = self.config.grad_accum_steps
        num_batches = len(self.train_loader)
        epoch_totals = {key: torch.zeros((), device=self.device) for key in LOSS_KEYS}
        window_totals = {key: torch.zeros((), device=self.device) for key in LOSS_KEYS}
        epoch_count = 0
        window_count = 0
        row_count = torch.zeros((), device=self.device)
        target_row_count = torch.zeros((), device=self.device)
        epoch_improved = False
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, raw_batch in enumerate(self.train_loader):
            batch = move_to_device(raw_batch, self.device)
            with self.autocast_context():
                outputs = self.model(batch["input_ids"], batch["attention_mask"])
            losses = self.loss_fn(outputs, batch)
            scaled_loss = losses["total"] / accumulation
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            for key in LOSS_KEYS:
                detached = losses[key].detach()
                epoch_totals[key] += detached
                window_totals[key] += detached
            epoch_count += 1
            window_count += 1
            row_count += batch["label"].numel()
            target_row_count += batch["has_targets"].sum()
            at_boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == num_batches
            if not at_boundary:
                continue
            self.optimizer_step()
            self.state.step += 1
            if self.state.step % self.config.log_every_steps == 0:
                self.log.write(
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": self.state.step,
                        "target_head_detach": self.config.target_head_detach,
                        "lr": self.scheduler.get_last_lr()[-1],
                        **{
                            f"train_loss_{key}": float(window_totals[key].item() / window_count)
                            for key in LOSS_KEYS
                        },
                    }
                )
                window_totals = {key: torch.zeros((), device=self.device) for key in LOSS_KEYS}
                window_count = 0
            eval_interval = self.config.eval_every_steps
            if eval_interval > 0 and self.state.step % eval_interval == 0:
                metrics = self.evaluate()
                improved = self.update_best(metrics)
                epoch_improved = epoch_improved or improved
                self.log.write(
                    {
                        "event": "eval",
                        "epoch": epoch + 1,
                        "step": self.state.step,
                        "target_head_detach": self.config.target_head_detach,
                        "improved": improved,
                        **metrics,
                    }
                )
                self.model.train()
        summary = {
            f"train_loss_{key}": float(epoch_totals[key].item() / max(epoch_count, 1))
            for key in LOSS_KEYS
        }
        summary["train_has_targets_share"] = float(
            target_row_count.item() / max(row_count.item(), 1.0)
        )
        summary["lr"] = self.scheduler.get_last_lr()[-1]
        return summary, epoch_improved

    def optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        accumulator = MetricAccumulator()
        loss_totals = {key: torch.zeros((), device=self.device) for key in LOSS_KEYS}
        batch_count = 0
        for raw_batch in self.val_loader:
            batch = move_to_device(raw_batch, self.device)
            with self.autocast_context():
                outputs = self.model(batch["input_ids"], batch["attention_mask"])
            losses = self.loss_fn(outputs, batch)
            for key in LOSS_KEYS:
                loss_totals[key] += losses[key].detach()
            accumulator.update(outputs, batch)
            batch_count += 1
        metrics = accumulator.compute()
        for key in LOSS_KEYS:
            metrics[f"val_loss_{key}"] = float(loss_totals[key].item() / max(batch_count, 1))
        return metrics

    def update_best(self, metrics: dict[str, float]) -> bool:
        value = metrics[self.config.selection_metric]
        if value <= self.state.best_metric:
            return False
        self.state.best_metric = value
        self.save_best()
        return True

    def save_best(self) -> None:
        self.model.save_pretrained(self.best_dir)
        (self.best_dir / "train_config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )

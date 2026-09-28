from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from hsd.common.labels import NUM_CLASSES, NUM_SEVERITY_LEVELS, NUM_TARGET_LABELS
from hsd.common.logging import get_logger
from hsd.models.losses import MultiTaskLoss, compute_class_weights
from hsd.models.multitask import ModelConfig, XLMRMultiTask
from hsd.training.datamodule import build_eval_loader, build_train_loader
from hsd.training.sampler import LengthGroupedSampler
from hsd.training.tokenization import TokenizationConfig, tokenize_dataset
from hsd.training.trainer import Trainer, TrainConfig, load_train_config, seed_everything

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--parquet-path", type=Path, default=Path("data/processed/dataset.parquet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/tokenized"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "eval_every_steps": args.eval_every_steps,
    }
    active = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **active)


def subsample(dataset: Dataset, limit: int, seed: int) -> Dataset:
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(min(limit, len(shuffled))))


def resampled_epoch_labels(train_loader: torch.utils.data.DataLoader, dataset: Dataset) -> np.ndarray:
    sampler = train_loader.sampler
    if not isinstance(sampler, LengthGroupedSampler) or sampler.resampler is None:
        raise TypeError("train loader must use LengthGroupedSampler with an EpochResampler")
    indices = sampler.resampler.resample(0)
    return np.asarray(dataset["label"])[indices]


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_train_config(args.config), args)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenization_config = TokenizationConfig(
        parquet_path=args.parquet_path,
        cache_dir=args.cache_dir,
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        num_proc=config.tokenize_num_proc,
    )
    datasets = tokenize_dataset(tokenization_config)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    if args.max_train_samples is not None:
        train_dataset = subsample(train_dataset, args.max_train_samples, config.seed)
        val_dataset = subsample(val_dataset, args.max_train_samples, config.seed)
    logger.info(f"train rows={len(train_dataset)} val rows={len(val_dataset)} device={device.type}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_loader = build_train_loader(
        train_dataset,
        tokenizer,
        config.batch_size,
        config.max_neither_ratio,
        config.seed,
        config.num_workers,
    )
    val_loader = build_eval_loader(
        val_dataset, tokenizer, config.batch_size * 2, config.num_workers
    )

    resampled_labels = resampled_epoch_labels(train_loader, train_dataset)
    class_weights = compute_class_weights(resampled_labels.tolist())
    logger.info(f"class weights={[round(float(w), 4) for w in class_weights]}")

    model_config = ModelConfig(
        model_name=config.model_name,
        hidden_dropout=0.1,
        target_head_detach=config.target_head_detach,
        num_classes=NUM_CLASSES,
        num_severity_levels=NUM_SEVERITY_LEVELS,
        num_target_labels=NUM_TARGET_LABELS,
    )
    model = XLMRMultiTask(model_config)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    loss_fn = MultiTaskLoss(
        class_weights=class_weights,
        lambda_severity=config.lambda_severity,
        lambda_target=config.lambda_target,
        label_smoothing=config.label_smoothing,
    )
    trainer = Trainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        device=device,
        resume=args.resume,
    )
    state = trainer.train()
    logger.info(f"finished: best val_macro_f1_class={state.best_metric:.4f} steps={state.step}")


if __name__ == "__main__":
    main()

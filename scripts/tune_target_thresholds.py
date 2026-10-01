from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from transformers import AutoTokenizer

from hsd.common.labels import TARGET_NAMES
from hsd.common.logging import get_logger
from hsd.models.multitask import XLMRMultiTask
from hsd.training.datamodule import build_eval_loader
from hsd.training.tokenization import TokenizationConfig, tokenize_dataset

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model/best"))
    parser.add_argument("--parquet-path", type=Path, default=Path("data/processed/dataset.parquet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/tokenized"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--tokenize-num-proc", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def collect_target_predictions(
    model: XLMRMultiTask,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            has_targets_mask = batch["has_targets"].cpu().numpy().astype(bool)
            if not has_targets_mask.any():
                continue
            outputs = model(batch["input_ids"], batch["attention_mask"])
            logits = outputs["target_logits"].cpu().numpy()[has_targets_mask]
            targets = batch["targets"].cpu().numpy()[has_targets_mask]
            all_logits.append(logits)
            all_targets.append(targets)
    return np.concatenate(all_logits), np.concatenate(all_targets)


def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if labels.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    if len(thresholds) == 0:
        return 0.5, 0.0
    return float(thresholds[best_index]), float(f1_scores[best_index])


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    model = XLMRMultiTask.from_pretrained(args.model_dir, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    tokenization_config = TokenizationConfig(
        parquet_path=args.parquet_path,
        cache_dir=args.cache_dir,
        tokenizer_name=model.config.model_name,
        max_length=args.max_length,
        num_proc=args.tokenize_num_proc,
    )
    datasets = tokenize_dataset(tokenization_config)
    val_loader = build_eval_loader(datasets["val"], tokenizer, args.batch_size, args.num_workers)

    logger.info("running inference on validation set")
    logits, targets = collect_target_predictions(model, val_loader, device)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    logger.info(f"scored {logits.shape[0]} rows with has_targets=True")
    print(f"{'label':<12}{'positives':<11}{'flat_0.5_f1':<13}{'best_thresh':<13}{'best_f1':<10}")
    for index, name in enumerate(TARGET_NAMES):
        label_probabilities = probabilities[:, index]
        label_targets = targets[:, index]
        positives = int(label_targets.sum())
        flat_predictions = (label_probabilities >= 0.5).astype(int)
        flat_f1 = f1_score(label_targets, flat_predictions, zero_division=0)
        best_threshold, best_f1 = find_best_threshold(label_probabilities, label_targets)
        print(f"{name:<12}{positives:<11}{flat_f1:<13.4f}{best_threshold:<13.4f}{best_f1:<10.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from hsd.common.labels import IGNORE_INDEX, SEVERITY_NAMES
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
    parser.add_argument("--max-eval-rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_severity_predictions(
    model: XLMRMultiTask,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    collected = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            if collected >= max_rows:
                break
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            valid_mask = (batch["severity"] != IGNORE_INDEX).cpu().numpy()
            if not valid_mask.any():
                continue
            outputs = model(batch["input_ids"], batch["attention_mask"])
            predictions = outputs["severity_logits"].argmax(dim=1).cpu().numpy()[valid_mask]
            labels = batch["severity"].cpu().numpy()[valid_mask]
            all_predictions.append(predictions)
            all_labels.append(labels)
            collected += len(labels)
    return np.concatenate(all_predictions)[:max_rows], np.concatenate(all_labels)[:max_rows]


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

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
    val_dataset = datasets["val"].shuffle(seed=args.seed)
    val_loader = build_eval_loader(val_dataset, tokenizer, args.batch_size, args.num_workers)

    logger.info(f"running inference on up to {args.max_eval_rows} validation rows")
    predictions, labels = collect_severity_predictions(model, val_loader, device, args.max_eval_rows)
    logger.info(f"scored {len(labels)} rows with valid severity")

    print("\nPer-class report:")
    print(classification_report(labels, predictions, target_names=SEVERITY_NAMES, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(SEVERITY_NAMES))))
    header = "".join(f"{name:>10}" for name in SEVERITY_NAMES)
    print(f"{'':>10}{header}")
    for row_index, row_name in enumerate(SEVERITY_NAMES):
        row_values = "".join(
            f"{matrix[row_index, col_index]:>10}" for col_index in range(len(SEVERITY_NAMES))
        )
        print(f"{row_name:>10}{row_values}")

    adjacent_errors = 0
    distant_errors = 0
    for true_index, predicted_index in zip(labels, predictions):
        if true_index == predicted_index:
            continue
        if abs(int(true_index) - int(predicted_index)) == 1:
            adjacent_errors += 1
        else:
            distant_errors += 1
    total_errors = adjacent_errors + distant_errors
    if total_errors > 0:
        print(f"\nadjacent-level errors: {adjacent_errors} ({adjacent_errors / total_errors:.1%})")
        print(f"distant-level errors: {distant_errors} ({distant_errors / total_errors:.1%})")


if __name__ == "__main__":
    main()

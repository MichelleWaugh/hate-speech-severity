# paste publish_model.py above
from __future__ import annotations

import argparse
import os
from pathlib import Path

from hsd.common.logging import get_logger
from hsd.models.hub import push_to_hub

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model/best"))
    parser.add_argument("--private", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        raise RuntimeError("HF_MODEL_REPO environment variable is not set")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model directory not found: {args.model_dir}")
    push_to_hub(args.model_dir, repo_id, private=args.private)
    logger.info(f"published {args.model_dir} to {repo_id}")


if __name__ == "__main__":
    main()

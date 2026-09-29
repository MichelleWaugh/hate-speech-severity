# paste hub.py above
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from hsd.common.logging import get_logger

logger = get_logger(__name__)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")
    return token


def push_to_hub(local_dir: Path, repo_id: str, private: bool = True) -> None:
    token = get_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, token=token)
    logger.info(f"pushed {local_dir} to {repo_id}")


def pull_from_hub(repo_id: str, local_dir: Path) -> Path:
    token = get_hf_token()
    downloaded_path = snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=token)
    logger.info(f"pulled {repo_id} into {downloaded_path}")
    return Path(downloaded_path)

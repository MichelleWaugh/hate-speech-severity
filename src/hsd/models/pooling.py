
from __future__ import annotations

import torch
from torch import Tensor, nn


class MeanPooling(nn.Module):
    def forward(self, token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

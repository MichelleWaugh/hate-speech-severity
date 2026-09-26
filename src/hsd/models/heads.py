
from __future__ import annotations

from torch import Tensor, nn


class SharedProjection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(pooled)))


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)

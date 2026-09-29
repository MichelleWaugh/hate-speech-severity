
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from hsd.common.labels import CLASS_NAMES, SEVERITY_NAMES, TARGET_NAMES
from hsd.models.heads import ClassificationHead, SharedProjection
from hsd.models.pooling import MeanPooling


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    hidden_dropout: float
    target_head_detach: bool
    num_classes: int
    num_severity_levels: int
    num_target_labels: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        return cls(**json.loads(text))


class XLMRMultiTask(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pooling = MeanPooling()
        self.projection = SharedProjection(hidden_size, config.hidden_dropout)
        self.class_head = ClassificationHead(hidden_size, config.num_classes)
        self.severity_head = ClassificationHead(hidden_size, config.num_severity_levels)
        self.target_head = ClassificationHead(hidden_size, config.num_target_labels)

    def gradient_checkpointing_enable(self) -> None:
        self.encoder.gradient_checkpointing_enable()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_output.last_hidden_state
        pooled = self.pooling(token_embeddings, attention_mask)
        shared_features = self.projection(pooled)

        class_logits = self.class_head(shared_features)
        severity_logits = self.severity_head(shared_features)

        target_input = pooled.detach() if self.config.target_head_detach else shared_features
        target_logits = self.target_head(target_input)

        return {
            "class_logits": class_logits,
            "severity_logits": severity_logits,
            "target_logits": target_logits,
        }

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "multitask.pt")
        (output_dir / "model_config.json").write_text(self.config.to_json())
        label_maps = {
            "class": CLASS_NAMES,
            "severity": SEVERITY_NAMES,
            "target": TARGET_NAMES,
        }
        (output_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2))
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: Path, device: torch.device | None = None) -> XLMRMultiTask:
        config = ModelConfig.from_json((model_dir / "model_config.json").read_text())
        model = cls(config)
        state_dict = torch.load(model_dir / "multitask.pt", map_location=device or "cpu")
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        model.eval()
        return model
